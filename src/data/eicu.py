import os
import logging
import numpy as np
import pandas as pd
from .variables import (
    CANONICAL_VARIABLES, EICU_VITAL_MAPPING, EICU_VITAL_FALLBACK,
    EICU_LAB_MAPPING, VAR_TO_IDX, PLAUS,
)
from .preprocessing import (
    HOURS, MIN_LOS_H, preprocess_timeseries, has_hemodynamic_window,
    MinMaxNormalizer, make_heartbeat,
)
from .sofa_sepsis import eicu_sofa_sepsis_labels

logging.basicConfig(level=logging.INFO)

# Reverse lookup: eicu column name → canonical var
_PERIODIC_COLS  = {col: var for var, col in EICU_VITAL_MAPPING.items()}
_APERIODIC_COLS = {col: var for var, col in EICU_VITAL_FALLBACK.items()}
_LAB_NAME_MAP   = {name.lower(): var for var, name in EICU_LAB_MAPPING.items()}

# vitalPeriodic in full eICU-CRD is ~146M rows (~35GB uncompressed) — MUST be
# read in chunks or it OOMs the pod. Chunk-read every table and window to the
# first HOURS*60 minutes inside each chunk so accumulated memory stays bounded.
_EICU_CHUNK = 1_000_000
_MAX_MIN    = HOURS * 60  # keep only readings in the first 48h
# eICU offsets can be slightly negative (drawn just before unit admit). The
# original loader used int(offset/60), which truncates toward zero, mapping
# offsets in (-60, 0) to hour 0. Match that exactly so we don't silently drop
# admission-time ABG labs: keep offset > _MIN_OFFSET (i.e. > -60).
_MIN_OFFSET = -60


def _resolve_path(path):
    """Return path if it exists, else the .csv fallback, else raise."""
    if os.path.exists(path):
        return path
    alt = path.replace(".csv.gz", ".csv")
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f"Cannot find {path} or {alt}")


def _try_read(path):
    """Read a (small) table whole. Only used for patient.csv.gz."""
    return pd.read_csv(_resolve_path(path), low_memory=False, encoding_errors="replace")


def _read_vitals_chunked(path, col_map, stay_ids, offset_col="observationoffset"):
    """
    Chunk-read a wide vital-signs table, keep only our stays and the first 48h,
    melt to long format (patientunitstayid, hour, var, valuenum). Only reads the
    columns we actually need (huge speedup on the 35GB periodic table).
    """
    resolved = _resolve_path(path)
    header = pd.read_csv(resolved, nrows=0, encoding_errors="replace").columns
    present = [c for c in col_map if c in header]
    if not present:
        return pd.DataFrame(columns=["patientunitstayid", "hour", "var", "valuenum"])
    usecols = ["patientunitstayid", offset_col] + present

    parts = []
    scanned = n_chunks = 0
    beat = make_heartbeat(f"read {os.path.basename(path)}")
    for chunk in pd.read_csv(resolved, usecols=usecols, chunksize=_EICU_CHUNK,
                             encoding_errors="replace"):
        n_chunks += 1
        scanned += len(chunk)
        beat(n_chunks, extra=f"{scanned:,} rows scanned")
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        chunk = chunk[(chunk[offset_col] > _MIN_OFFSET) & (chunk[offset_col] < _MAX_MIN)]
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["hour"] = (chunk[offset_col] / 60).astype(int)
        melted = chunk.melt(
            id_vars=["patientunitstayid", "hour"], value_vars=present,
            var_name="eicu_col", value_name="valuenum",
        )
        melted = melted[melted["valuenum"].notna()]
        melted["var"] = melted["eicu_col"].map(col_map)
        parts.append(melted[["patientunitstayid", "hour", "var", "valuenum"]])

    if not parts:
        return pd.DataFrame(columns=["patientunitstayid", "hour", "var", "valuenum"])
    return pd.concat(parts, ignore_index=True)


def _read_labs_chunked(path, stay_ids):
    """Chunk-read lab.csv.gz, keep our stays + first 48h, map labname → canonical var."""
    resolved = _resolve_path(path)
    usecols = ["patientunitstayid", "labresultoffset", "labresult", "labname"]
    parts = []
    scanned = n_chunks = 0
    beat = make_heartbeat(f"read {os.path.basename(path)}")
    for chunk in pd.read_csv(resolved, usecols=usecols, chunksize=_EICU_CHUNK,
                             encoding_errors="replace"):
        n_chunks += 1
        scanned += len(chunk)
        beat(n_chunks, extra=f"{scanned:,} rows scanned")
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        chunk = chunk[chunk["labresultoffset"].notna() & chunk["labresult"].notna()]
        chunk = chunk[(chunk["labresultoffset"] > _MIN_OFFSET) & (chunk["labresultoffset"] < _MAX_MIN)]
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["hour"] = (chunk["labresultoffset"] / 60).astype(int)
        chunk["var"] = chunk["labname"].str.strip().str.lower().map(_LAB_NAME_MAP)
        chunk = chunk[chunk["var"].notna()]
        parts.append(
            chunk[["patientunitstayid", "hour", "var", "labresult"]]
            .rename(columns={"labresult": "valuenum"})
        )
    if not parts:
        return pd.DataFrame(columns=["patientunitstayid", "hour", "var", "valuenum"])
    return pd.concat(parts, ignore_index=True)


def _read_nursecharting_chunked(path, stay_ids):
    """Supplementary vitals from nurseCharting.

    eICU's vitalPeriodic carries temperature for only ~6% of hours, while
    nurseCharting holds the bulk of it (and already provides a clean
    'Temperature (C)' column, so no unit conversion is needed). Without this the
    Temp channel would be ~82% observed at the PhysioNet training site but ~6% at
    eICU — an availability gap that damages zero-shot transfer.
    """
    try:
        resolved = _resolve_path(path)
    except FileNotFoundError:
        logging.warning("eICU nurseCharting not found — Temp will stay sparse")
        return pd.DataFrame(columns=["patientunitstayid", "hour", "var", "valuenum"])

    wanted = {"temperature (c)": "Temp", "respiratory rate": "Resp"}
    usecols = ["patientunitstayid", "nursingchartoffset",
               "nursingchartcelltypevalname", "nursingchartvalue"]
    parts = []
    scanned = n_chunks = 0
    beat = make_heartbeat(f"read {os.path.basename(path)}")
    for chunk in pd.read_csv(resolved, usecols=usecols, chunksize=_EICU_CHUNK,
                             encoding_errors="replace", low_memory=False):
        n_chunks += 1
        scanned += len(chunk)
        beat(n_chunks, extra=f"{scanned:,} rows scanned")
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        chunk = chunk[(chunk["nursingchartoffset"] > _MIN_OFFSET)
                      & (chunk["nursingchartoffset"] < _MAX_MIN)]
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["var"] = (chunk["nursingchartcelltypevalname"].astype(str)
                        .str.strip().str.lower().map(wanted))
        chunk = chunk[chunk["var"].notna()]
        if chunk.empty:
            continue
        chunk["valuenum"] = pd.to_numeric(chunk["nursingchartvalue"], errors="coerce")
        chunk = chunk[chunk["valuenum"].notna()]
        chunk["hour"] = (chunk["nursingchartoffset"] / 60).astype(int)
        parts.append(chunk[["patientunitstayid", "hour", "var", "valuenum"]])

    if not parts:
        return pd.DataFrame(columns=["patientunitstayid", "hour", "var", "valuenum"])
    return pd.concat(parts, ignore_index=True)


def _build_all_timeseries(long_df, stay_ids, pid_to_idx):
    """
    Build (n_stays, HOURS, N_VARS) array from long-format event table.
    long_df columns: [patientunitstayid, hour, var, valuenum]
    """
    n_stays = len(stay_ids)
    n_vars  = len(CANONICAL_VARIABLES)
    all_ts  = np.full((n_stays, HOURS, n_vars), np.nan)

    if long_df.empty:
        return all_ts

    ev = long_df[long_df["patientunitstayid"].isin(stay_ids)].copy()
    ev = ev[(ev["hour"] >= 0) & (ev["hour"] < HOURS)]
    ev["col_idx"] = ev["var"].map(VAR_TO_IDX)
    ev = ev[ev["col_idx"].notna()]
    ev["col_idx"] = ev["col_idx"].astype(int)

    # Clip to plausible range
    for var, (lo, hi) in PLAUS.items():
        mask = ev["var"] == var
        ev.loc[mask, "valuenum"] = ev.loc[mask, "valuenum"].clip(lo, hi)

    ev["_arr_idx"] = ev["patientunitstayid"].map(pid_to_idx)
    ev = ev[ev["_arr_idx"].notna()]
    ev["_arr_idx"] = ev["_arr_idx"].astype(int)

    # Median per (arr_idx, hour, col_idx) — handles duplicate readings
    agg = (ev.groupby(["_arr_idx", "hour", "col_idx"])["valuenum"]
             .median()
             .reset_index())

    all_ts[agg["_arr_idx"].values, agg["hour"].values, agg["col_idx"].values] = agg["valuenum"].values
    return all_ts


def load_eicu(data_path, fraction=1.0, seed=42, keep_raw=False):
    patients = _try_read(os.path.join(data_path, "patient.csv.gz"))
    patients = patients[patients["age"].notna()]
    patients["age_numeric"] = pd.to_numeric(
        patients["age"].replace("> 89", 90), errors="coerce"
    )
    patients = patients[patients["age_numeric"] >= 18]
    patients["los_h"] = patients["unitdischargeoffset"] / 60.0
    patients = patients[patients["los_h"] >= MIN_LOS_H].reset_index(drop=True)

    if fraction < 1.0:
        n_keep = max(1, int(len(patients) * fraction))
        patients = patients.sample(n=n_keep, random_state=seed).reset_index(drop=True)

    stay_ids   = set(patients["patientunitstayid"].values)
    pid_to_idx = {pid: i for i, pid in enumerate(patients["patientunitstayid"].values)}
    # SOFA-based Sepsis-3 (A1): mode="single" (single-point) to HARMONIZE the
    # SOFA-eval mode with MIMIC (avoids reintroducing a per-dataset label-def
    # shift). Full-eICU single-point ~8.8% (in the 8-20% target). The window
    # variant (~12%) is reported as a sensitivity analysis, not the primary label.
    # eICU suspected infection = abx + (culture OR infection dx) since microLab is
    # sparse. Replaces ICD-code sepsis to remove the label-definition confound.
    sofa_labels, _ = eicu_sofa_sepsis_labels(data_path, patients, mode="single")
    logging.info(f"eICU: loading {len(stay_ids)} stays")

    # ── Vitals (chunked — vitalPeriodic is ~146M rows) ───────────────────────
    logging.info("eICU: reading vitalPeriodic (chunked)...")
    long_p = _read_vitals_chunked(
        os.path.join(data_path, "vitalPeriodic.csv.gz"), _PERIODIC_COLS, stay_ids)

    logging.info("eICU: reading vitalAperiodic (chunked)...")
    long_a = _read_vitals_chunked(
        os.path.join(data_path, "vitalAperiodic.csv.gz"), _APERIODIC_COLS, stay_ids)

    # ── Labs (chunked) ───────────────────────────────────────────────────────
    logging.info("eICU: reading lab (chunked)...")
    long_labs = _read_labs_chunked(os.path.join(data_path, "lab.csv.gz"), stay_ids)

    # ── Build arrays per source, then compose with fallback priority ─────────
    # Priority mirrors the original loader's clinical intent:
    #   periodic (invasive art-line) > aperiodic (non-invasive cuff) > labs.
    # Aperiodic BP only fills cells the invasive line left empty; labs occupy
    # disjoint (ABG) columns. Values within each source are median-binned per
    # (patient, hour, var), per the CLAUDE.md standardization spec.
    logging.info("eICU: reading nurseCharting (chunked)...")
    long_nc = _read_nursecharting_chunked(
        os.path.join(data_path, "nurseCharting.csv.gz"), stay_ids)

    ts_p = _build_all_timeseries(long_p,    stay_ids, pid_to_idx)  # HR/SBP/DBP/MAP/SpO2
    ts_a = _build_all_timeseries(long_a,    stay_ids, pid_to_idx)  # SBP/DBP/MAP (cuff)
    ts_l = _build_all_timeseries(long_labs, stay_ids, pid_to_idx)  # ABG + chem labs
    ts_n = _build_all_timeseries(long_nc,   stay_ids, pid_to_idx)  # Temp/Resp (nurse)

    all_ts = ts_p
    for src in (ts_a, ts_l, ts_n):
        fill = np.isnan(all_ts) & ~np.isnan(src)
        all_ts[fill] = src[fill]
    del long_p, long_a, long_labs, long_nc, ts_a, ts_l, ts_n

    # ── Filter and assemble samples ──────────────────────────────────────────
    all_raw_ts  = []
    all_samples = []
    skipped = {"empty": 0, "no_hemo": 0}

    beat = make_heartbeat("eICU assembly", total=len(patients))
    for k, (_, pat) in enumerate(patients.iterrows()):
        beat(k)
        pid    = pat["patientunitstayid"]
        raw_ts = all_ts[pid_to_idx[pid]]

        if np.all(np.isnan(raw_ts)):
            skipped["empty"] += 1
            continue
        if not has_hemodynamic_window(raw_ts):
            skipped["no_hemo"] += 1
            continue

        mortality  = int(pat.get("unitdischargestatus", "") == "Expired")
        los_3d     = int(pat["los_h"] > 72)
        sepsis     = int(sofa_labels.get(int(pid), 0))
        hospital_id = pat.get("hospitalid", 0)
        unit_type   = str(pat.get("unittype", "eICU_Unknown"))

        all_raw_ts.append(raw_ts)
        all_samples.append({
            "raw_ts":      raw_ts,
            "site_id":     3,
            "patient_id":  str(pid),
            "mortality":   mortality,
            "los_3d":      los_3d,
            "sepsis":      sepsis,
            "hospital_id": hospital_id,
            "unit_type":   unit_type,
        })

    logging.info(f"eICU: {len(all_samples)} stays loaded, skipped: {skipped}")

    normalizer = MinMaxNormalizer()
    normalizer.fit(all_raw_ts)

    processed = []
    beat = make_heartbeat("eICU preprocess", total=len(all_samples))
    for j, sample in enumerate(all_samples):
        beat(j)
        result = preprocess_timeseries(sample["raw_ts"], normalizer)
        entry = {
            "values":      result["values"],
            "mask":        result["mask"],
            "abg_mask":    result["abg_mask"],
            "c_mask":      result["c_mask"],
            # ICD-based sepsis is admission-level; broadcast the binary flag.
            "label":       np.full(HOURS, float(sample["sepsis"])),
            "site_id":     sample["site_id"],
            "patient_id":  sample["patient_id"],
            "mortality":   sample["mortality"],
            "los_3d":      sample["los_3d"],
            "hospital_id": sample.get("hospital_id", 0),
            "unit_type":   sample.get("unit_type", "eICU_Unknown"),
        }
        if keep_raw:
            entry["raw_ts"] = sample["raw_ts"]
        processed.append(entry)

    return processed, normalizer
