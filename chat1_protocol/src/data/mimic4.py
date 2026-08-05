import os
import logging
import numpy as np
import pandas as pd
from .variables import (
    CANONICAL_VARIABLES, MIMIC_ITEM_IDS, VAR_TO_IDX, PLAUS,
    MIMIC_EXPANDED_CHART, MIMIC_EXPANDED_LAB, MIMIC_UNIT_CONVERT,
)
from .preprocessing import (
    HOURS, MIN_LOS_H, preprocess_timeseries, has_hemodynamic_window,
    MinMaxNormalizer, make_heartbeat,
)
from .sofa_sepsis import mimic_sofa_sepsis_labels

logging.basicConfig(level=logging.INFO)

MIN_AGE = 18
_CHART_CHUNK = 500_000
_LAB_CHUNK   = 500_000

# Flat itemid → (canonical_var, col_idx) lookups built once at import time
_ITEMID_TO_VAR = {iid: var for var, iids in MIMIC_ITEM_IDS.items() for iid in iids}
_CHART_VARS = ["SBP", "DBP", "MAP", "HR", "SpO2"] + [
    v for v in MIMIC_EXPANDED_CHART if v in MIMIC_ITEM_IDS]
_LAB_VARS = ["pH", "HCO3", "pCO2", "PaO2"] + [
    v for v in MIMIC_EXPANDED_LAB if v in MIMIC_ITEM_IDS]
_ALL_CHART_IDS = set(iid for var in _CHART_VARS for iid in MIMIC_ITEM_IDS[var])
_ALL_LAB_IDS   = set(iid for var in _LAB_VARS   for iid in MIMIC_ITEM_IDS[var])


def load_stays(data_path):
    stays = pd.read_csv(
        os.path.join(data_path, "icu", "icustays.csv.gz"),
        parse_dates=["intime", "outtime"], encoding_errors="replace",
    )
    patients = pd.read_csv(os.path.join(data_path, "hosp", "patients.csv.gz"),
                           usecols=["subject_id", "anchor_age"], encoding_errors="replace")
    admits   = pd.read_csv(os.path.join(data_path, "hosp", "admissions.csv.gz"),
                           usecols=["hadm_id", "hospital_expire_flag"], encoding_errors="replace")

    stays = stays.merge(patients, on="subject_id", how="left")
    stays = stays.merge(admits,   on="hadm_id",    how="left")
    stays["los_h"] = stays["los"] * 24
    stays = stays[stays["anchor_age"] >= MIN_AGE]
    stays = stays[stays["los_h"] >= MIN_LOS_H]
    return stays.reset_index(drop=True)


def _read_filtered_chunks(path, id_col, keep_ids, keep_itemids, usecols, chunksize):
    """Read a large CSV in chunks, filtering to keep_ids and keep_itemids immediately."""
    chunks = []
    scanned = kept = n_chunks = 0
    beat = make_heartbeat(f"read {os.path.basename(path)}")
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols,
                              parse_dates=["charttime"], low_memory=False,
                              encoding_errors="replace"):
        n_chunks += 1
        scanned += len(chunk)
        chunk = chunk[chunk["itemid"].isin(keep_itemids)]
        chunk = chunk[chunk[id_col].isin(keep_ids)]
        chunk = chunk[chunk["valuenum"].notna()]
        kept += len(chunk)
        if len(chunk):
            chunks.append(chunk)
        beat(n_chunks, extra=f"{scanned:,} rows scanned, {kept:,} kept")
    beat(n_chunks, force=True, extra=f"{scanned:,} rows scanned, {kept:,} kept (done)")
    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)


def _build_all_timeseries(events_df, stays_df, id_col, n_stays):
    """
    Vectorized timeseries construction.

    events_df must have columns: [id_col, 'charttime', 'itemid', 'valuenum']
    stays_df  must have columns: [id_col, 'intime', '_arr_idx']
    Returns np.ndarray of shape (n_stays, HOURS, N_VARS), filled with NaN.
    """
    if events_df.empty:
        return np.full((n_stays, HOURS, len(CANONICAL_VARIABLES)), np.nan)

    ev = events_df.merge(stays_df[[id_col, "intime", "_arr_idx"]], on=id_col, how="inner")
    # Drop rows with null charttime — present in MIMIC-IV 3.1 labevents
    ev = ev[ev["charttime"].notna()]
    if ev.empty:
        return np.full((n_stays, HOURS, len(CANONICAL_VARIABLES)), np.nan)
    ev["hour"] = ((ev["charttime"] - ev["intime"]).dt.total_seconds() / 3600).astype(int)
    ev = ev[(ev["hour"] >= 0) & (ev["hour"] < HOURS)]
    ev["var"] = ev["itemid"].map(_ITEMID_TO_VAR)
    ev = ev[ev["var"].notna()]
    ev["col_idx"] = ev["var"].map(VAR_TO_IDX)

    # Unit harmonization BEFORE clipping (e.g. Fahrenheit temps -> Celsius).
    # Without this, F values would be clipped away by the Celsius bounds.
    for iid, fn in MIMIC_UNIT_CONVERT.items():
        m = ev["itemid"] == iid
        if m.any():
            ev.loc[m, "valuenum"] = fn(ev.loc[m, "valuenum"])

    # Clip to plausible range per variable
    for var, (lo, hi) in PLAUS.items():
        mask = ev["var"] == var
        ev.loc[mask, "valuenum"] = ev.loc[mask, "valuenum"].clip(lo, hi)

    # Median per (array_idx, hour, col_idx)
    agg = (ev.groupby(["_arr_idx", "hour", "col_idx"])["valuenum"]
             .median()
             .reset_index())

    all_ts = np.full((n_stays, HOURS, len(CANONICAL_VARIABLES)), np.nan)
    arr_idx = agg["_arr_idx"].astype(int).values
    hours   = agg["hour"].astype(int).values
    cols    = agg["col_idx"].astype(int).values
    vals    = agg["valuenum"].values
    all_ts[arr_idx, hours, cols] = vals
    return all_ts


def load_mimic4(data_path, fraction=1.0, seed=42, keep_raw=False):
    stays = load_stays(data_path)

    if fraction < 1.0:
        n_keep = max(1, int(len(stays) * fraction))
        stays = stays.sample(n=n_keep, random_state=seed).reset_index(drop=True)

    logging.info(f"MIMIC-IV: loading {len(stays)} stays")

    # SOFA-based Sepsis-3 (A1): mode="single" (single-point) -> ~31% on full
    # MIMIC, matching the clinical Sepsis-3 label used for PhysioNet. Replaces the
    # old ICD-code sepsis label to remove the label-definition confound.
    sofa_labels, _ = mimic_sofa_sepsis_labels(data_path, stays, mode="single")

    stay_id_set  = set(stays["stay_id"].values)
    hadm_id_set  = set(stays["hadm_id"].values)

    # Sequential integer index for numpy array construction
    stays = stays.reset_index(drop=True)
    stays["_arr_idx"] = stays.index

    # ── Chartevents (vitals) ─────────────────────────────────────────────────
    chart_path = os.path.join(data_path, "icu", "chartevents.csv.gz")
    logging.info("MIMIC-IV: reading chartevents (chunked)...")
    charts = _read_filtered_chunks(
        chart_path, "stay_id", stay_id_set, _ALL_CHART_IDS,
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        chunksize=_CHART_CHUNK,
    )

    # ── Labevents (ABG) ──────────────────────────────────────────────────────
    lab_path = os.path.join(data_path, "hosp", "labevents.csv.gz")
    logging.info("MIMIC-IV: reading labevents (chunked)...")
    # labevents uses hadm_id; rename charttime column if needed
    lab_cols = ["hadm_id", "itemid", "charttime", "valuenum"]
    labs = _read_filtered_chunks(
        lab_path, "hadm_id", hadm_id_set, _ALL_LAB_IDS,
        usecols=lab_cols,
        chunksize=_LAB_CHUNK,
    )
    # Map hadm_id → stay_id for uniform downstream processing
    hadm_to_stay = stays.set_index("hadm_id")["stay_id"].to_dict()
    if not labs.empty:
        labs["stay_id"] = labs["hadm_id"].map(hadm_to_stay)
        labs = labs.dropna(subset=["stay_id"])
        labs["stay_id"] = labs["stay_id"].astype(int)

    # ── Vectorized timeseries build ──────────────────────────────────────────
    stays_ref = stays[["stay_id", "intime", "_arr_idx"]]

    dfs_to_concat = [charts]
    if not labs.empty:
        dfs_to_concat.append(labs[["stay_id", "charttime", "itemid", "valuenum"]])
    all_events = pd.concat(dfs_to_concat, ignore_index=True)
    all_ts = _build_all_timeseries(all_events, stays_ref, "stay_id", len(stays))

    # ── Filter and assemble samples ──────────────────────────────────────────
    all_raw_ts  = []
    all_samples = []
    skipped = {"empty": 0, "no_hemo": 0}

    beat = make_heartbeat("MIMIC assembly", total=len(stays))
    for i, stay in stays.iterrows():
        beat(i)
        raw_ts = all_ts[stay["_arr_idx"]]

        if np.all(np.isnan(raw_ts)):
            skipped["empty"] += 1
            continue
        if not has_hemodynamic_window(raw_ts):
            skipped["no_hemo"] += 1
            continue

        mortality = int(stay["hospital_expire_flag"]) if pd.notna(stay["hospital_expire_flag"]) else 0
        los_3d    = int(stay["los_h"] > 72)
        sepsis    = int(sofa_labels.get(int(stay["stay_id"]), 0))

        all_raw_ts.append(raw_ts)
        all_samples.append({
            "raw_ts":        raw_ts,
            "site_id":       2,
            "stay_id":       stay["stay_id"],
            "subject_id":    stay["subject_id"],
            "mortality":     mortality,
            "los_3d":        los_3d,
            "sepsis":        sepsis,
            "first_careunit": stay.get("first_careunit", ""),
        })

    logging.info(f"MIMIC-IV: {len(all_samples)} stays loaded, skipped: {skipped}")

    normalizer = MinMaxNormalizer()
    normalizer.fit(all_raw_ts)

    processed = []
    beat = make_heartbeat("MIMIC preprocess", total=len(all_samples))
    for j, sample in enumerate(all_samples):
        beat(j)
        result = preprocess_timeseries(sample["raw_ts"], normalizer)
        entry = {
            "values":        result["values"],
            "mask":          result["mask"],
            "abg_mask":      result["abg_mask"],
            "c_mask":        result["c_mask"],
            # ICD-based sepsis is admission-level (no hourly timing), so broadcast
            # the binary flag across all hours; sepsis_binary = (label.sum() > 0).
            "label":         np.full(HOURS, float(sample["sepsis"])),
            "site_id":       sample["site_id"],
            "stay_id":       sample["stay_id"],
            "subject_id":    sample["subject_id"],
            "patient_id":    str(sample["stay_id"]),
            "mortality":     sample["mortality"],
            "los_3d":        sample["los_3d"],
            "first_careunit": sample["first_careunit"],
            "unit_type":     sample.get("first_careunit", "MIMIC_Unknown"),
        }
        if keep_raw:
            entry["raw_ts"] = sample["raw_ts"]
        processed.append(entry)

    return processed, normalizer
