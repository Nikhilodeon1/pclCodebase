import os
import logging
import numpy as np
import pandas as pd
from .variables import CANONICAL_VARIABLES, PHYSIONET_MAPPING, VAR_TO_IDX
from .preprocessing import (
    HOURS, MIN_LOS_H, preprocess_timeseries, has_hemodynamic_window,
    MinMaxNormalizer, make_heartbeat,
)

logging.basicConfig(level=logging.INFO)

PHYSIONET_COLS_NEEDED = list(PHYSIONET_MAPPING.values()) + ["SepsisLabel", "ICULOS"]


def _find_psv_col(header, canonical_var):
    physionet_name = PHYSIONET_MAPPING[canonical_var]
    if physionet_name in header:
        return header.index(physionet_name)
    for i, h in enumerate(header):
        if h.lower() == physionet_name.lower():
            return i
    return None


def load_patient_psv(filepath):
    df = pd.read_csv(filepath, sep="|", encoding_errors="replace")
    return df


def extract_timeseries(df):
    n_rows = len(df)
    n_vars = len(CANONICAL_VARIABLES)
    ts = np.full((n_rows, n_vars), np.nan)

    for canonical, physionet_col in PHYSIONET_MAPPING.items():
        col_idx = VAR_TO_IDX[canonical]
        if physionet_col in df.columns:
            ts[:, col_idx] = df[physionet_col].values.astype(float)

    if "HCO3" not in df.columns and "BaseExcess" in df.columns:
        be = df["BaseExcess"].values.astype(float)
        valid = ~np.isnan(be)
        hco3_idx = VAR_TO_IDX["HCO3"]
        ts[valid, hco3_idx] = 24.0 + 0.5 * be[valid]

    if "SaO2" in df.columns:
        pao2_idx = VAR_TO_IDX["PaO2"]
        pao2_nans = np.isnan(ts[:, pao2_idx])
        sao2 = df["SaO2"].values.astype(float)
        valid = pao2_nans & ~np.isnan(sao2)
        sao2_valid = np.clip(sao2[valid], 1.0, 99.9) / 100.0
        inner = 23400.0 * sao2_valid / (1.0 - sao2_valid + 1e-8)
        coeff = np.cbrt(inner)
        ts[valid, pao2_idx] = np.clip(coeff, 20.0, 700.0)

    return ts


def get_sepsis_labels(df):
    if "SepsisLabel" in df.columns:
        return df["SepsisLabel"].values.astype(float)
    return np.zeros(len(df))


def get_site_id(filepath):
    if "training_setA" in filepath or "setA" in filepath:
        return 0
    elif "training_setB" in filepath or "setB" in filepath:
        return 1
    return -1


def load_physionet2019(data_dir, fraction=1.0, sites=None, seed=42, keep_raw=False):
    all_samples = []
    all_raw_ts = []

    site_dirs = []
    set_a = os.path.join(data_dir, "training_setA")
    set_b = os.path.join(data_dir, "training_setB")

    if sites is None or 0 in sites:
        if os.path.isdir(set_a):
            site_dirs.append((set_a, 0))
    if sites is None or 1 in sites:
        if os.path.isdir(set_b):
            site_dirs.append((set_b, 1))

    patient_files = []
    for site_dir, site_id in site_dirs:
        files = sorted([
            f for f in os.listdir(site_dir)
            if f.endswith(".psv")
        ])
        for f in files:
            patient_files.append((os.path.join(site_dir, f), site_id))

    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        n_keep = max(1, int(len(patient_files) * fraction))
        indices = rng.choice(len(patient_files), size=n_keep, replace=False)
        patient_files = [patient_files[i] for i in sorted(indices)]

    logging.info(f"PhysioNet 2019: loading {len(patient_files)} patients")

    skipped = {"short": 0, "empty": 0, "no_hemo": 0}

    beat = make_heartbeat("PhysioNet load", total=len(patient_files))
    for n_file, (filepath, site_id) in enumerate(patient_files):
        beat(n_file)
        df = load_patient_psv(filepath)

        if len(df) < MIN_LOS_H:
            skipped["short"] += 1
            continue

        raw_ts = extract_timeseries(df)

        if np.all(np.isnan(raw_ts)):
            skipped["empty"] += 1
            continue

        padded = np.full((HOURS, len(CANONICAL_VARIABLES)), np.nan)
        copy_len = min(HOURS, raw_ts.shape[0])
        padded[:copy_len, :] = raw_ts[:copy_len, :]

        if not has_hemodynamic_window(padded):
            skipped["no_hemo"] += 1
            continue

        sepsis_labels = get_sepsis_labels(df)
        sepsis_padded = np.zeros(HOURS)
        copy_len_labels = min(HOURS, len(sepsis_labels))
        sepsis_padded[:copy_len_labels] = sepsis_labels[:copy_len_labels]

        patient_id = os.path.basename(filepath).replace(".psv", "")

        mortality = 0
        los_h = len(df)
        los_3d = int(los_h > 72)

        u1 = df["Unit1"].iloc[0] if "Unit1" in df.columns else np.nan
        u2 = df["Unit2"].iloc[0] if "Unit2" in df.columns else np.nan
        if u1 == 1:
            unit_type = "PN_Unit1"
        elif u2 == 1:
            unit_type = "PN_Unit2"
        else:
            unit_type = "PN_Unknown"

        all_raw_ts.append(padded)
        all_samples.append({
            "raw_ts": padded,
            "sepsis_label": sepsis_padded,
            "site_id": site_id,
            "patient_id": patient_id,
            "mortality": mortality,
            "los_3d": los_3d,
            "unit_type": unit_type,
        })

    logging.info(
        f"PhysioNet 2019: {len(all_samples)} patients loaded, "
        f"skipped: {skipped}"
    )

    normalizer = MinMaxNormalizer()
    normalizer.fit(all_raw_ts)

    processed = []
    for sample in all_samples:
        result = preprocess_timeseries(sample["raw_ts"], normalizer)
        entry = {
            "values": result["values"],
            "mask": result["mask"],
            "abg_mask": result["abg_mask"],
            "c_mask": result["c_mask"],
            "label": sample["sepsis_label"],
            "site_id": sample["site_id"],
            "patient_id": sample["patient_id"],
            "mortality": sample["mortality"],
            "los_3d": sample["los_3d"],
            "unit_type": sample["unit_type"],
        }
        if keep_raw:
            entry["raw_ts"] = sample["raw_ts"]
        processed.append(entry)

    return processed, normalizer
