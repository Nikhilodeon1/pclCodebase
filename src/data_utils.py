import logging
logging.basicConfig(level=logging.INFO)
"""
data_utils.py — backward-compatible shim.
New code should use src.data.* modules directly.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.data.variables import (
    CANONICAL_VARIABLES as VARIABLES,
    PLAUS,
    MIMIC_ITEM_IDS as ITEM_IDS,
    VAR_TO_IDX,
)

HOURS = 48
MAX_FFILL = 6
MIN_LOS_H = 24
MIN_AGE = 18


# ── STAGE 1 ───────────────────────────────────────────────────────────────────

def load_stays(data_path: str) -> pd.DataFrame:
    stays = pd.read_csv(f"{data_path}/icu/icustays.csv.gz", parse_dates=["intime","outtime"])
    patients = pd.read_csv(f"{data_path}/hosp/patients.csv.gz")
    admits = pd.read_csv(f"{data_path}/hosp/admissions.csv.gz")

    stays = stays.merge(patients[["subject_id","anchor_age"]], on="subject_id", how="left")
    stays = stays.merge(admits[["hadm_id","hospital_expire_flag"]], on="hadm_id", how="left")

    stays["los_h"] = stays["los"] * 24
    stays = stays[stays["anchor_age"] >= MIN_AGE]
    stays = stays[stays["los_h"] >= MIN_LOS_H]
    return stays.reset_index(drop=True)


# ── STAGE 2 ───────────────────────────────────────────────────────────────────

def load_chartevents(data_path: str, stay_ids: list) -> pd.DataFrame:
    charts = pd.read_csv(f"{data_path}/icu/chartevents.csv.gz", parse_dates=["charttime"])

    id2var = {}
    for var in ["SBP","DBP","MAP","HR","SpO2"]:
        for iid in ITEM_IDS[var]:
            id2var[iid] = var

    all_ids = [i for v in ["SBP","DBP","MAP","HR","SpO2"] for i in ITEM_IDS[v]]
    charts = charts[charts["itemid"].isin(all_ids)]
    charts = charts[charts["stay_id"].isin(stay_ids)]
    charts = charts[charts["valuenum"].notna()]
    charts["variable"] = charts["itemid"].map(id2var)
    return charts[["stay_id","charttime","variable","valuenum"]].copy()


def load_labevents(data_path: str, hadm_ids: list) -> pd.DataFrame:
    labs = pd.read_csv(f"{data_path}/hosp/labevents.csv.gz", parse_dates=["charttime"])

    id2var = {}
    for var in ["pH","HCO3","pCO2","PaO2"]:
        for iid in ITEM_IDS[var]:
            id2var[iid] = var

    all_ids = [i for v in ["pH","HCO3","pCO2","PaO2"] for i in ITEM_IDS[v]]
    labs = labs[labs["itemid"].isin(all_ids)]
    labs = labs[labs["hadm_id"].isin(hadm_ids)]
    labs = labs[labs["valuenum"].notna()]
    labs["variable"] = labs["itemid"].map(id2var)
    return labs[["hadm_id","charttime","variable","valuenum"]].copy()


# ── STAGE 3 ───────────────────────────────────────────────────────────────────

def build_timeseries(stay_row, charts_stay, labs_stay) -> np.ndarray:
    """
    Builds a (HOURS, V) time series for one ICU stay.

    Look-ahead bias prevention:
      - All events are binned by floor(delta_hours) relative to ICU intime.
      - Only events with 0 <= h < HOURS are included.
      - Lab events are joined via hadm_id and charttime, NOT discharge time,
        so a lab drawn at hour 10 is placed at hour 10 — never earlier.
      - Forward-fill (MAX_FFILL=6 hours) propagates the LAST KNOWN value
        forward in time, which is causal: at prediction time T, the model
        only sees values from T or earlier.
      - No backward-fill is applied — missing values before the first
        observation remain NaN (filled to 0 after masking).
    """
    intime = stay_row["intime"]
    ts = np.full((HOURS, len(VARIABLES)), np.nan)

    for _, row in charts_stay.iterrows():
        h = int((row["charttime"] - intime).total_seconds() // 3600)
        if 0 <= h < HOURS:
            col = VARIABLES.index(row["variable"])
            lo, hi = PLAUS[row["variable"]]
            val = np.clip(row["valuenum"], lo, hi)
            ts[h, col] = val if np.isnan(ts[h, col]) else (ts[h, col] + val) / 2

    for _, row in labs_stay.iterrows():
        h = int((row["charttime"] - intime).total_seconds() // 3600)
        if 0 <= h < HOURS:
            col = VARIABLES.index(row["variable"])
            lo, hi = PLAUS[row["variable"]]
            val = np.clip(row["valuenum"], lo, hi)
            if np.isnan(ts[h, col]):
                ts[h, col] = val

    # Causal forward-fill only (no backward-fill — that would be look-ahead bias)
    for col in range(ts.shape[1]):
        last_val, hours_since = np.nan, 0
        for h in range(HOURS):
            if not np.isnan(ts[h, col]):
                last_val, hours_since = ts[h, col], 0
            elif not np.isnan(last_val) and hours_since < MAX_FFILL:
                ts[h, col] = last_val
                hours_since += 1
            else:
                hours_since += 1
    return ts


# ── STAGE 4 ───────────────────────────────────────────────────────────────────

def compute_constraint_mask(ts: np.ndarray) -> np.ndarray:
    """Returns (HOURS, 5) bool array — True where constraint is computable."""
    idx = {v: i for i, v in enumerate(VARIABLES)}
    mask = np.zeros((HOURS, 5), dtype=bool)
    for h in range(HOURS):
        mask[h, 0] = not any(np.isnan(ts[h, idx[v]]) for v in ["SBP","DBP","MAP"])
        mask[h, 1] = not any(np.isnan(ts[h, idx[v]]) for v in ["SBP","DBP","MAP"])
        mask[h, 2] = not any(np.isnan(ts[h, idx[v]]) for v in ["HR", "SBP"])
        mask[h, 3] = not any(np.isnan(ts[h, idx[v]]) for v in ["pH","HCO3","pCO2"])
        mask[h, 4] = not any(np.isnan(ts[h, idx[v]]) for v in ["SpO2","PaO2"])
    return mask


# ── DATASET ───────────────────────────────────────────────────────────────────

class ICUDataset(Dataset):
    """
    Each item:
      x       : (HOURS, n_vars) float32  — normalized, NaN → 0
      mask    : (HOURS, n_vars) bool     — True where observed
      c_mask  : (HOURS, 5)     bool     — constraint availability
      stay_id : int
    """
    def __init__(self, data_path: str, stays: pd.DataFrame):
        self.samples = []
        self._build(data_path, stays)

    def _build(self, data_path, stays):
        charts = load_chartevents(data_path, stays["stay_id"].tolist())
        labs = load_labevents(data_path, stays["hadm_id"].tolist())

        self.constraint_coverage = np.zeros(5)
        n_skipped = 0

        for _, stay in stays.iterrows():
            sid = stay["stay_id"]
            hid = stay["hadm_id"]

            ts = build_timeseries(stay,
                                  charts[charts["stay_id"] == sid],
                                  labs[labs["hadm_id"] == hid])
            c_mask = compute_constraint_mask(ts)

            if np.all(np.isnan(ts)):
                n_skipped += 1
                continue

            self.constraint_coverage += c_mask.any(axis=0).astype(float)

            obs_mask = ~np.isnan(ts)
            ts_filled = np.where(obs_mask, ts, 0.0)

            for col, var in enumerate(VARIABLES):
                lo, hi = PLAUS[var]
                ts_filled[:, col] = (ts_filled[:, col] - lo) / (hi - lo + 1e-8)

            self.samples.append({
                "x": torch.tensor(ts_filled, dtype=torch.float32),
                "mask": torch.tensor(obs_mask, dtype=torch.bool),
                "c_mask": torch.tensor(c_mask, dtype=torch.bool),
                "stay_id": sid,
            })

        self.n_skipped = n_skipped

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

    def coverage_report(self):
        labels = ["MAP","PP (SBP-DBP)","Shock Index","Henderson-Hasselbalch","SpO2-PaO2"]
        total = max(len(self.samples), 1)
        logging.info(f"\nConstraint coverage ({total} stays):")
        for i, label in enumerate(labels):
            pct = 100 * self.constraint_coverage[i] / total
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            logging.info(f"{label:<25} [{bar}] {pct:.1f}%")
        logging.info(f"\nSkipped (no data): {self.n_skipped}")


def attach_ood_unit_label(dataset, stays):
    """
    Attaches a binary 'ood_unit' label to each sample.
      1 = cardiac/neuro care unit (the OOD split used in training)
      0 = general ICU care unit (in-distribution)

    Used for the linear probe invariance experiment (Experiment 4).
    The ideal result: ERM probe AUROC >> 0.5, PCL probe AUROC ≈ 0.5.
    This would mean PCL representations don't encode which unit the patient
    came from, while ERM representations do.
    """
    OOD_UNITS = {
        "Cardiac Vascular Intensive Care Unit (CVICU)",
        "Coronary Care Unit (CCU)",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)",
        "Neuro Stepdown",
    }
    unit_map = stays.set_index("stay_id")["first_careunit"].to_dict()
    n_ood = 0
    for sample in dataset.samples:
        unit = unit_map.get(sample["stay_id"], "")
        is_ood = 1.0 if unit in OOD_UNITS else 0.0
        sample["ood_unit"] = torch.tensor(is_ood, dtype=torch.float32)
        n_ood += int(is_ood)
    logging.info(f"OOD unit label attached: {n_ood}/{len(dataset.samples)} OOD, "
          f"{len(dataset.samples)-n_ood}/{len(dataset.samples)} in-dist")


def attach_labels(dataset, stays: pd.DataFrame):
    """
    Attaches task labels to existing dataset samples in-place.
    Call this after building ICUDataset.

    Adds to each sample:
      'mortality' : 1 if patient died in hospital, 0 otherwise
      'los_3d'    : 1 if ICU stay > 3 days, 0 otherwise

    Look-ahead note: hospital_expire_flag is a discharge-time label.
    It is used only as the prediction TARGET, not as an input feature,
    so it does not introduce look-ahead bias into the model inputs.
    """
    label_map = {}
    for _, row in stays.iterrows():
        label_map[row["stay_id"]] = {
            "mortality": int(row["hospital_expire_flag"]) if pd.notna(row["hospital_expire_flag"]) else 0,
            "los_3d": int(row["los_h"] > 72),
        }

    n_labeled = 0
    for sample in dataset.samples:
        sid = sample["stay_id"]
        if sid in label_map:
            sample["mortality"] = torch.tensor(label_map[sid]["mortality"], dtype=torch.float32)
            sample["los_3d"] = torch.tensor(label_map[sid]["los_3d"], dtype=torch.float32)
            n_labeled += 1
        else:
            sample["mortality"] = torch.tensor(0.0)
            sample["los_3d"] = torch.tensor(0.0)

    mort = sum(s["mortality"].item() for s in dataset.samples)
    los3d = sum(s["los_3d"].item() for s in dataset.samples)
    total = len(dataset.samples)
    logging.info(f"Labels attached to {n_labeled} stays")
    logging.info(f"Mortality positives : {int(mort)}/{total} ({100*mort/total:.1f}%)")
    logging.info(f"LOS >3d positives : {int(los3d)}/{total} ({100*los3d/total:.1f}%)")

def make_ood_loaders_by_unit(dataset, stays,
                             train_units=None,
                             ood_units=None,
                             batch_size=16):
    """
    Splits dataset by subject_id instead of stay_id to prevent data leakage.
    Train/Val on general ICUs, test on cardiac/neuro ICUs (OOD).
    """
    if train_units is None:
        train_units = [
            "Medical Intensive Care Unit (MICU)",
            "Surgical Intensive Care Unit (SICU)",
            "Medical/Surgical Intensive Care Unit (MICU/SICU)",
            "Trauma SICU (TSICU)",
        ]
    if ood_units is None:
        ood_units = [
            "Cardiac Vascular Intensive Care Unit (CVICU)",
            "Coronary Care Unit (CCU)",
            "Neuro Surgical Intensive Care Unit (Neuro SICU)",
            "Neuro Stepdown",
        ]

    stay_info = stays.set_index("stay_id")[["subject_id", "first_careunit"]].to_dict('index')

    train_unit_stays = stays[stays['first_careunit'].isin(train_units)]
    unique_train_subjects = train_unit_stays['subject_id'].unique()

    np.random.seed(42)
    np.random.shuffle(unique_train_subjects)
    n_val_subs = max(1, int(len(unique_train_subjects) * 0.2))
    
    val_subjects = set(unique_train_subjects[:n_val_subs])
    
    train_samples, val_samples, ood_samples = [], [], []

    for sample in dataset.samples:
        sid = sample["stay_id"]
        info = stay_info.get(sid)
        if not info:
            continue
            
        unit = info["first_careunit"]
        subj_id = info["subject_id"]

        if unit in ood_units:
            ood_samples.append(sample)
        elif unit in train_units:
            if subj_id in val_subjects:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

    class SubsetDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, i):
            return self.samples[i]

    train_ds = SubsetDataset(train_samples)
    val_ds = SubsetDataset(val_samples)
    ood_ds = SubsetDataset(ood_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False)

    logging.info(f"Split complete (Subject-level):")
    logging.info(f" - Train: {len(train_ds)} stays")
    logging.info(f" - Val:   {len(val_ds)} stays")
    logging.info(f" - OOD:   {len(ood_ds)} stays")
    
    return train_loader, val_loader, ood_loader


def make_stratified_loaders(dataset, task_name="mortality", train_frac=0.8,
                             batch_size=16, seed=42):
    """
    Stratified train/val split that preserves the positive-class ratio.

    This is critical for small demo cohorts: a random split can produce
    an all-negative validation set, causing AUROC to be undefined or
    stuck at 0.5 for every epoch. Stratified splitting guarantees at
    least one positive in each split as long as the dataset has >= 2
    positive samples.

    Falls back to random split if no labels are present.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch.utils.data import Subset

    try:
        labels = np.array([s[task_name].item() for s in dataset.samples])
    except KeyError:
        logging.warning(
            f"make_stratified_loaders: label '{task_name}' not found — "
            "falling back to random split. Call attach_labels() first."
        )
        n = len(dataset)
        n_train = int(n * train_frac)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        train_ds = Subset(dataset, idx[:n_train].tolist())
        val_ds = Subset(dataset, idx[n_train:].tolist())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=1.0 - train_frac, random_state=seed
    )
    idx = np.arange(len(labels))
    train_idx, val_idx = next(sss.split(idx, labels))

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())

    n_pos_train = int(labels[train_idx].sum())
    n_pos_val = int(labels[val_idx].sum())
    logging.info(
        f"Stratified split ({task_name}): "
        f"train={len(train_ds)} ({n_pos_train} pos), "
        f"val={len(val_ds)} ({n_pos_val} pos)"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
