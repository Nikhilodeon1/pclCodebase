import time
import logging
import numpy as np
import torch
from .variables import CANONICAL_VARIABLES, PLAUS, ABG_VARIABLES, HEMO_VARIABLES, VAR_TO_IDX

HOURS = 48
MAX_FFILL = 6
MIN_LOS_H = 24


def make_heartbeat(label, total=None, interval=20.0):
    """
    Returns a `beat(i, force=False)` callback that logs progress at most once
    every `interval` seconds. Keeps long silent loops (chunked CSV reads, per-
    patient assembly) emitting output so remote sessions aren't killed for
    inactivity and progress is visible.
    """
    state = {"t0": time.time(), "last": time.time()}

    def beat(i, force=False, extra=""):
        now = time.time()
        if force or (now - state["last"]) >= interval:
            frac = f"/{total}" if total else ""
            suffix = f" | {extra}" if extra else ""
            logging.info(f"  [{label}] {i}{frac} ... {now - state['t0']:.0f}s elapsed{suffix}")
            state["last"] = now

    return beat


def hourly_bin_median(timestamps_h, values, n_hours=HOURS):
    binned = np.full(n_hours, np.nan)
    if len(timestamps_h) == 0:
        return binned
    bins = {}
    for t, v in zip(timestamps_h, values):
        h = int(t)
        if 0 <= h < n_hours:
            bins.setdefault(h, []).append(v)
    for h, vals in bins.items():
        binned[h] = np.median(vals)
    return binned


def forward_fill(series, max_gap=MAX_FFILL):
    out = series.copy()
    last_val = np.nan
    gap = 0
    for i in range(len(out)):
        if not np.isnan(out[i]):
            last_val = out[i]
            gap = 0
        elif not np.isnan(last_val) and gap < max_gap:
            out[i] = last_val
            gap += 1
        else:
            gap += 1
    return out


def clip_plausible(values, var_name):
    lo, hi = PLAUS[var_name]
    return np.clip(values, lo, hi)


def build_observation_mask(ts):
    return ~np.isnan(ts)


def build_abg_mask(ts):
    abg_indices = [VAR_TO_IDX[v] for v in ABG_VARIABLES]
    abg_present = np.ones(ts.shape[0], dtype=bool)
    for idx in abg_indices:
        abg_present &= ~np.isnan(ts[:, idx])
    return abg_present


def build_constraint_mask(ts):
    idx = VAR_TO_IDX
    mask = np.zeros((ts.shape[0], 5), dtype=bool)
    for h in range(ts.shape[0]):
        mask[h, 0] = all(not np.isnan(ts[h, idx[v]]) for v in ["SBP", "DBP", "MAP"])
        mask[h, 1] = all(not np.isnan(ts[h, idx[v]]) for v in ["SBP", "DBP", "MAP"])
        mask[h, 2] = all(not np.isnan(ts[h, idx[v]]) for v in ["HR", "SBP"])
        mask[h, 3] = all(not np.isnan(ts[h, idx[v]]) for v in ABG_VARIABLES[:3])
        mask[h, 4] = all(not np.isnan(ts[h, idx[v]]) for v in ["SpO2", "PaO2"])
    return mask


def has_hemodynamic_window(ts):
    hemo_indices = [VAR_TO_IDX[v] for v in HEMO_VARIABLES]
    for h in range(ts.shape[0]):
        if all(not np.isnan(ts[h, idx]) for idx in hemo_indices):
            return True
    return False


class ZScoreNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, all_timeseries):
        stacked = []
        for ts in all_timeseries:
            mask = ~np.isnan(ts)
            stacked.append(ts)
        stacked = np.concatenate(stacked, axis=0)
        self.mean = np.nanmean(stacked, axis=0)
        self.std = np.nanstd(stacked, axis=0)
        self.std[self.std < 1e-8] = 1.0

    def transform(self, ts):
        return (ts - self.mean) / self.std

    def fit_transform(self, all_timeseries):
        self.fit(all_timeseries)
        return [self.transform(ts) for ts in all_timeseries]


class MinMaxNormalizer:
    def __init__(self):
        pass

    def transform(self, ts):
        normed = ts.copy()
        for col, var in enumerate(CANONICAL_VARIABLES):
            lo, hi = PLAUS[var]
            normed[:, col] = (normed[:, col] - lo) / (hi - lo + 1e-8)
        return normed

    def fit(self, all_timeseries):
        pass

    def fit_transform(self, all_timeseries):
        return [self.transform(ts) for ts in all_timeseries]


def preprocess_timeseries(raw_ts, normalizer=None):
    n_vars = len(CANONICAL_VARIABLES)
    ts = np.full((HOURS, n_vars), np.nan)

    for col in range(n_vars):
        if raw_ts.shape[1] > col:
            ts[:, col] = raw_ts[:min(HOURS, raw_ts.shape[0]), col]

    for col, var in enumerate(CANONICAL_VARIABLES):
        observed = ~np.isnan(ts[:, col])
        if observed.any():
            ts[observed, col] = clip_plausible(ts[observed, col], var)

    for col in range(n_vars):
        ts[:, col] = forward_fill(ts[:, col])

    obs_mask = build_observation_mask(ts)
    abg_mask = build_abg_mask(ts)
    c_mask = build_constraint_mask(ts)

    if normalizer is not None:
        ts_normed = normalizer.transform(ts)
    else:
        ts_normed = ts.copy()
        for col, var in enumerate(CANONICAL_VARIABLES):
            lo, hi = PLAUS[var]
            ts_normed[:, col] = (ts_normed[:, col] - lo) / (hi - lo + 1e-8)

    ts_filled = np.where(obs_mask, ts_normed, 0.0)

    return {
        "values": ts_filled,
        "mask": obs_mask,
        "abg_mask": abg_mask,
        "c_mask": c_mask,
    }
