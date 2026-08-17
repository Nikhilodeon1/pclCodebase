"""External testbed for detector 5: MIMIC-IV demo and eICU demo.

Detector 5 was built and tuned against PhysioNet 2019 raw PSV files. This module
puts the SAME diagnostic (`run_check_on_stays`) on data it was not built for.

Two things must be got right or the external result is meaningless:

  * Read `raw_ts` (loader called with keep_raw=True), never `sample["values"]`.
    The processed values are MinMax-normalized and forward-filled. Normalization
    breaks the physiological identities the residuals are built on, and
    forward-filling erases the pre-imputation missingness pattern that IS the
    signal detector 5 measures.

  * The oxygen term differs from the PhysioNet instantiation. On PhysioNet,
    detector 5's third component compares two saturation channels (O2Sat vs
    SaO2). The demo datasets record SpO2 and PaO2 but no SaO2, so the external
    oxygen term is the Severinghaus saturation/tension relation instead. The
    external component set is therefore analogous, not identical, and results
    are not a like-for-like replication of the PhysioNet component set.

Ground truth comes from controlled ablation (see detectors/PREREGISTRATION.md):
physiology is held fixed and only recording availability is changed, so the
expected verdict is known by construction rather than assumed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.data.variables import VAR_TO_IDX

# Per-term inputs, in canonical variable names.
TERMS = {
    "MAP":          ["SBP", "DBP", "MAP"],
    "HH":           ["pH", "HCO3", "pCO2"],
    "Severinghaus": ["SpO2", "PaO2"],
}
EXTERNAL_KEYS = ["MAP", "HH", "Severinghaus"]


def load_site(which, keep_raw=True):
    """('mimic'|'eicu') -> list of raw_ts arrays, one per stay."""
    from config import MIMIC_DIR, EICU_DIR
    if which == "mimic":
        from src.data.mimic4 import load_mimic4
        samples, _ = load_mimic4(MIMIC_DIR, fraction=1.0, seed=0,
                                 keep_raw=keep_raw)
    elif which == "eicu":
        from src.data.eicu import load_eicu
        samples, _ = load_eicu(EICU_DIR, fraction=1.0, seed=0,
                               keep_raw=keep_raw)
    else:
        raise ValueError(f"unknown site {which!r}")
    missing = [s for s in samples if "raw_ts" not in s]
    if missing:
        raise RuntimeError("loader returned no raw_ts; keep_raw was not honoured")
    return [np.asarray(s["raw_ts"], dtype=float) for s in samples]


def stay_components(ts):
    """Per-stay {term: mean residual}, in the same normalized units the
    PhysioNet version uses, over timesteps where the term is computable."""
    out = {}
    col = {v: VAR_TO_IDX[v] for v in VAR_TO_IDX}

    sbp, dbp, mp = ts[:, col["SBP"]], ts[:, col["DBP"]], ts[:, col["MAP"]]
    ok = ~np.isnan(sbp) & ~np.isnan(dbp) & ~np.isnan(mp)
    if ok.any():
        out["MAP"] = float(np.mean(
            np.abs((mp[ok] - (dbp[ok] + (sbp[ok] - dbp[ok]) / 3.0)) / 180.0)))

    ph, hco3, pco2 = ts[:, col["pH"]], ts[:, col["HCO3"]], ts[:, col["pCO2"]]
    ok = (~np.isnan(ph) & ~np.isnan(hco3) & ~np.isnan(pco2)
          & (hco3 > 0) & (pco2 > 0))
    if ok.any():
        pred = 6.1 + np.log10(hco3[ok] / (0.0307 * pco2[ok]))
        out["HH"] = float(np.mean(((ph[ok] - pred) / 1.4) ** 2))

    spo2, pao2 = ts[:, col["SpO2"]], ts[:, col["PaO2"]]
    ok = ~np.isnan(spo2) & ~np.isnan(pao2) & (pao2 > 0)
    if ok.any():
        # Severinghaus: predicted saturation from oxygen tension.
        inner = 23400.0 / (pao2[ok] ** 3 + 150.0 * pao2[ok]) + 1.0
        pred = 100.0 / inner
        out["Severinghaus"] = float(np.mean(((spo2[ok] - pred) / 50.0) ** 2))

    return out


def to_stays(arrays):
    """Per-stay component dicts, dropping stays where nothing is computable."""
    out = []
    for ts in arrays:
        c = stay_components(ts)
        if c:
            out.append(c)
    return out


def ablate(arrays, variable, p, seed):
    """Delete a fraction `p` of one variable's observations, per stay.

    Physiology is untouched -- the remaining values are the original ones. Only
    whether a value was RECORDED changes, which is precisely the confound
    detector 5 exists to catch, so the expected verdict is known by
    construction.
    """
    idx = VAR_TO_IDX[variable]
    rng = np.random.default_rng(seed)
    out = []
    for ts in arrays:
        a = ts.copy()
        obs = np.flatnonzero(~np.isnan(a[:, idx]))
        if len(obs):
            n_drop = int(round(p * len(obs)))
            if n_drop:
                a[rng.permutation(obs)[:n_drop], idx] = np.nan
        out.append(a)
    return out


def split_arrays(arrays, seed):
    """Two disjoint random halves."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(arrays))
    half = len(arrays) // 2
    return ([arrays[i] for i in perm[:half]],
            [arrays[i] for i in perm[half:]])
