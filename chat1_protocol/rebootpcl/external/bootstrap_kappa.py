"""Bootstrap confidence intervals for detector 1's kappa on its control cases.

Motivation: on the MIMIC-IV DEMO (117 stays) the window-vs-single-point control gives kappa
0.651 against a 0.60 flag threshold -- a margin of 0.051 on an audit subset of
only 117 patients. "Wide CI, so more concerning" is not a usable statement. If
0.60 falls INSIDE the interval, the control is statistically indistinguishable
from a flag at that sample size, which is a far stronger claim than proximity.

Resampling is over PATIENTS with replacement, which is the unit the audit subset
samples and the unit kappa is computed across.

    PCL_TEST_MODE=1 python rebootpcl/external/bootstrap_kappa.py --iters 5000
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.checks.check1_label_shift import cohens_kappa, KAPPA_FLAG

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")


def bootstrap_kappa_ci(a, b, iters=5000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for Cohen's kappa between two label arrays.

    Returns (point, lo, hi, draws). Degenerate resamples -- ones where a
    resampled arm is constant, so kappa is undefined -- are dropped rather than
    coerced to 0 or 1, either of which would bias the interval.
    """
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    if len(a) != len(b):
        raise ValueError("label arrays must be the same length")
    n = len(a)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        k = cohens_kappa(a[idx], b[idx])
        if np.isfinite(k):
            draws.append(k)
    draws = np.asarray(draws)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(cohens_kappa(a, b)), float(lo), float(hi), draws


def audit_subset_distribution(a, b, audit_n, iters=5000, seed=0, alpha=0.05):
    """Kappa across repeated AUDIT SUBSET draws, matching how the detector runs.

    Detector 1 does not score the whole cohort: it draws an audit subset of
    min(500, n) patients WITHOUT replacement and computes kappa on that. So the
    uncertainty that actually governs its verdict is subset-draw variability at
    the operating size, not bootstrap variability over the full cohort. For a
    cohort at or below the audit size the two coincide.
    """
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    n = len(a)
    k_n = min(audit_n, n)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iters):
        idx = rng.choice(n, size=k_n, replace=False)
        k = cohens_kappa(a[idx], b[idx])
        if np.isfinite(k):
            draws.append(k)
    draws = np.asarray(draws)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), draws, k_n


def report(name, a, b, threshold, iters, seed, audit_n=500):
    point, lo, hi, draws = bootstrap_kappa_ci(a, b, iters=iters, seed=seed)
    crosses = (lo <= threshold <= hi)
    p_below = float(np.mean(draws <= threshold))
    print(f"\n{name}")
    print(f"  n={len(a)}  kappa={point:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  threshold {threshold} inside CI: {crosses}")
    print(f"  P(kappa <= {threshold}) across resamples = {p_below:.3f}"
          f"   <-- probability this control would FLAG")
    if crosses:
        print("  ==> statistically indistinguishable from a flag at this n")

    a_lo, a_hi, a_draws, k_n = audit_subset_distribution(
        a, b, audit_n, iters=iters, seed=seed)
    a_crosses = (a_lo <= threshold <= a_hi)
    a_p = float(np.mean(a_draws <= threshold))
    note = "same as cohort" if k_n >= len(a) else "detector operating point"
    print(f"  audit-subset draws (n={k_n}, {note}): "
          f"95% [{a_lo:.3f}, {a_hi:.3f}]  P(flag)={a_p:.3f}"
          f"{'  <-- CROSSES threshold' if a_crosses else ''}")

    return {"n": int(len(a)), "kappa": point, "ci_lo": lo, "ci_hi": hi,
            "threshold_inside_ci": bool(crosses), "p_would_flag": p_below,
            "audit_n": int(k_n), "audit_ci_lo": a_lo, "audit_ci_hi": a_hi,
            "audit_threshold_inside_ci": bool(a_crosses),
            "audit_p_would_flag": a_p, "iters": int(iters)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default=None,
                    help="npz written by run_external1.py --save-labels; skips "
                         "re-labelling, which costs hours on full data")
    args = ap.parse_args()

    from rebootpcl.external.run_external1 import (mimic_labels, eicu_labels,
                                                  load_labels)

    print("=" * 78)
    print("DETECTOR 1 — bootstrap CI on the control cases")
    print("=" * 78)
    print(f"{args.iters} resamples, flag threshold {KAPPA_FLAG}", flush=True)

    out = {}

    if args.labels:
        print(f"\nloading cached label arrays from {args.labels}", flush=True)
        (m_ids, m_icd, m_sofa), (e_ids, e_icd, e_sofa) = load_labels(args.labels)
        print(f"  MIMIC n={len(m_ids)}  eICU n={len(e_ids)}", flush=True)
    else:
        print(f"\nlabelling MIMIC-IV from "
              f"{os.environ.get('MIMIC_DIR', '<config default>')} ...", flush=True)
        m_ids, m_icd, m_sofa = mimic_labels(verbose=True)
        e_ids = e_icd = e_sofa = None
    base = "SOFA win[-48,+24]"
    for label in ("SOFA single-point", "SOFA win[-24,+12]", "SOFA win[-72,+24]"):
        out[f"MIMIC {base} vs {label}"] = report(
            f"MIMIC control: {base} vs {label}", m_sofa[base], m_sofa[label],
            KAPPA_FLAG, args.iters, args.seed)
    out["MIMIC POSITIVE ICD vs SOFA"] = report(
        f"MIMIC positive: ICD vs {base}", m_icd, m_sofa[base],
        KAPPA_FLAG, args.iters, args.seed)

    if e_sofa is None:
        print("\nlabelling eICU ...", flush=True)
        e_ids, e_icd, e_sofa = eicu_labels(verbose=True)
    for label in ("SOFA single-point", "SOFA win[-24,+12]", "SOFA win[-72,+24]"):
        out[f"eICU {base} vs {label}"] = report(
            f"eICU control: {base} vs {label}", e_sofa[base], e_sofa[label],
            KAPPA_FLAG, args.iters, args.seed)
    out["eICU POSITIVE ICD vs SOFA"] = report(
        f"eICU positive: ICD vs {base}", e_icd, e_sofa[base],
        KAPPA_FLAG, args.iters, args.seed)

    print("\n" + "=" * 78)
    risky = {k: v for k, v in out.items()
             if "POSITIVE" not in k
             and (v["threshold_inside_ci"] or v["audit_threshold_inside_ci"])}
    if risky:
        print("CONTROLS INDISTINGUISHABLE FROM A FLAG AT THEIR SAMPLE SIZE:")
        for k, v in risky.items():
            print(f"  {k}: kappa={v['kappa']:.3f} "
                  f"CI [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}] "
                  f"P(flag)={v['p_would_flag']:.3f}")
    else:
        print("every control's CI excludes the flag threshold")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "bootstrap_kappa.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
