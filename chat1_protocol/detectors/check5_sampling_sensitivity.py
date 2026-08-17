"""Is check 5's headline number a property of Site A, or of how it was sampled?

The reported result -- "composition explains 49% of the cross-site gap" -- was
produced with the stay list taken as `sorted(listdir)[:n]`, an alphabetical
prefix. PhysioNet filenames are patient IDs, so a prefix is not a random sample
of the site. Under a seeded random sample of the same size the ratio came out at
0.27, below the detector's own 0.30 threshold, and the positive case went silent.

This script measures both, so the difference is characterised rather than
assumed: the legacy prefix once, and several independent random samples.

    python detectors/check5_sampling_sensitivity.py --seeds 5 --n 1200
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.checks.check5_missingness_scale import run, COMP_GAP_RATIO_FLAG

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def positive(cases):
    return next(c for c in cases if c.expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=1200)
    args = ap.parse_args()

    print("=" * 78)
    print("CHECK 5 — sampling sensitivity of the composition gap ratio")
    print("=" * 78)
    print(f"n={args.n} stays per arm | flag threshold {COMP_GAP_RATIO_FLAG}\n")

    print("legacy alphabetical prefix (how the 0.49 headline was produced):",
          flush=True)
    leg = positive(run(seed=0, n=args.n, legacy=True))
    print(f"  gap_ratio={leg.stats['composition_gap_ratio']:.3f}  "
          f"avail_ratio={leg.stats['max_avail_ratio']:.1f}  "
          f"flagged={leg.flagged}\n", flush=True)

    print(f"seeded random samples ({args.seeds} seeds):", flush=True)
    shares, flags = [], []
    for s in range(args.seeds):
        c = positive(run(seed=s, n=args.n))
        shares.append(c.stats["composition_gap_ratio"])
        flags.append(bool(c.flagged))
        print(f"  seed {s}: gap_ratio={c.stats['composition_gap_ratio']:.3f}  "
              f"avail_ratio={c.stats['max_avail_ratio']:.1f}  "
              f"flagged={c.flagged}", flush=True)

    shares = np.array(shares)
    print("\n" + "-" * 78)
    print(f"legacy prefix     : {leg.stats['composition_gap_ratio']:.3f}")
    print(f"random mean +/- sd: {shares.mean():.3f} +/- "
          f"{shares.std(ddof=1) if len(shares) > 1 else 0.0:.3f}   "
          f"range [{shares.min():.3f}, {shares.max():.3f}]")
    print(f"positive flagged on {sum(flags)}/{len(flags)} random seeds")

    lo, hi = shares.min(), shares.max()
    outside = not (lo <= leg.stats["composition_gap_ratio"] <= hi)
    print("\nverdict:", (
        "the legacy number lies OUTSIDE the random-sampling range — it is a "
        "property of the alphabetical prefix, not of the site"
        if outside else
        "the legacy number lies inside the random-sampling range — the "
        "difference is sampling variance, not a prefix artifact"))
    if 0 < sum(flags) < len(flags):
        print("the positive case is SEED-UNSTABLE: it cannot be reported as a "
              "point estimate")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "check5_sampling.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"legacy_gap_ratio": leg.stats["composition_gap_ratio"],
                   "legacy_flagged": bool(leg.flagged),
                   "random_gap_ratio": [float(x) for x in shares],
                   "random_flagged": flags,
                   "n": args.n, "threshold": COMP_GAP_RATIO_FLAG}, fh, indent=2)


if __name__ == "__main__":
    main()
