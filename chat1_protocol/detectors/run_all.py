"""Reproduce the five-row confound-detector table.

    python rebootpcl/run_all.py                       # fast: static live, rest cached
    PCL_TEST_MODE=1 python rebootpcl/run_all.py --full   # re-run everything (~25 min)

Checks 3 and 4 are static analysis and take seconds. Checks 1, 2 and 5 read data
or train, so --fast reads their last recorded result from results/ rather than
re-running them. A check with no recorded result is reported as absent, never as
an all-zero matrix -- that would read as a run that found nothing.
"""
import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rebootpcl.harness import confusion, fmt_matrix, fmt_uncertainty

# Measured uncertainty per real-data detector, from the runs of record. Static
# detectors (3, 4) take None and render their reason instead. Sources:
#   check1  rebootpcl/bootstrap_kappa.out   (thinnest control, MIMIC n=117)
#   check2  rebootpcl/check2_5seed.out      (paired rel. delta at the 5% floor)
#   check5  rebootpcl/check5_sampling.out   (flag rate across random samples)
UNCERTAINTY = {
    # Full-scale (MIMIC-IV 3.1, n=74,829) at the detector's OPERATING POINT --
    # repeated 500-patient audit draws, not the whole cohort. The thinnest
    # legitimate-variation control, which is the number that matters for this
    # row. Demo-scale was [0.528, 0.776] P(flag)=0.216 at n=117; more data made
    # it worse, not better.
    1: {"ci_lo": 0.539, "ci_hi": 0.663, "p_would_flag": 0.484},
    2: {"mean": -0.031, "sd": 0.018},
    5: {"flag_rate": 0.4, "n": 5},
    3: None,
    4: None,
}

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

#     n: (label, module, slow)
CHECKS = {
    1: ("label-definition shift", "check1_label_shift", True),
    2: ("pretraining leakage", "check2_pretrain_leakage", True),
    3: ("OOD-contaminated selection", "check3_selection_audit", False),
    4: ("circular constraints", "check4_circularity", False),
    5: ("missingness/scale", "check5_missingness_scale", True),
}
SLOW = {n for n, (_, _, slow) in CHECKS.items() if slow}


def _mod(n):
    return importlib.import_module(f"rebootpcl.checks.{CHECKS[n][1]}")


def _cached(n):
    p = os.path.join(RESULTS, f"check{n}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)["counts"]


def _record(n, cases):
    os.makedirs(RESULTS, exist_ok=True)
    counts = confusion(cases)
    with open(os.path.join(RESULTS, f"check{n}.json"), "w", encoding="utf-8") as fh:
        json.dump({"counts": counts,
                   "cases": [dict(c._asdict()) for c in cases]}, fh, indent=2)
    return counts


def collect_fast():
    """Run the seconds-long static checks; load cached results for the rest."""
    rows = {}
    for n in CHECKS:
        rows[n] = _cached(n) if n in SLOW else _record(n, _mod(n).run(verbose=False))
    return rows


def collect_full(seed=0, n_stays=1200):
    rows = {}
    for n in sorted(CHECKS):
        label = CHECKS[n][0]
        print(f"\n{'=' * 78}\ncheck{n} — {label}\n{'=' * 78}", flush=True)
        m = _mod(n)
        if n in (3, 4):
            cases = m.run(verbose=True)
        elif n == 5:
            cases = m.run(seed=seed, n=n_stays, verbose=True)
        elif n == 2:
            cases = m.run(seed=seed, stays=900, epochs=3, seeds=5, verbose=True)
        else:
            cases = m.run(seed=seed, verbose=True)
        rows[n] = _record(n, cases)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="re-run every check instead of using cached results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stays", type=int, default=1200,
                    help="stays per arm for check 5")
    args = ap.parse_args()

    rows = collect_full(args.seed, args.stays) if args.full else collect_fast()

    print("\n" + "=" * 78)
    print("CONFOUND DETECTOR VALIDATION" +
          ("  (full re-run)" if args.full else "  (fast: 3,4 live; 1,2,5 cached)"))
    print("=" * 78)
    missing = []
    for n in sorted(CHECKS):
        counts = rows.get(n)
        if counts is None:
            missing.append(n)
            print(f"check{n}  {CHECKS[n][0]:<28}NO CACHED RESULT — run with --full")
            continue
        print(f"check{n}  " + fmt_matrix(CHECKS[n][0], counts))
        print(f"{'':8}{'':<28}{fmt_uncertainty(n, UNCERTAINTY.get(n))}")

    if missing:
        print(f"\nincomplete: checks {missing} have no recorded result. "
              "The table is NOT reproduced.")
        return 1
    total = {k: sum(r[k] for r in rows.values()) for k in ("TP", "FP", "FN", "TN")}
    print("\n" + fmt_matrix("TOTAL", total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
