"""Multi-seed stability for the stochastic detectors (1, 2, 5).

Checks 3 and 4 are deterministic static analysis over fixed source files, so a
seed sweep over them would produce a flag rate of exactly 0 or 1 with zero
variance by construction. That is not evidence of stability and they are
excluded rather than padded into the table.

    PCL_TEST_MODE=1 python detectors/run_seeds.py --seeds 5
"""
import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.harness import seed_sweep

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SEEDED_CHECKS = {
    1: "check1_label_shift",
    2: "check2_pretrain_leakage",
    5: "check5_missingness_scale",
}


def summarize(sweep):
    """Mark any case whose verdict is not unanimous across seeds as unstable.

    A detector whose verdict depends on the seed cannot support a point-estimate
    confusion matrix, which is the whole reason for this sweep. `flag_rate` is
    how often it fired; `agrees_with_truth_rate` is how often that was right,
    which differs for cases whose ground truth is False.
    """
    out = {}
    for name, d in sweep.items():
        out[name] = dict(d)
        out[name]["unstable"] = (0.0 < d["flag_rate"] < 1.0)
        out[name]["agrees_with_truth_rate"] = (
            d["flag_rate"] if d["expected"] else 1.0 - d["flag_rate"])
    return out


def sweep_check(num, seeds, n_stays, check2_seeds):
    m = importlib.import_module(f"detectors.checks.{SEEDED_CHECKS[num]}")
    if num == 5:
        return seed_sweep(lambda s: m.run(seed=s, n=n_stays), seeds)
    if num == 2:
        # check 2 is internally multi-seed; the outer seed shifts the whole seed
        # block, so this sweep measures whether the VERDICT reproduces rather
        # than whether one training run does.
        return seed_sweep(
            lambda s: m.run(seed=s, stays=900, epochs=3, seeds=check2_seeds),
            seeds)
    return seed_sweep(lambda s: m.run(seed=s), seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--stays", type=int, default=1200,
                    help="stays per arm for check 5")
    ap.add_argument("--check2-seeds", type=int, default=5,
                    help="inner seeds per check-2 run; the dominant cost")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="restrict to these check numbers")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    wanted = sorted(args.only) if args.only else sorted(SEEDED_CHECKS)

    all_out = {}
    for num in wanted:
        print(f"\n=== check{num} over {len(seeds)} seeds ===", flush=True)
        summary = summarize(
            sweep_check(num, seeds, args.stays, args.check2_seeds))
        all_out[f"check{num}"] = summary
        for name, d in summary.items():
            mark = "  <-- UNSTABLE" if d["unstable"] else ""
            print(f"  {name:<40} flag_rate={d['flag_rate']:.2f} "
                  f"({sum(d['flags'])}/{d['n']}) expected={d['expected']}{mark}",
                  flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "seeds.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            merged = json.load(fh)
        merged.update(all_out)      # keep results for checks not run this time
        all_out = merged
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(all_out, fh, indent=2)

    unstable = [(c, n) for c, s in all_out.items()
                for n, d in s.items() if d["unstable"]]
    print("\n" + "-" * 78)
    if unstable:
        print(f"UNSTABLE cases ({len(unstable)}):")
        for c, n in unstable:
            print(f"  {c}  {n}")
        print("These cannot be reported as point estimates. Report flag rate.")
    else:
        print("All cases unanimous across seeds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
