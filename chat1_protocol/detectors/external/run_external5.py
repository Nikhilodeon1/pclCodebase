"""Detector 5 external validation — the cases fixed in PREREGISTRATION.md.

  E1  injected positive : eICU split in two, arm 2's HCO3 ablated   -> expect flag
  E2  negative control  : eICU split in two, no ablation            -> expect silent
  E3  natural pair      : MIMIC-IV demo vs eICU demo                -> DESCRIPTIVE ONLY
  E4  regression guard  : PhysioNet Site A against itself           -> expect silent

E3 is never scored: we have no independent knowledge of whether the two demo
databases truly differ in recording practice, and deciding that from the
availability ratio would be circular, since that is Variant B's whole signal.

Both variants run on every case. The decision rule between them is fixed in the
pre-registration and is NOT re-derived here.

    PCL_TEST_MODE=1 python detectors/external/run_external5.py --seeds 5
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from detectors.harness import Case, confusion, fmt_matrix
from detectors.checks.check5_missingness_scale import (
    VARIANTS, run_check_on_stays, run_check, sample_files)
from detectors.external.pipeline5 import (EXTERNAL_KEYS, ablate, load_site,
                                          split_arrays, to_stays)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
ABLATION_LEVELS = [0.50, 0.80, 0.95]


def _run(name_a, A, name_b, B, rule, verbose):
    return run_check_on_stays(name_a, A, name_b, B, verbose=verbose, rule=rule,
                              keys=EXTERNAL_KEYS)


def cases_for_seed(eicu, mimic, seed, rule, physionet_n, verbose=False):
    """All four pre-registered cases at one seed, under one decision rule."""
    out = []
    arm1, arm2 = split_arrays(eicu, seed)

    # E1 — injected positive, one case per ablation level.
    for p in ABLATION_LEVELS:
        A = to_stays(arm1)
        B = to_stays(ablate(arm2, "HCO3", p=p, seed=seed))
        flagged, st = _run(f"eICU arm1", A, f"eICU arm2 (HCO3 ablated {int(p*100)}%)",
                           B, rule, verbose)
        out.append(Case(f"E1 ablation {int(p * 100)}%", flagged, True,
                        dict(st, seed=seed, level=p)))

    # E2 — negative control, identical recording practice on both arms.
    flagged, st = _run("eICU arm1", to_stays(arm1), "eICU arm2", to_stays(arm2),
                       rule, verbose)
    out.append(Case("E2 eICU split, no ablation", flagged, False,
                    dict(st, seed=seed)))

    # E4 — regression guard on the original dataset.
    fa, _ = sample_files(seed, physionet_n)
    half = len(fa) // 2
    flagged, st = run_check(f"PhysioNet A (half 1)", fa[:half],
                            f"PhysioNet A (half 2)", fa[half:],
                            verbose=verbose, rule=rule)
    out.append(Case("E4 PhysioNet A vs itself", flagged, False,
                    dict(st, seed=seed)))

    return out


def descriptive_e3(eicu, mimic, rule, verbose=False):
    """E3 is reported, never scored. Returns the statistics only."""
    flagged, st = _run("MIMIC-IV demo", to_stays(mimic), "eICU demo",
                       to_stays(eicu), rule, verbose)
    return flagged, st


def save_arrays(path, eicu, mimic):
    """Cache the loaded raw_ts arrays.

    Wired in BEFORE the first full-scale launch, not after discovering it was
    needed -- that mistake cost detector 1 a repeated multi-hour labelling pass.
    Loading full eICU alone means parsing ~146M rows of vitalPeriodic.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        eicu=np.stack(eicu) if eicu else np.empty((0,)),
        mimic=np.stack(mimic) if mimic else np.empty((0,)))
    print(f"raw_ts arrays cached -> {path}", flush=True)


def load_arrays(path):
    z = np.load(path, allow_pickle=False)
    return list(z["eicu"]), list(z["mimic"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--physionet-n", type=int, default=800,
                    help="stays per arm for the E4 regression guard")
    ap.add_argument("--save-arrays", default=None,
                    help="cache loaded raw_ts here so reruns skip the load")
    ap.add_argument("--arrays", default=None,
                    help="reuse a cache written by --save-arrays")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    print("=" * 78)
    print("DETECTOR 5 — external validation (pre-registered)")
    print("=" * 78)
    if args.arrays:
        print(f"loading cached raw_ts from {args.arrays} ...", flush=True)
        eicu, mimic = load_arrays(args.arrays)
    else:
        print("loading external sites ...", flush=True)
        eicu = load_site("eicu")
        mimic = load_site("mimic")
        if args.save_arrays:
            save_arrays(args.save_arrays, eicu, mimic)
    print(f"eICU {len(eicu)} stays | MIMIC-IV {len(mimic)} stays", flush=True)
    print(f"components: {EXTERNAL_KEYS} "
          "(oxygen term is Severinghaus, not the PhysioNet O2Sat-vs-SaO2 pair)")
    print(f"E4 uses {args.physionet_n} PhysioNet stays per arm, "
          f"{len(seeds)} seeds\n", flush=True)

    report = {}
    for vname, rule in sorted(VARIANTS.items()):
        print("=" * 78)
        print(f"VARIANT {vname}")
        print("=" * 78, flush=True)
        per_seed = []
        for s in seeds:
            per_seed.append(cases_for_seed(eicu, mimic, s, rule,
                                           args.physionet_n))
            print(f"  seed {s} done", flush=True)

        by_name = {}
        for cases in per_seed:
            for c in cases:
                by_name.setdefault(c.name, []).append(c)

        print()
        for name, cs in by_name.items():
            n_flag = sum(c.flagged for c in cs)
            ratios = [c.stats["max_avail_ratio"] for c in cs]
            gap_ratios = [c.stats["composition_gap_ratio"] for c in cs]
            print(f"  {name:<34} flagged {n_flag}/{len(cs)}  "
                  f"expected={cs[0].expected}  "
                  f"avail_ratio={np.mean(ratios):.1f}  "
                  f"gap_ratio={np.mean(gap_ratios):.3f}")

        flat = [c for cases in per_seed for c in cases]
        counts = confusion(flat)
        print("\n  " + fmt_matrix(vname, counts))
        report[vname] = {
            "counts": counts,
            "cases": {n: {"flag_rate": sum(c.flagged for c in cs) / len(cs),
                          "expected": cs[0].expected,
                          "mean_avail_ratio": float(np.mean(
                              [c.stats["max_avail_ratio"] for c in cs])),
                          "mean_gap_ratio": float(np.mean(
                              [c.stats["composition_gap_ratio"] for c in cs]))}
                      for n, cs in by_name.items()},
        }

        flagged, st = descriptive_e3(eicu, mimic, rule)
        report[vname]["E3_descriptive"] = {"flagged": bool(flagged), **st}
        print(f"\n  E3 (MIMIC vs eICU, NOT SCORED): flagged={flagged}  "
              f"avail_ratio={st['max_avail_ratio']:.1f}  "
              f"gap_ratio={st['composition_gap_ratio']:.3f}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "external5.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 78)
    print("PRE-REGISTERED DECISION RULE")
    print("=" * 78)
    a, b = report["A_conjunction"], report["B_availability_only"]
    b_clean = all(b["cases"][n]["flag_rate"] == 0.0
                  for n in b["cases"] if not b["cases"][n]["expected"])
    print(f"  B false positives on E2/E4: "
          f"{'none' if b_clean else 'PRESENT — B is rejected'}")
    for p in ABLATION_LEVELS:
        n = f"E1 ablation {int(p * 100)}%"
        print(f"  {n:<24} A {a['cases'][n]['flag_rate']:.2f}   "
              f"B {b['cases'][n]['flag_rate']:.2f}")
    print("\nAdoption of B additionally requires detection at a strictly lower "
          "ablation level than A, consistent in sign across seeds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
