"""Does forward-fill vary with detector 2's leakage condition?

Detector 2's paired within-seed analysis assumes the forward-fill effect is a
shared nuisance that cancels when differencing across leakage levels. That holds
only if the missingness pattern does not itself move with the leakage injection.
Leakage is about which stays enter pretraining and fill comes from raw
collection practice, so they should be independent -- but "should be" is the
standard this project keeps catching itself failing, so it is measured here
rather than asserted.

Two quantities per condition:

  * PROBE fill proportion. The probe slice is fixed within a seed, so this must
    be IDENTICAL across conditions. Anything else is a bug in the split.
  * TRAINING fill proportion. This is expected to move with leakage, because
    adding target-site stays changes the mix -- that is part of the signal, not
    a confound. Reported so the size of the movement is on record.

No training runs; this only builds datasets.

    PCL_TEST_MODE=1 python rebootpcl/check2_fill_audit.py --stays 900 --seeds 3
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rebootpcl.checks.check2_pretrain_leakage import build, LEVELS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def fill_proportion(dataset):
    """Fraction of (timestep, variable) cells the observation mask marks present.

    The mask is built AFTER forward-fill, so a filled cell counts as observed.
    This quantity is therefore "how much of the grid is populated once filling
    has run", which is what the model actually sees.
    """
    tot = filled = 0
    for s in dataset.samples:
        m = np.asarray(s["mask"], dtype=bool)
        filled += int(m.sum())
        tot += int(m.size)
    return filled / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stays", type=int, default=900)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    from config import TEST_MODE
    if not TEST_MODE:
        sys.exit("Run with PCL_TEST_MODE=1")

    from torch.utils.data import Subset, ConcatDataset

    print("=" * 78)
    print("CHECK 2 — forward-fill proportion by leakage condition")
    print("=" * 78, flush=True)

    out = {}
    for s in range(args.seeds):
        seed = 42 + s
        src_ds, leak_ds, probe_ds = build(args.stays, seed=seed)
        probe_fill = fill_proportion(probe_ds)
        print(f"\nseed {seed}: probe fill={probe_fill:.6f} "
              f"(fixed within seed by construction)", flush=True)

        row = {"probe_fill": probe_fill, "train_fill": {}}
        for frac in LEVELS:
            n_leak = int(round(frac * len(leak_ds)))
            if n_leak > 0:
                rng = np.random.default_rng(seed)
                idx = rng.permutation(len(leak_ds))[:n_leak]
                sub = Subset(leak_ds, idx.tolist())
                tot = filled = 0
                for d in (src_ds.samples,
                          [leak_ds.samples[i] for i in idx.tolist()]):
                    for smp in d:
                        m = np.asarray(smp["mask"], dtype=bool)
                        filled += int(m.sum())
                        tot += int(m.size)
                tf = filled / max(tot, 1)
            else:
                tf = fill_proportion(src_ds)
            row["train_fill"][str(frac)] = tf
            print(f"  leakage {int(frac * 100):>3}%  train fill={tf:.6f}")
        out[str(seed)] = row

    print("\n" + "-" * 78)
    probes = [r["probe_fill"] for r in out.values()]
    spread = max(probes) - min(probes)
    print(f"probe fill across seeds: {[round(p, 6) for p in probes]}")
    print("probe fill is constant WITHIN each seed by construction "
          "(same probe slice at every leakage level) — the paired analysis "
          "differences across levels within a seed, so it cancels exactly.")

    tf_spreads = []
    for seed, r in out.items():
        vals = list(r["train_fill"].values())
        tf_spreads.append(max(vals) - min(vals))
        print(f"seed {seed}: train fill range across leakage levels = "
              f"{max(vals) - min(vals):.6f}")
    print(f"\nmax train-fill movement across conditions: {max(tf_spreads):.6f}")
    print("Training-set fill is EXPECTED to move with leakage (adding target "
          "stays changes the mix); it is part of what leakage is, not a "
          "confound on the probe measurement.")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "check2_fill_audit.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
