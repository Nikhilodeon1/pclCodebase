"""Aggregate the per-seed gate-check JSONs into one verdict table.

Reports, per label-free signal, the Spearman rho against true OOD AUROC and the
mean selection regret, as mean +/- std across seeds. All signals are
lower-is-better, so MORE NEGATIVE rho = better selector; regret closer to 0 =
better. Site-A validation selection and random-lambda are the reference rows.
"""
import os
import sys
import glob
import json

import numpy as np

SITES = ["PhysioNet-B", "MIMIC-IV", "eICU"]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    paths = sorted(glob.glob(os.path.join(here, "gate_seed*.json")))
    if not paths:
        print("No gate_seed*.json found — run the gate checks first.")
        return
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append((os.path.basename(p).replace(".json", ""), json.load(f)))
    signals = runs[0][1].get("signals", ["violation", "entropy", "recon_mse", "repr_dist"])

    def ms(vals):
        v = [x for x in vals if x is not None and x == x]
        if not v:
            return float("nan"), float("nan")
        return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)

    print("=" * 78)
    print(f"GATE CHECK SUMMARY over {len(runs)} seed(s): {[r[0] for r in runs]}")
    print("=" * 78)
    print("Spearman rho vs true OOD AUROC (lower-is-better signals => MORE NEGATIVE = better)")
    print(f"{'signal':<12}{'rho (all)':>18}{'rho (excl lam=0)':>22}")
    for s in signals:
        a_m, a_s = ms([r[1]["spearman_all"].get(s) for r in runs])
        n_m, n_s = ms([r[1]["spearman_excl_lambda0"].get(s) for r in runs])
        print(f"{s:<12}{a_m:>11.3f}+/-{a_s:<5.3f}{n_m:>15.3f}+/-{n_s:<5.3f}")

    print("\nSelection regret, mean over sites (0 = picked the best lambda)")
    hdr = f"{'signal':<12}" + "".join(f"{st:>18}" for st in SITES) + f"{'mean':>12}"
    print(hdr)
    rows = signals + ["site_a_val"]
    for s in rows:
        per_site, cells = [], []
        for st in SITES:
            m, sd = ms([r[1]["regret"].get(st, {}).get(s, {}).get("regret") for r in runs])
            per_site.append(f"{m:>11.4f}+/-{sd:<5.3f}" if m == m else f"{'n/a':>18}")
            if m == m:
                cells.append(m)
        mean = np.mean(cells) if cells else float("nan")
        print(f"{s:<12}" + "".join(f"{c:>18}" for c in per_site) + f"{mean:>12.4f}")
    rnd = []
    for st in SITES:
        m, _ = ms([r[1]["regret"].get(st, {}).get("random_lambda_expected") for r in runs])
        if m == m:
            rnd.append(m)
    if rnd:
        print(f"{'random':<12}" + " " * (18 * len(SITES)) + f"{np.mean(rnd):>12.4f}")

    print("\nPicked lambda per seed/site (violation signal):")
    for name, d in runs:
        picks = {st: d["regret"].get(st, {}).get("violation", {}).get("picked") for st in SITES}
        best = {st: d["regret"].get(st, {}).get("best_lambda") for st in SITES}
        print(f"  {name}: picked={picks}  true_best={best}")


if __name__ == "__main__":
    main()
