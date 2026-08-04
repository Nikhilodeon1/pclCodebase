"""
Two controls for the violation-augmented conformal result.

(1) SITE-MEAN-ONLY. Replace each sample's standardized violation with its SITE
    MEAN. If coverage and set size are unchanged, the augmentation is a constant
    per-site offset and the per-sample violation contributes nothing -- the
    physiology is decorative.

(2) MATCHED-COVERAGE EFFICIENCY. Augmentation raises coverage partly by making
    sets bigger. The fair question is whether it buys coverage more cheaply than
    simply widening the baseline. For each target site we find the calibration
    alpha at which the BASELINE reaches the augmented method's coverage, and
    compare set sizes at that matched coverage. If baseline gets there with
    smaller sets, augmentation is strictly worse.

Pure array math over the cached per-sample scores. No model, no GPU, seconds.

    python rebootpcl/conformal_diagnostics.py [path/to/per_sample_scores.npz]
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.10
SITES = ["PhysioNet-B", "MIMIC-IV", "eICU"]
CAL = "SiteA-cal"


def threshold(scores, alpha):
    n = len(scores)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q, method="higher"))


def evaluate(p, y, extra, thr):
    s1, s0 = (1.0 - p) + extra, p + extra
    in1, in0 = s1 <= thr, s0 <= thr
    size = in1.astype(int) + in0.astype(int)
    cov = np.where(y > 0.5, in1, in0)
    return float(cov.mean()), float(size.mean()), float((size == 0).mean())


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "per_sample_scores.npz")
    if not os.path.exists(path):
        sys.exit(f"missing {path} — copy it from the pod (rebootpcl/per_sample_scores.npz)")
    d = np.load(path)
    pc, yc, vc = d[f"{CAL}__p"], d[f"{CAL}__y"], d[f"{CAL}__v"]
    mu, sd = float(vc.mean()), float(vc.std()) or 1.0
    z = lambda v: (v - mu) / sd

    base_cal = np.where(yc > 0.5, 1.0 - pc, pc)
    aug_cal = base_cal + z(vc)
    thr_b = threshold(base_cal, ALPHA)
    thr_a = threshold(aug_cal, ALPHA)

    # The site-mean control must be calibrated the same way: on calibration data
    # the "site mean" is the calibration mean, so z() of it is exactly 0.
    smean_cal = base_cal + 0.0
    thr_s = threshold(smean_cal, ALPHA)

    print("=" * 78)
    print("DIAGNOSTIC 1 — per-sample violation vs. site-mean-only")
    print("=" * 78)
    print(f"{'Site':<14}{'base cov':>10}{'base sz':>9}{'aug cov':>9}{'aug sz':>8}"
          f"{'mean cov':>10}{'mean sz':>9}")
    rows = {}
    for name in [CAL] + SITES:
        if f"{name}__p" not in d:
            continue
        p, y, v = d[f"{name}__p"], d[f"{name}__y"], d[f"{name}__v"]
        bc, bs, _ = evaluate(p, y, 0.0, thr_b)
        ac, asz, _ = evaluate(p, y, z(v), thr_a)
        # site-mean-only: every sample in this site gets the site's mean violation
        mc, ms, _ = evaluate(p, y, np.full(len(p), z(v.mean())), thr_s if name == CAL else thr_a)
        rows[name] = dict(base=(bc, bs), aug=(ac, asz), smean=(mc, ms))
        print(f"{name:<14}{bc:>10.3f}{bs:>9.3f}{ac:>9.3f}{asz:>8.3f}{mc:>10.3f}{ms:>9.3f}")
    print("\nIf 'mean' matches 'aug', the per-sample violation adds nothing.")
    gaps = [abs(rows[s]['aug'][0] - rows[s]['smean'][0]) for s in SITES if s in rows]
    szg = [abs(rows[s]['aug'][1] - rows[s]['smean'][1]) for s in SITES if s in rows]
    print(f"  max |coverage difference| across target sites: {max(gaps):.4f}")
    print(f"  max |set-size difference| across target sites: {max(szg):.4f}")

    print("\n" + "=" * 78)
    print("DIAGNOSTIC 2 — matched-coverage efficiency")
    print("=" * 78)
    print("Widen the BASELINE (sweep calibration alpha) until it reaches the")
    print("augmented method's coverage at each site, then compare set sizes.\n")
    print(f"{'Site':<14}{'aug cov':>9}{'aug sz':>9}{'base sz @ same cov':>22}{'alpha*':>9}{'verdict':>12}")
    alphas = np.linspace(0.001, 0.5, 500)
    thrs = [threshold(base_cal, a) for a in alphas]
    for name in SITES:
        if name not in rows:
            continue
        p, y = d[f"{name}__p"], d[f"{name}__y"]
        target_cov, aug_sz = rows[name]["aug"]
        best = None
        for a, t in zip(alphas, thrs):
            c, s, _ = evaluate(p, y, 0.0, t)
            if c >= target_cov:
                best = (a, s, c)      # first alpha reaching the target coverage
                break
        if best is None:
            print(f"{name:<14}{target_cov:>9.3f}{aug_sz:>9.3f}{'unreachable':>22}{'-':>9}{'aug wins':>12}")
            continue
        a, s, c = best
        verdict = "aug better" if aug_sz < s - 1e-9 else "baseline better"
        print(f"{name:<14}{target_cov:>9.3f}{aug_sz:>9.3f}{s:>22.3f}{a:>9.3f}{verdict:>12}")

    out = os.path.join(HERE, "conformal_diagnostics.json")
    json.dump({k: {kk: list(vv) for kk, vv in v.items()} for k, v in rows.items()},
              open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
