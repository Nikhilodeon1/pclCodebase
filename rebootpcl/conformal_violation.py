"""
Does the physiological-violation score help conformal prediction hold its
coverage under hospital shift?

Split conformal prediction calibrates a nonconformity threshold on held-out
source data and applies it unchanged at deployment. Under distribution shift the
exchangeability assumption fails and empirical coverage drops below the nominal
level. The question here is whether adding a per-sample constraint-violation term
to the nonconformity score recovers any of that lost coverage.

Two scores, calibrated independently on PhysioNet Site A held-out data:

    baseline(x, y)   = 1 - p_model(y | x)
    augmented(x, y)  = baseline(x, y) + z(violation(x))

where z() is standardized using Site-A statistics only. Note the violation term
is class-independent: it shifts both classes equally, so it changes which samples
get large or small prediction sets, not which class is preferred.

Everything is inference-only. Per-sample violation scores were never persisted by
the main pipeline (the OOD-detector experiment collapsed them to a per-site mean),
so stage 1 recomputes them with one forward pass and caches the result; stage 2 is
pure CPU arithmetic and reruns in seconds.

This script is deliberately standalone. It never writes paper_results.json and
never touches the resume cache.

Usage (from repo root, on the machine holding the checkpoints + preprocessed cache):

    source runpod_env.sh
    PCL_EXPANDED_VARS=1 CACHE_DIR=cache_v17_prod \
        python rebootpcl/conformal_violation.py \
            --ckpt results_lambda17/ckpt/lambda_0.0_sepsis.pt
"""
import os
import sys
import json
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "per_sample_scores.npz")
ALPHA = 0.10                      # target 90% coverage
SITES = ["PhysioNet-B", "MIMIC-IV", "eICU"]


# ── per-sample constraint violation ──────────────────────────────────────────
@torch.no_grad()
def violation_per_sample(preds, c_mask):
    """Mean physiological residual per sample, over positions where each
    constraint is computable.

    Mirrors the residuals in src/losses/pcl_loss.py (MAP L1, Henderson-Hasselbalch
    MSE, Severinghaus MSE) but keeps the batch dimension instead of reducing to a
    scalar. No pretrain masking is applied, matching how the OOD-detector
    experiment scored samples, so the value is a deterministic property of the
    input and the model.
    """
    from src.data.variables import VAR_TO_IDX as IDX
    g = lambda v: preds[:, :, IDX[v]]
    eps = 1e-8

    SBP_d = g("SBP") * (300 - 40) + 40
    DBP_d = g("DBP") * (200 - 20) + 20
    HCO3 = g("HCO3") * (60 - 5) + 5
    pCO2 = g("pCO2") * (150 - 10) + 10
    PaO2 = torch.clamp(g("PaO2") * (700 - 20) + 20, min=1.0)

    # MAP identity (L1 in normalized space)
    map_pred = (DBP_d + (SBP_d - DBP_d) / 3.0 - 20) / (200 - 20 + eps)
    r_map = (g("MAP") - map_pred).abs()

    # Henderson-Hasselbalch
    hc = torch.clamp(HCO3 / (0.0307 * pCO2 + eps), min=0.01, max=1000.0)
    ph_pred = torch.clamp((6.1 + torch.log10(hc) - 6.5) / (7.9 - 6.5 + eps), 0.0, 1.0)
    r_hh = (g("pH") - ph_pred) ** 2

    # Severinghaus
    inner = PaO2 ** 3 + 150.0 * PaO2
    sao2 = inner / (inner + 23400.0 + eps)
    spo2_pred = torch.clamp((sao2 * 100.0 - 50.0) / 50.0, 0.0, 1.0)
    r_spo2 = (g("SpO2") - spo2_pred) ** 2

    total = torch.zeros(preds.shape[0], device=preds.device)
    n_act = torch.zeros(preds.shape[0], device=preds.device)
    for r, k in ((r_map, 0), (r_hh, 3), (r_spo2, 4)):
        m = c_mask[:, :, k].float()
        cnt = m.sum(dim=1)
        # per-constraint mean over its computable timesteps
        per = (r * m).sum(dim=1) / torch.clamp(cnt, min=1.0)
        total += torch.where(cnt > 0, per, torch.zeros_like(per))
        n_act += (cnt > 0).float()
    return (total / torch.clamp(n_act, min=1.0)).cpu().numpy()


@torch.no_grad()
def score_loader(model, loader, task, device):
    """One pass -> (probs, labels, violations) aligned per sample."""
    model.eval()
    P, Y, V = [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        c = batch["c_mask"].to(device, non_blocking=True)
        reps = model.encode(x, m)
        logits = model.classify(reps, task, obs_mask=m).squeeze(-1)
        P.append(torch.sigmoid(logits).float().cpu().numpy().ravel())
        Y.append(batch[task].cpu().numpy().ravel())
        V.append(violation_per_sample(model.predict(reps).float(), c))
    return (np.concatenate(P), np.concatenate(Y), np.concatenate(V))


# ── conformal machinery ──────────────────────────────────────────────────────
def conformal_threshold(scores, alpha=ALPHA):
    """Split-conformal quantile with the finite-sample correction."""
    n = len(scores)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q, method="higher"))


def sets_and_coverage(p, y, extra, thr):
    """Binary prediction sets under score s(y) = (1 - p_y) + extra.

    Class 1 enters the set when (1 - p) + extra <= thr; class 0 when p + extra
    <= thr. Returns (coverage, mean set size, empty-set rate).
    """
    s1 = (1.0 - p) + extra
    s0 = p + extra
    in1, in0 = s1 <= thr, s0 <= thr
    size = in1.astype(int) + in0.astype(int)
    covered = np.where(y > 0.5, in1, in0)
    return float(covered.mean()), float(size.mean()), float((size == 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results_lambda17/ckpt/lambda_0.0_sepsis.pt",
                    help="ERM checkpoint (lambda=0 from the sweep)")
    ap.add_argument("--refresh", action="store_true", help="recompute the score cache")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    # ── stage 1: per-sample probs / labels / violations (cached) ─────────────
    if args.refresh or not os.path.exists(CACHE):
        from config import TEST_MODE
        if TEST_MODE:
            sys.exit("ABORT: PCL_TEST_MODE=1. Run in production mode "
                     "(source runpod_env.sh) against the full-scale checkpoint.")
        from run_paper_experiments import load_all_data, make_ood_loaders, TASK
        from src.baselines import fresh_model
        from src.training.train_utils import load_state_dict_flexible

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"device={device}  ckpt={args.ckpt}")
        ds, *_ = load_all_data()
        _, val_loader, ood_loaders = make_ood_loaders(ds, batch_size=args.batch_size)

        model = fresh_model().to(device)
        model.add_classification_head(TASK)
        model = model.to(device)
        model.load_state_dict(load_state_dict_flexible(args.ckpt, device))

        # Persist each site the moment it finishes. eICU alone is 130k samples;
        # a failure there must not discard the sites already scored, and a rerun
        # should resume rather than start over.
        blob = {}
        for name, loader in [("SiteA-cal", val_loader)] + [
                (s, ood_loaders[s]) for s in SITES if s in ood_loaders]:
            part = os.path.join(HERE, f"_part_{name}.npz")
            if os.path.exists(part) and not args.refresh:
                pd_ = np.load(part)
                p, y, v = pd_["p"], pd_["y"], pd_["v"]
                logging.info(f"  {name:12s} loaded from {os.path.basename(part)} "
                             f"(n={len(p)})")
            else:
                p, y, v = score_loader(model, loader, TASK, device)
                np.savez_compressed(part, p=p, y=y, v=v)
                logging.info(f"  {name:12s} n={len(p):7d} pos={y.mean():.3f} "
                             f"mean_viol={v.mean():.5f}  -> {os.path.basename(part)}")
            blob[f"{name}__p"], blob[f"{name}__y"], blob[f"{name}__v"] = p, y, v
        np.savez_compressed(CACHE, **blob)
        logging.info(f"cached -> {CACHE}")
    else:
        logging.info(f"using cached scores at {CACHE} (--refresh to recompute)")

    d = np.load(CACHE)
    pc, yc, vc = d["SiteA-cal__p"], d["SiteA-cal__y"], d["SiteA-cal__v"]

    # Standardize the violation using calibration-set statistics ONLY.
    mu, sd = float(vc.mean()), float(vc.std())
    sd = sd if sd > 1e-12 else 1.0
    z = lambda v: (v - mu) / sd

    # Nonconformity of the TRUE class on calibration data.
    base_cal = np.where(yc > 0.5, 1.0 - pc, pc)
    aug_cal = base_cal + z(vc)
    thr_base = conformal_threshold(base_cal)
    thr_aug = conformal_threshold(aug_cal)
    logging.info(f"calibrated on n={len(pc)}  thr_base={thr_base:.4f}  "
                 f"thr_aug={thr_aug:.4f}")

    rows = []
    for name in ["SiteA-cal"] + SITES:
        if f"{name}__p" not in d:
            continue
        p, y, v = d[f"{name}__p"], d[f"{name}__y"], d[f"{name}__v"]
        bc, bs, be = sets_and_coverage(p, y, 0.0, thr_base)
        ac, asz, ae = sets_and_coverage(p, y, z(v), thr_aug)
        rows.append((name, len(p), bc, bs, ac, asz, be, ae))

    hdr = (f"{'Site':<14}{'n':>8}{'base cov':>11}{'base size':>11}"
           f"{'aug cov':>10}{'aug size':>10}")
    print("\n" + "=" * len(hdr))
    print(f"SPLIT CONFORMAL @ {int((1-ALPHA)*100)}% TARGET COVERAGE")
    print("=" * len(hdr))
    print(hdr)
    for name, n, bc, bs, ac, asz, be, ae in rows:
        tag = "  (calib)" if name == "SiteA-cal" else ""
        print(f"{name:<14}{n:>8}{bc:>11.3f}{bs:>11.3f}{ac:>10.3f}{asz:>10.3f}{tag}")
    print(f"{'target':<14}{'':>8}{1-ALPHA:>11.3f}{'':>11}{1-ALPHA:>10.3f}")
    print("\nEmpty-set rate (score exceeded threshold for both classes):")
    for name, n, bc, bs, ac, asz, be, ae in rows:
        print(f"  {name:<14} baseline {be:.3f}   augmented {ae:.3f}")

    out = os.path.join(HERE, "conformal_results.json")
    json.dump({"alpha": ALPHA, "thr_base": thr_base, "thr_aug": thr_aug,
               "viol_mu": mu, "viol_sd": sd,
               "rows": [{"site": r[0], "n": int(r[1]),
                         "base_coverage": r[2], "base_set_size": r[3],
                         "aug_coverage": r[4], "aug_set_size": r[5],
                         "base_empty": r[6], "aug_empty": r[7]} for r in rows]},
              open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
