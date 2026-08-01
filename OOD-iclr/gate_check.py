"""
Gate check: can UNLABELED target data select the constraint weight?

Motivation. The full-scale lambda sweep showed Site-A validation AUROC spans only
~1pp across lambda while true OOD AUROC spans ~11pp, i.e. training-domain
validation cannot select lambda. This asks whether a label-free signal computed
on unlabeled target data can. Two candidate signals:

  * violation : mean physiological-constraint residual (MAP / Henderson-Hasselbalch
                / Severinghaus) of the model's own reconstructions on target data.
  * entropy   : mean predictive entropy of the sepsis head on target data.

Protocol. For every already-trained lambda checkpoint x every target site, run ONE
inference-only pass (no training, no backprop) to get both scores, pair them with
that (lambda, site) cell's already-known true OOD AUROC, and report:

  * Spearman correlation between each score and true AUROC, pooled over all
    (lambda, site) cells -- so the unit of analysis is the cell, not the lambda.
    Signs: a good selector has NEGATIVE correlation (lower violation / lower
    entropy should mean higher AUROC), so we also report correlation against
    -score as "aligned" for readability.
  * Selection regret per site: true AUROC of the lambda each signal would pick,
    minus true AUROC of the actually-best lambda for that site. 0 = perfect
    selection; more negative = worse. Site-A-val selection and a random-lambda
    baseline are included for reference.

Everything here is inference-only. Usage (from repo root):

    PCL_EXPANDED_VARS=1 CACHE_DIR=cache_v17_prod \
      python OOD-iclr/gate_check.py --ckpt-dir results_lambda17/ckpt \
                                    --results results_lambda17/paper_results.json
"""
import os
import sys
import json
import glob
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SITES = ["PhysioNet-B", "MIMIC-IV", "eICU"]


# ── scoring ──────────────────────────────────────────────────────────────────
def _binary_entropy(p, eps=1e-8):
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


@torch.no_grad()
def score_loader(model, loader, pcl_loss, task, device, mask_prob, seed=0):
    """One inference pass -> (mean violation, mean predictive entropy).

    The violation is computed exactly as during PCL training: on the model's
    reconstructions, at timesteps where the constraint is computable AND at least
    one required variable was masked out (so the model must impute it rather than
    copy the input). Masking is drawn from a fixed seed so the score is
    deterministic and comparable across checkpoints.
    """
    from src.models.backbone import apply_random_mask

    model.eval()
    viol_sum, viol_n = 0.0, 0
    ent_sum, ent_n = 0.0, 0

    g = torch.Generator(device="cpu").manual_seed(seed)
    for bi, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        c = batch["c_mask"].to(device, non_blocking=True)

        # Deterministic mask: same pattern for every checkpoint.
        torch.manual_seed(seed * 100003 + bi)
        x_masked, pretrain_mask = apply_random_mask(x, m, mask_prob)

        reps = model.encode(x_masked, m)
        preds = model.predict(reps)
        out = pcl_loss(preds.float(), c, pretrain_mask)

        # Mean over the three ACTIVE equality constraints, counting only those
        # with active positions in this batch (SI/PP are excluded from L_PCL).
        parts = [out["losses"][k].item() for k in ("MAP", "HH", "SpO2")
                 if out["active"][k] > 0]
        if parts:
            viol_sum += float(np.mean(parts)) * x.shape[0]
            viol_n += x.shape[0]

        # Predictive entropy from the sepsis head on the UNMASKED input.
        reps_full = model.encode(x, m)
        logits = model.classify(reps_full, task, obs_mask=m).squeeze(-1)
        p = torch.sigmoid(logits).float().cpu().numpy()
        e = _binary_entropy(p)
        ent_sum += float(e.sum())
        ent_n += e.size

    return (viol_sum / max(viol_n, 1), ent_sum / max(ent_n, 1))


# ── correlation / regret ─────────────────────────────────────────────────────
def spearman(a, b):
    """Spearman rho via Pearson on ranks (avoids a scipy dependency)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")

    def rank(v):
        order = v.argsort()
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        for u in np.unique(v):
            idx = np.where(v == u)[0]
            if len(idx) > 1:
                r[idx] = r[idx].mean()
        return r

    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def regret(scores_by_lam, auroc_by_lam, lower_is_better=True):
    """AUROC of the argmin/argmax-selected lambda minus the best achievable."""
    lams = [l for l in scores_by_lam
            if scores_by_lam[l] is not None and auroc_by_lam.get(l) is not None]
    if not lams:
        return float("nan"), None
    pick = (min if lower_is_better else max)(lams, key=lambda l: scores_by_lam[l])
    best = max(lams, key=lambda l: auroc_by_lam[l])
    return auroc_by_lam[pick] - auroc_by_lam[best], pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="results_lambda17/ckpt")
    ap.add_argument("--results", default="results_lambda17/paper_results.json")
    ap.add_argument("--out", default="OOD-iclr/gate_check_results.json")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from config import CACHE_DIR, MASK_PROB, TEST_MODE
    from run_paper_experiments import load_all_data, make_ood_loaders, TASK
    from src.baselines import fresh_model
    from src.losses.pcl_loss import PhysiologicalConstraintLoss
    from src.training.train_utils import load_state_dict_flexible

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"device={device}  TEST_MODE={TEST_MODE}  CACHE_DIR={CACHE_DIR}")

    # True OOD AUROC per (lambda, site) from the completed sweep.
    with open(args.results) as f:
        sweep = json.load(f).get("ablation_lambda", {})
    true_auroc = {}
    for k, v in sweep.items():
        if k == "_val_selected_lambda" or not isinstance(v, dict):
            continue
        true_auroc[k] = {"val": v.get("val"), **(v.get("ood") or {})}
    if not true_auroc:
        sys.exit(f"No ablation_lambda entries in {args.results}")

    # Checkpoints actually on disk.
    ck = {}
    for p in sorted(glob.glob(os.path.join(args.ckpt_dir, f"lambda_*_{TASK}.pt"))):
        lam = os.path.basename(p)[len("lambda_"):-len(f"_{TASK}.pt")]
        if lam in true_auroc:
            ck[lam] = p
    if not ck:
        sys.exit(f"No lambda_*_{TASK}.pt checkpoints in {args.ckpt_dir}")
    logging.info(f"checkpoints: {sorted(ck, key=float)}")

    ds, *_ = load_all_data()
    _, val_loader, ood_loaders = make_ood_loaders(ds, batch_size=args.batch_size)
    pcl_loss = PhysiologicalConstraintLoss().to(device)

    scores = {}   # lam -> site -> {violation, entropy}
    for lam in sorted(ck, key=float):
        model = fresh_model().to(device)
        model.add_classification_head(TASK)
        model = model.to(device)
        model.load_state_dict(load_state_dict_flexible(ck[lam], device))
        scores[lam] = {}
        for site in SITES:
            if site not in ood_loaders:
                continue
            v, e = score_loader(model, ood_loaders[site], pcl_loss, TASK,
                                device, MASK_PROB, seed=args.seed)
            scores[lam][site] = {"violation": v, "entropy": e}
            logging.info(f"  lam={lam:>4} {site:12s} violation={v:.6f} entropy={e:.6f} "
                         f"true_auroc={true_auroc[lam].get(site)}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── pooled correlations over (lambda, site) cells ────────────────────────
    cells = []
    for lam in scores:
        for site, s in scores[lam].items():
            a = true_auroc[lam].get(site)
            if a is None:
                continue
            cells.append((lam, site, s["violation"], s["entropy"], a))
    v_rho = spearman([c[2] for c in cells], [c[4] for c in cells])
    e_rho = spearman([c[3] for c in cells], [c[4] for c in cells])

    # ── per-site selection regret ────────────────────────────────────────────
    reg = {}
    for site in SITES:
        au = {l: true_auroc[l].get(site) for l in scores if site in scores[l]}
        vi = {l: scores[l][site]["violation"] for l in scores if site in scores[l]}
        en = {l: scores[l][site]["entropy"] for l in scores if site in scores[l]}
        va = {l: true_auroc[l].get("val") for l in scores if site in scores[l]}
        r_v, p_v = regret(vi, au, lower_is_better=True)
        r_e, p_e = regret(en, au, lower_is_better=True)
        r_s, p_s = regret(va, au, lower_is_better=False)   # Site-A val selection
        lams = [l for l in au if au[l] is not None]
        rand = float(np.mean([au[l] for l in lams]) - max(au[l] for l in lams)) if lams else float("nan")
        reg[site] = {"violation": {"regret": r_v, "picked": p_v},
                     "entropy":   {"regret": r_e, "picked": p_e},
                     "site_a_val": {"regret": r_s, "picked": p_s},
                     "random_lambda_expected": rand,
                     "best_lambda": (max(lams, key=lambda l: au[l]) if lams else None)}

    out = {"n_cells": len(cells),
           "spearman_violation_vs_auroc": v_rho,
           "spearman_entropy_vs_auroc": e_rho,
           "scores": scores, "true_auroc": true_auroc, "regret": reg}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ── report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"GATE CHECK  ({len(cells)} (lambda, site) cells)")
    print("=" * 68)
    print("Spearman rho vs true OOD AUROC   (negative = good selector;")
    print("                                  'aligned' flips sign for readability)")
    print(f"  violation : rho = {v_rho:+.3f}   aligned = {-v_rho:+.3f}")
    print(f"  entropy   : rho = {e_rho:+.3f}   aligned = {-e_rho:+.3f}")
    print("\nSelection regret (true AUROC of picked lambda - best achievable; 0 = perfect)")
    print(f"{'site':<13}{'violation':>22}{'entropy':>22}{'site-A val':>22}{'random':>10}")
    for site in SITES:
        if site not in reg:
            continue
        r = reg[site]
        f = lambda d: (f"{d['regret']:+.4f} (lam={d['picked']})"
                       if d["regret"] == d["regret"] else "n/a")
        print(f"{site:<13}{f(r['violation']):>22}{f(r['entropy']):>22}"
              f"{f(r['site_a_val']):>22}{r['random_lambda_expected']:>10.4f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
