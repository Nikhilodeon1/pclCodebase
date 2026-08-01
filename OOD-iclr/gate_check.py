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
def source_mean_repr(model, loader, device):
    """Mean pooled encoder representation on SOURCE (Site-A val) data.

    Reference point for the representation-distance baseline: how far the target
    domain's representation centroid sits from the source centroid.
    """
    tot, n = None, 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        r = model.encode(x, m).mean(dim=1)          # (B, d)
        s = r.sum(dim=0).double()
        tot = s if tot is None else tot + s
        n += r.shape[0]
    return (tot / max(n, 1)) if tot is not None else None


@torch.no_grad()
def score_loader(model, loader, pcl_loss, task, device, mask_prob, seed=0,
                 src_mu=None):
    """One inference pass -> dict of label-free scores.

    Scores (all lower-is-better as selection signals):
      violation  : physiological-constraint residual (the PCL signal)
      entropy    : predictive entropy of the sepsis head
      recon_mse  : plain masked-reconstruction error -- the natural "generic
                   unsupervised score" control. If violation only matches this,
                   the physiology adds nothing beyond reconstruction quality.
      repr_dist  : L2 distance from the source representation centroid -- the
                   natural "domain-distance" control.

    The violation is computed exactly as during PCL training: on the model's
    reconstructions, at timesteps where the constraint is computable AND at least
    one required variable was masked out (so the model must impute it rather than
    copy the input). Masking is drawn from a fixed seed so the score is
    deterministic and comparable across checkpoints.
    """
    from src.models.backbone import apply_random_mask
    from src.training.train_utils import masked_prediction_loss

    model.eval()
    viol_sum, viol_n = 0.0, 0
    ent_sum, ent_n = 0.0, 0
    mse_sum, mse_n = 0.0, 0
    rep_sum, rep_n = None, 0

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

        # Plain masked-reconstruction error on the same masked positions.
        mse = masked_prediction_loss(preds, x, pretrain_mask)
        if torch.isfinite(mse):
            mse_sum += float(mse.item()) * x.shape[0]
            mse_n += x.shape[0]

        # Predictive entropy from the sepsis head on the UNMASKED input.
        reps_full = model.encode(x, m)
        logits = model.classify(reps_full, task, obs_mask=m).squeeze(-1)
        p = torch.sigmoid(logits).float().cpu().numpy()
        e = _binary_entropy(p)
        ent_sum += float(e.sum())
        ent_n += e.size

        # Running sum for the target representation centroid.
        rp = reps_full.mean(dim=1).sum(dim=0).double()
        rep_sum = rp if rep_sum is None else rep_sum + rp
        rep_n += reps_full.shape[0]

    out = {"violation": viol_sum / max(viol_n, 1),
           "entropy": ent_sum / max(ent_n, 1),
           "recon_mse": mse_sum / max(mse_n, 1)}
    if src_mu is not None and rep_sum is not None and rep_n > 0:
        tgt_mu = rep_sum / rep_n
        out["repr_dist"] = float(torch.linalg.norm(tgt_mu - src_mu).item())
    else:
        out["repr_dist"] = float("nan")
    return out


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

    from config import CACHE_DIR, MASK_PROB, TEST_MODE, D_MODEL, N_LAYERS
    from run_paper_experiments import load_all_data, make_ood_loaders, TASK
    from src.baselines import fresh_model
    from src.losses.pcl_loss import PhysiologicalConstraintLoss
    from src.training.train_utils import load_state_dict_flexible

    # The sweep checkpoints are PRODUCTION-sized (d_model=256, 6 layers) and were
    # trained on full data. Running here in test mode builds a d_model=64/2-layer
    # model and loads demo data, which fails with an opaque state_dict size
    # mismatch -- fail loudly and early instead.
    if TEST_MODE:
        sys.exit("ABORT: PCL_TEST_MODE=1 (default). This must run in PRODUCTION mode "
                 "against the full-scale checkpoints/cache.\n"
                 "  Fix: source runpod_env.sh   (or export PCL_TEST_MODE=0 plus data paths)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"device={device}  d_model={D_MODEL} n_layers={N_LAYERS}  CACHE_DIR={CACHE_DIR}")

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
        src_mu = source_mean_repr(model, val_loader, device)
        for site in SITES:
            if site not in ood_loaders:
                continue
            s = score_loader(model, ood_loaders[site], pcl_loss, TASK,
                             device, MASK_PROB, seed=args.seed, src_mu=src_mu)
            scores[lam][site] = s
            logging.info(f"  lam={lam:>4} {site:12s} viol={s['violation']:.6f} "
                         f"ent={s['entropy']:.6f} mse={s['recon_mse']:.6f} "
                         f"rdist={s['repr_dist']:.4f} true={true_auroc[lam].get(site)}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── pooled correlations over (lambda, site) cells ────────────────────────
    SIGNALS = ["violation", "entropy", "recon_mse", "repr_dist"]
    cells = []
    for lam in scores:
        for site, s in scores[lam].items():
            a = true_auroc[lam].get(site)
            if a is None:
                continue
            cells.append({"lam": lam, "site": site, "auroc": a, **{k: s[k] for k in SIGNALS}})

    def _rho(sig, subset=None):
        cs = subset if subset is not None else cells
        cs = [c for c in cs if c[sig] == c[sig]]  # drop NaN
        return spearman([c[sig] for c in cs], [c["auroc"] for c in cs])

    rhos = {s: _rho(s) for s in SIGNALS}
    # Robustness: lambda=0 never saw the constraint loss, so verify the signal is
    # not carried by that single configuration.
    nz = [c for c in cells if float(c["lam"]) > 0]
    rhos_nz = {s: _rho(s, nz) for s in SIGNALS}
    rhos_site = {st: {s: _rho(s, [c for c in cells if c["site"] == st]) for s in SIGNALS}
                 for st in SITES}

    # ── per-site selection regret ────────────────────────────────────────────
    reg = {}
    for site in SITES:
        au = {l: true_auroc[l].get(site) for l in scores if site in scores[l]}
        lams = [l for l in au if au[l] is not None]
        entry = {}
        for sig in SIGNALS:
            sc = {l: scores[l][site][sig] for l in scores if site in scores[l]}
            sc = {l: v for l, v in sc.items() if v == v}
            r, p = regret(sc, au, lower_is_better=True)
            entry[sig] = {"regret": r, "picked": p}
        va = {l: true_auroc[l].get("val") for l in scores if site in scores[l]}
        r_s, p_s = regret(va, au, lower_is_better=False)   # Site-A val selection
        entry["site_a_val"] = {"regret": r_s, "picked": p_s}
        entry["random_lambda_expected"] = (
            float(np.mean([au[l] for l in lams]) - max(au[l] for l in lams))
            if lams else float("nan"))
        entry["best_lambda"] = max(lams, key=lambda l: au[l]) if lams else None
        reg[site] = entry

    out = {"n_cells": len(cells), "signals": SIGNALS,
           "spearman_all": rhos, "spearman_excl_lambda0": rhos_nz,
           "spearman_within_site": rhos_site,
           "scores": scores, "true_auroc": true_auroc, "regret": reg}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ── report ───────────────────────────────────────────────────────────────
    # All signals are lower-is-better, so a NEGATIVE rho means a GOOD selector.
    # Only raw rho is printed (no sign-flipped column) to avoid misreading.
    print("\n" + "=" * 78)
    print(f"GATE CHECK  ({len(cells)} (lambda, site) cells)   [seed {args.seed}]")
    print("=" * 78)
    print("Spearman rho vs true OOD AUROC. All signals are lower-is-better,")
    print("so MORE NEGATIVE rho = BETTER selector. Raw rho only.")
    print(f"{'signal':<12}{'all':>10}{'excl lam=0':>12}" + "".join(f"{s:>14}" for s in SITES))
    for s in SIGNALS:
        row = f"{s:<12}{rhos[s]:>10.3f}{rhos_nz[s]:>12.3f}"
        row += "".join(f"{rhos_site[st][s]:>14.3f}" for st in SITES)
        print(row)
    print("\nSelection regret (true AUROC of picked lambda - best achievable; 0 = perfect)")
    hdr = f"{'site':<13}" + "".join(f"{s:>22}" for s in SIGNALS) + f"{'site-A val':>22}{'random':>10}"
    print(hdr)
    for site in SITES:
        if site not in reg:
            continue
        r = reg[site]
        fmt = lambda d: (f"{d['regret']:+.4f} (l={d['picked']})"
                         if d["regret"] == d["regret"] else "n/a")
        row = f"{site:<13}" + "".join(f"{fmt(r[s]):>22}" for s in SIGNALS)
        row += f"{fmt(r['site_a_val']):>22}{r['random_lambda_expected']:>10.4f}"
        print(row)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
