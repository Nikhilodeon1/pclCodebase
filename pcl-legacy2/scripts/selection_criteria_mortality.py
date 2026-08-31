"""
pcl-legacy2, Step 4 + 5 — mortality task, selection-criteria comparison.

Step 4: URTC's inference-only selection-criteria protocol (violation,
entropy, recon_mse, repr_dist — see _archive/OOD-iclr/gate_check.py) reused
against the 18 already-fine-tuned mortality checkpoints, plus the two new
required baselines: ATC and MMD (both inference-time only, no new training).

NOT importing gate_check.py directly: its own top-level sys.path.insert(0, ...)
points at _archive/, which would shadow chat1_protocol's src/ with the
un-fixed copy (dropping mortality_hospital again, silently). The scoring
logic is reimplemented here against chat1_protocol/src instead — see
../README.md and src/data/dataset.py for why that matters.

Step 5 (encoder-bias confound): recon_mse is a physiology-agnostic signal —
plain masked-reconstruction error. ERM and DRO never saw the physiology
constraint at all during pretraining; PCL did. If recon_mse's correlation
with true OOD AUROC is really just tracking "how good is this encoder at
masked prediction" rather than anything physiology-specific, it should
behave differently for {erm, dro} vs {pcl} — this script reports that split
explicitly, not just the pooled correlation.

Cell = (method, source, seed); "target" is the other site. True OOD AUROC
comes from the already-saved finetune_mortality.py JSON for that cell, not
recomputed here.

Usage (same env as finetune_mortality.py; needs the 18 result JSONs +
matching checkpoints already in ../results/mortality/):
    python selection_criteria_mortality.py
"""
import argparse
import glob
import json
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

os.environ["PCL_TEST_MODE"] = "0"  # see finetune_mortality.py for why this must be forced

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEGACY2_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_LEGACY2_ROOT)
_CHAT1 = os.path.join(_REPO_ROOT, "chat1_protocol")
sys.path.insert(0, _CHAT1)
sys.path.insert(0, _REPO_ROOT)

RESULTS_DIR = os.path.join(_LEGACY2_ROOT, "results", "mortality")
FT_CKPT_DIR = os.path.join(RESULTS_DIR, "ckpt")
OUT_PATH = os.path.join(RESULTS_DIR, "selection_criteria.json")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")

TASK = "mortality_hospital"
SIGNALS_LOWER_BETTER = ["violation", "entropy", "recon_mse", "repr_dist", "mmd"]
SIGNALS_HIGHER_BETTER = ["atc"]
ALL_SIGNALS = SIGNALS_LOWER_BETTER + SIGNALS_HIGHER_BETTER


# ── pure helpers (no torch/model needed — testable standalone) ────────────────
def spearman(a, b):
    """Spearman rho via Pearson on ranks (avoids a scipy dependency). Copied
    from gate_check.py's implementation, not imported — see module docstring."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")

    def rank(v):
        order = v.argsort()
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        for u in np.unique(v):
            idx = np.where(v == u)[0]
            if len(idx) > 1:
                r[idx] = r[idx].mean()
        return r

    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def regret(scores_by_key, auroc_by_key, lower_is_better=True):
    """True AUROC of the picked key minus the best achievable. 0 = perfect."""
    keys = [k for k in scores_by_key if scores_by_key[k] is not None and auroc_by_key.get(k) is not None]
    if not keys:
        return float("nan"), None
    pick = (min if lower_is_better else max)(keys, key=lambda k: scores_by_key[k])
    best = max(keys, key=lambda k: auroc_by_key[k])
    return auroc_by_key[pick] - auroc_by_key[best], pick


def _binary_entropy(p, eps=1e-8):
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _atc_threshold(confidences, correct):
    """Garg et al. 2022 ATC: threshold s.t. fraction of SOURCE samples above it
    equals source accuracy. Confidence = 1 - binary_entropy(p) (higher = more sure)."""
    n = len(confidences)
    acc = float(np.mean(correct))
    order = np.argsort(-confidences)  # descending
    k = int(round(acc * n))
    k = max(1, min(k, n))
    return float(confidences[order[k - 1]])


def _rbf_mmd(x, y, n_sub=500, seed=0):
    """MMD^2 with RBF kernel, median-heuristic bandwidth. Subsamples both sides
    to n_sub for tractable O(n^2) — full source/target sites run into the tens
    of thousands of rows."""
    rng = np.random.default_rng(seed)
    if len(x) > n_sub:
        x = x[rng.choice(len(x), n_sub, replace=False)]
    if len(y) > n_sub:
        y = y[rng.choice(len(y), n_sub, replace=False)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")

    def _sqdist(a, b):
        return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)

    both = np.concatenate([x, y], axis=0)
    d2 = _sqdist(both, both)
    bandwidth = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    bandwidth = max(bandwidth, 1e-6)

    def _k(a, b):
        return np.exp(-_sqdist(a, b) / bandwidth)

    kxx = _k(x, x); kyy = _k(y, y); kxy = _k(x, y)
    m, n = len(x), len(y)
    # unbiased estimator, excluding diagonal on the within-sample terms
    txx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1)) if m > 1 else 0.0
    tyy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) if n > 1 else 0.0
    txy = kxy.mean()
    return float(max(txx + tyy - 2 * txy, 0.0))


# ── model-dependent scoring (one inference pass per checkpoint) ───────────────
def score_checkpoint(model, src_loader, tgt_loader, device, seed=0):
    import torch
    from src.losses.pcl_loss import PhysiologicalConstraintLoss
    from src.models.backbone import apply_random_mask
    from src.training.train_utils import masked_prediction_loss
    from config import MASK_PROB

    pcl_loss = PhysiologicalConstraintLoss().to(device)
    model.eval()

    @torch.no_grad()
    def _pass(loader, want_correctness=False):
        """One forward pass. Always collects confidences (target needs them for
        ATC too, no labels required for that). want_correctness=True also pulls
        labels, for the source-side ATC threshold calibration only."""
        viol_sum, viol_n = 0.0, 0
        ent_sum, ent_n = 0.0, 0
        mse_sum, mse_n = 0.0, 0
        rep_sum, rep_n = None, 0
        confidences, corrects = [], []
        for bi, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            c = batch["c_mask"].to(device, non_blocking=True)

            torch.manual_seed(seed * 100003 + bi)
            x_masked, pretrain_mask = apply_random_mask(x, m, MASK_PROB)
            reps_masked = model.encode(x_masked, m)
            preds = model.predict(reps_masked)
            out = pcl_loss(preds.float(), c, pretrain_mask)
            parts = [out["losses"][k].item() for k in ("MAP", "HH", "SpO2") if out["active"][k] > 0]
            if parts:
                viol_sum += float(np.mean(parts)) * x.shape[0]
                viol_n += x.shape[0]
            mse = masked_prediction_loss(preds, x, pretrain_mask)
            if torch.isfinite(mse):
                mse_sum += float(mse.item()) * x.shape[0]
                mse_n += x.shape[0]

            reps_full = model.encode(x, m)
            logits = model.classify(reps_full, TASK, obs_mask=m).squeeze(-1)
            p = torch.sigmoid(logits).float().cpu().numpy()
            e = _binary_entropy(p)
            ent_sum += float(e.sum()); ent_n += e.size
            confidences.append(1.0 - e)
            if want_correctness:
                y = batch[TASK].cpu().numpy()
                corrects.append(((p >= 0.5).astype(float) == y).astype(float))

            rp = reps_full.mean(dim=1).sum(dim=0).double()
            rep_sum = rp if rep_sum is None else rep_sum + rp
            rep_n += reps_full.shape[0]

        return {
            "violation": viol_sum / max(viol_n, 1),
            "entropy": ent_sum / max(ent_n, 1),
            "recon_mse": mse_sum / max(mse_n, 1),
            "centroid": (rep_sum / rep_n) if rep_n else None,
            "confidences": np.concatenate(confidences) if confidences else np.array([]),
            "corrects": np.concatenate(corrects) if corrects else np.array([]),
        }

    @torch.no_grad()
    def _pooled_reps(loader, n_sub=500, seed_=0):
        rng = np.random.default_rng(seed_)
        out = []
        n_total = 0
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            reps = model.encode(x, m).mean(dim=1).cpu().numpy()
            out.append(reps)
            n_total += len(reps)
            if n_total >= n_sub * 3:  # enough to subsample from without reading whole site
                break
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1))

    src = _pass(src_loader, want_correctness=True)
    tgt = _pass(tgt_loader, want_correctness=False)

    repr_dist = float("nan")
    if src["centroid"] is not None and tgt["centroid"] is not None:
        repr_dist = float(torch.linalg.norm(tgt["centroid"] - src["centroid"]).item())

    atc = float("nan")
    if len(src["confidences"]) > 0 and len(tgt["confidences"]) > 0:
        thresh = _atc_threshold(src["confidences"], src["corrects"])
        atc = float(np.mean(tgt["confidences"] >= thresh))

    src_reps = _pooled_reps(src_loader)
    tgt_reps = _pooled_reps(tgt_loader)
    mmd = _rbf_mmd(src_reps, tgt_reps, seed=seed)

    # All target-side signals: violation/entropy/recon_mse are scored on the
    # unlabeled target site, same as gate_check.py's protocol (score_loader was
    # called on the OOD loader, not the source).
    return {
        "violation": tgt["violation"],
        "entropy": tgt["entropy"],
        "recon_mse": tgt["recon_mse"],
        "repr_dist": repr_dist,
        "atc": atc,
        "mmd": mmd,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=1.0)
    args = ap.parse_args()

    result_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    result_files = [f for f in result_files if os.path.basename(f) != "selection_criteria.json"]
    if len(result_files) != 18:
        logging.warning(f"Expected 18 fine-tune result JSONs, found {len(result_files)} in {RESULTS_DIR}. "
                         "Continuing anyway, but check finetune_mortality.py ran to completion.")

    import torch
    from torch.utils.data import DataLoader
    from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, TEST_MODE, D_MODEL, N_LAYERS
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset, make_patient_split_loaders
    from src.training.train_utils import load_state_dict_flexible

    if TEST_MODE or D_MODEL != 256 or N_LAYERS != 6:
        raise RuntimeError(f"TEST_MODE={TEST_MODE} D_MODEL={D_MODEL} N_LAYERS={N_LAYERS} — see finetune_mortality.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load + cache each site once (same cache finetune_mortality.py wrote).
    def _cached_site(site):
        cache_path = os.path.join(CACHE_DIR, f"{site}_frac{args.fraction}.pkl")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"{cache_path} missing — run finetune_mortality.py first (or with --cache-only) "
                f"so both sites are cached before this script runs."
            )
        import pickle
        with open(cache_path, "rb") as fp:
            return pickle.load(fp)

    site_samples = {}
    for site in ("mimic", "eicu"):
        site_samples[site] = _cached_site(site)
        logging.info(f"[CACHE] Loaded {site}: {len(site_samples[site])} samples")

    site_datasets = {s: ICUDataset(v) for s, v in site_samples.items()}

    cells = []
    for f in result_files:
        with open(f) as fp:
            d = json.load(fp)
        method, source, target, seed = d["method"], d["source"], d["target"], d["seed"]
        true_auroc = d["ood"]["auroc"]
        if true_auroc is None:
            logging.warning(f"{f}: null OOD AUROC, skipping cell")
            continue

        train_loader, val_loader = make_patient_split_loaders(
            site_datasets[source], train_frac=0.8, batch_size=BATCH_SIZE, seed=seed, stratify_key=TASK,
        )
        tgt_loader = DataLoader(site_datasets[target], batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        ckpt_path = os.path.join(FT_CKPT_DIR, f"{method}_{source}to{target}_s{seed}.pt")
        model = fresh_model(seed=seed).to(device)
        model.add_classification_head(TASK)
        model.load_state_dict(load_state_dict_flexible(ckpt_path, device))

        scores = score_checkpoint(model, val_loader, tgt_loader, device, seed=seed)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        cell = {"method": method, "source": source, "target": target, "seed": seed,
                "true_auroc": true_auroc, **scores}
        cells.append(cell)
        logging.info(f"  {method:4s} {source}->{target} s{seed}: "
                     f"viol={scores['violation']:.5f} recon={scores['recon_mse']:.5f} "
                     f"ent={scores['entropy']:.5f} repr_dist={scores['repr_dist']:.3f} "
                     f"atc={scores['atc']:.3f} mmd={scores['mmd']:.5f} true={true_auroc:.4f}")

    # ── pooled Spearman rho vs true OOD AUROC ──────────────────────────────
    def _rho(sig, subset):
        cs = [c for c in subset if c[sig] == c[sig]]  # drop NaN
        return spearman([c[sig] for c in cs], [c["true_auroc"] for c in cs])

    rhos_all = {s: _rho(s, cells) for s in ALL_SIGNALS}
    # Step 5: split recon_mse by whether the method ever saw the physiology
    # constraint during pretraining (pcl) or not (erm, dro).
    constrained = [c for c in cells if c["method"] == "pcl"]
    unconstrained = [c for c in cells if c["method"] in ("erm", "dro")]
    recon_mse_by_method_type = {
        "pcl (constraint-trained)": _rho("recon_mse", constrained),
        "erm+dro (no constraint)": _rho("recon_mse", unconstrained),
    }

    # ── selection regret per direction, averaged score/AUROC across 3 seeds per method ──
    directions = sorted(set((c["source"], c["target"]) for c in cells))
    regret_by_direction = {}
    for source, target in directions:
        dcells = [c for c in cells if c["source"] == source and c["target"] == target]
        methods = sorted(set(c["method"] for c in dcells))
        auroc_by_method = {m: float(np.mean([c["true_auroc"] for c in dcells if c["method"] == m])) for m in methods}
        entry = {}
        for sig in ALL_SIGNALS:
            score_by_method = {m: float(np.mean([c[sig] for c in dcells if c["method"] == m and c[sig] == c[sig]]))
                                for m in methods}
            r, picked = regret(score_by_method, auroc_by_method, lower_is_better=(sig in SIGNALS_LOWER_BETTER))
            entry[sig] = {"regret": r, "picked": picked}
        entry["best_method"] = max(auroc_by_method, key=lambda m: auroc_by_method[m])
        regret_by_direction[f"{source}->{target}"] = entry

    out = {
        "n_cells": len(cells), "signals": ALL_SIGNALS,
        "spearman_all": rhos_all,
        "step5_recon_mse_by_method_type": recon_mse_by_method_type,
        "regret_by_direction": regret_by_direction,
        "cells": cells,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print(f"SELECTION CRITERIA — mortality  ({len(cells)} cells)")
    print("=" * 78)
    print("Spearman rho vs true OOD AUROC (lower-is-better signals want NEGATIVE rho;")
    print("atc wants POSITIVE rho — it's a predicted-accuracy signal, not an error signal).")
    for s in ALL_SIGNALS:
        print(f"  {s:<12} {rhos_all[s]:+.3f}")
    print("\nStep 5 — recon_mse rho split by constraint exposure:")
    for k, v in recon_mse_by_method_type.items():
        print(f"  {k:<26} {v:+.3f}")
    print("\nSelection regret by direction (0 = perfect; picked method in parens):")
    for direction, entry in regret_by_direction.items():
        print(f"  {direction}  (best={entry['best_method']})")
        for sig in ALL_SIGNALS:
            r = entry[sig]
            print(f"    {sig:<12} regret={r['regret']:+.4f}  picked={r['picked']}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
