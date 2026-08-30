"""
deadpcl, per-confound ablation (external review item #7).

"We fixed all five confounds and the result flipped" doesn't say whether one
confound dominated or all five contributed roughly equally. This reintroduces
exactly ONE evaluation confound at a time on top of the otherwise-corrected
pipeline and re-measures zero-shot sepsis AUROC, so the paper can report
per-confound AUROC drift instead of only the two aggregate endpoints (Table 1
= all five present, Table 2 = all five fixed).

Only confounds 2 and 4 need a new training run -- confounds 1, 3, and 5 either
cost nothing (re-analysis of data that already exists) or already have a
number in the paper. See deadpcl/pcl_findings_draft.tex Section 5 for the
full reasoning. This script only implements 2 and 4:

  --confound 2  Pretraining leakage. Pretrain on unlabeled data pooled from
                ALL FOUR sites (PhysioNet A+B, MIMIC-IV, eICU) instead of
                source-only. The standard 3-constraint PhysiologicalConstraintLoss
                is unchanged -- only the pretraining DATA composition differs
                from the corrected baseline.
  --confound 4  Severinghaus circularity. Pretraining stays source-only
                (unchanged), but the SpO2/Severinghaus term is dropped from
                L_phys via SubsetConstraintLoss(excluded_constraint="SpO2") --
                MAP and Henderson-Hasselbalch stay active. This is the "drop
                the loss term, not the PaO2 input" reading of confound 4:
                it isolates the constraint's circularity specifically, not a
                general "does PaO2 matter as a feature" question, which is a
                different ablation this script does not run.

Everything else matches the corrected Table 2 PCL row exactly: 17 variables,
SOFA-derived sepsis labels (the loaders already default to this, so confound 1
is fixed for free just by using them unmodified), lambda=0.5 chosen by
source-domain validation (config.LAMBDA_PCL default, unchanged), fine-tuned
and evaluated zero-shot on Site B / MIMIC-IV / eICU.

Resumable: skips and exits 0 if the result JSON already exists (pass
--overwrite to force a rerun). Run seed 42 alone first for each confound --
its wall-clock time is the only real cost estimate before committing the
other two seeds. Full experiment suite at this scale took ~6h on one H100
for ~30 runs in the original (9-variable) setup (fmain_type2.tex); expect a
single (confound, seed) pretrain+finetune+eval cycle to land in the same
10-20 min ballpark, so 2 confounds x 3 seeds is roughly 1-2 H100-hours total.

Usage (env vars MIMIC_DIR / EICU_DIR / PHYSIONET_DIR must already be
exported, e.g. via runpod_env.sh):
    python run_confound_ablation.py --confound 2 --seed 42
    python run_confound_ablation.py --confound 4 --seed 42
"""
import argparse
import json
import logging
import os
import pickle
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Production checkpoints/architecture only -- same reasoning as
# pcl-legacy2/scripts/finetune_mortality.py: TEST_MODE changes D_MODEL/N_LAYERS
# themselves, not just data volume, so it must be forced off before ANY
# `from config import ...` anywhere in this process, or the run silently
# builds an incompatible model and only errors after both datasets are read.
os.environ["PCL_TEST_MODE"] = "0"

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEADPCL_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_DEADPCL_ROOT)
_CHAT1 = os.path.join(_REPO_ROOT, "chat1_protocol")
sys.path.insert(0, _CHAT1)
sys.path.insert(0, _REPO_ROOT)  # for pod_monitor.py, shared across all PCL sub-projects

OUT_DIR = os.path.join(_DEADPCL_ROOT, "results", "ablation")
PRETRAIN_CKPT_DIR = os.path.join(OUT_DIR, "ckpt_pretrain")
FT_CKPT_DIR = os.path.join(OUT_DIR, "ckpt_finetune")
CACHE_DIR = os.path.join(OUT_DIR, "cache")

TASK = "sepsis"
SITES = ("site_b", "mimic", "eicu")

# Table 2's already-published corrected PCL row, sourced from
# chat2_papers/paper/build_urtc.py -- printed alongside each new result as a
# reference point ONLY, never overwritten by this script. If this drifts from
# the live paper, the paper is the source of truth, not this constant.
REFERENCE_PCL_AUROC = {"source": 0.798, "site_b": 0.655, "mimic": 0.622, "eicu": 0.572}


def _load_site_cached(loader_name, fraction, seed):
    """Cached full-site load. Mirrors finetune_mortality.py's _load_site: the
    full CSV/PSV read takes minutes and this script re-invokes it across
    seeds and confounds, so without a cache every run re-pays that cost in
    pure CPU-bound GPU-pod idle time -- exactly what pod_monitor.py exists to
    flag if you forget to switch pods during it."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{loader_name}_frac{fraction}_seed{seed}.pkl")
    if os.path.exists(cache_path):
        logging.info(f"[CACHE] Loading {loader_name} samples from {cache_path}")
        with open(cache_path, "rb") as fp:
            return pickle.load(fp)

    from config import PHYSIONET_DIR, MIMIC_DIR, EICU_DIR
    from src.data.physionet2019 import load_physionet2019
    from src.data.mimic4 import load_mimic4
    from src.data.eicu import load_eicu

    if loader_name == "physionet_a":
        samples, _ = load_physionet2019(PHYSIONET_DIR, fraction=fraction, sites=[0], seed=seed)
    elif loader_name == "physionet_b":
        samples, _ = load_physionet2019(PHYSIONET_DIR, fraction=fraction, sites=[1], seed=seed)
    elif loader_name == "mimic":
        samples, _ = load_mimic4(MIMIC_DIR, fraction=fraction, seed=seed)
    elif loader_name == "eicu":
        samples, _ = load_eicu(EICU_DIR, fraction=fraction, seed=seed)
    else:
        raise ValueError(loader_name)

    try:
        with open(cache_path, "wb") as fp:
            pickle.dump(samples, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"[CACHE] Saved {loader_name} samples ({len(samples)}) -> {cache_path}")
    except Exception as e:
        logging.warning(f"[CACHE] Could not write {loader_name} cache: {e}")

    return samples


def _build_pretrain_pool(confound, fraction, seed):
    """Confound 2: pool all four sites, unlabeled, for pretraining (reintroduces
    the leak). Confound 4: source-only, matching the corrected baseline (the
    confound here is in the loss function, not the data)."""
    site_a = _load_site_cached("physionet_a", fraction, seed)
    if confound == 2:
        pool = (site_a
                + _load_site_cached("physionet_b", fraction, seed)
                + _load_site_cached("mimic", fraction, seed)
                + _load_site_cached("eicu", fraction, seed))
        logging.info(f"[confound 2] pretraining pool: {len(pool)} stays pooled across "
                      f"all 4 sites (leak reintroduced)")
        return pool
    logging.info(f"[confound 4] pretraining pool: {len(site_a)} stays, source-only "
                  f"(unchanged from corrected baseline)")
    return site_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confound", type=int, choices=[2, 4], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fraction", type=float, default=1.0,
                     help="Subsample fraction per site, for a cheap dry run before full scale.")
    ap.add_argument("--pretrain-epochs", type=int, default=None,
                     help="Override PRETRAIN_EPOCHS (config default 30 full-scale). "
                          "Use a small value only to sanity-check plumbing.")
    ap.add_argument("--finetune-epochs", type=int, default=None,
                     help="Override FINETUNE_EPOCHS (config default 30 full-scale).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cache-only", action="store_true",
                     help="Load + cache every site's data, then exit before touching the "
                          "GPU/model. Run this on a cheap CPU pod; the cache lives on the "
                          "network volume, so switching to the GPU pod afterwards hits it "
                          "immediately instead of re-reading the full CSVs there.")
    args = ap.parse_args()

    tag = f"confound{args.confound}_s{args.seed}"
    out_path = os.path.join(OUT_DIR, f"{tag}.json")

    if os.path.exists(out_path) and not args.overwrite:
        logging.info(f"[RESUME] {out_path} already exists — skipping. Pass --overwrite to force.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PRETRAIN_CKPT_DIR, exist_ok=True)
    os.makedirs(FT_CKPT_DIR, exist_ok=True)

    from pod_monitor import watch_pod
    watch_pod(verbose=True)  # loud reminder to switch pods if GPU sits idle during the
                              # data-load phase below, or if you're still on a cheap pod
                              # once pretraining actually starts

    from config import TEST_MODE, D_MODEL, N_LAYERS
    if TEST_MODE or D_MODEL != 256 or N_LAYERS != 6:
        raise RuntimeError(
            f"config resolved to TEST_MODE={TEST_MODE}, D_MODEL={D_MODEL}, N_LAYERS={N_LAYERS} — "
            "incompatible with the production-scale comparison this ablation needs. "
            "This should be impossible (script forces PCL_TEST_MODE=0); check for another "
            "PCL_TEST_MODE export shadowing it, or a stale config.py."
        )

    pretrain_samples = _build_pretrain_pool(args.confound, args.fraction, args.seed)
    if args.cache_only:
        logging.info("[CACHE-ONLY] Pretrain pool cached. Also warming target-site caches "
                      "for the eval step, then exiting before touching the GPU.")
        for name in ("physionet_b", "mimic", "eicu"):
            _load_site_cached(name, args.fraction, args.seed)
        return

    import torch
    from torch.utils.data import DataLoader
    from config import (BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, LAMBDA_PCL, MASK_PROB,
                         PRETRAIN_EPOCHS as _CFG_PT_EPOCHS, FINETUNE_EPOCHS as _CFG_FT_EPOCHS)
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset, make_patient_split_loaders
    from src.eval.evaluate_utils import evaluate_model, run_finetuning
    from src.training.train_utils import run_pretraining, load_state_dict_flexible
    from src.losses.pcl_loss import PhysiologicalConstraintLoss
    from src.ablations import SubsetConstraintLoss

    pretrain_epochs = args.pretrain_epochs if args.pretrain_epochs is not None else _CFG_PT_EPOCHS
    finetune_epochs = args.finetune_epochs if args.finetune_epochs is not None else _CFG_FT_EPOCHS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Run: confound={args.confound} seed={args.seed} lambda={LAMBDA_PCL} "
                 f"pretrain_epochs={pretrain_epochs} finetune_epochs={finetune_epochs} "
                 f"device={device}")

    t0 = time.time()

    # ── Pretrain (the actual ablated step) ──────────────────────────────────
    pretrain_ds = ICUDataset(pretrain_samples)
    pretrain_ds.coverage_report()
    pt_train_loader, pt_val_loader = make_patient_split_loaders(
        pretrain_ds, train_frac=0.8, batch_size=BATCH_SIZE, seed=args.seed,
        stratify_key=TASK,
    )

    if args.confound == 2:
        pcl_loss_fn = PhysiologicalConstraintLoss()          # unchanged: full 3-constraint loss
    else:
        pcl_loss_fn = SubsetConstraintLoss(excluded_constraint="SpO2")  # drop Severinghaus term

    model = fresh_model(seed=args.seed)
    pretrain_ckpt = os.path.join(PRETRAIN_CKPT_DIR, f"{tag}.pt")
    run_pretraining(
        model, pcl_loss_fn, pt_train_loader, pt_val_loader,
        n_epochs=pretrain_epochs, lam=LAMBDA_PCL, mask_prob=MASK_PROB,
        device=device, save_path=pretrain_ckpt,
    )
    # run_pretraining checkpoints the BEST val-loss epoch to disk but does not
    # reload it into `model` afterwards (same as run_finetuning below) — the
    # in-memory model after the loop holds the LAST epoch, not the best one.
    model.load_state_dict(load_state_dict_flexible(pretrain_ckpt, device="cpu"))
    logging.info(f"Pretraining complete (confound {args.confound}): {pretrain_ckpt}")

    # ── Fine-tune on source-site sepsis labels ──────────────────────────────
    model.add_classification_head(TASK)
    src_ds = ICUDataset(_load_site_cached("physionet_a", args.fraction, args.seed))
    n_pos = sum(1 for s in src_ds.samples if s[TASK].item() > 0)
    if n_pos == 0 or n_pos == len(src_ds):
        raise RuntimeError(
            f"Source site has {n_pos}/{len(src_ds)} {TASK}+ — single-class, cannot fine-tune."
        )
    ft_train_loader, ft_val_loader = make_patient_split_loaders(
        src_ds, train_frac=0.8, batch_size=BATCH_SIZE, seed=args.seed, stratify_key=TASK,
    )
    ft_ckpt = os.path.join(FT_CKPT_DIR, f"{tag}.pt")
    run_finetuning(
        model, ft_train_loader, ft_val_loader, TASK,
        n_epochs=finetune_epochs, device=device, save_path=ft_ckpt,
    )
    model.load_state_dict(load_state_dict_flexible(ft_ckpt, device))

    # ── Evaluate: in-domain (source val) + zero-shot OOD (3 target sites) ──
    in_domain = evaluate_model(model, ft_val_loader, TASK, device=device,
                                split_name=f"{tag} in-domain (source)")
    ood = {}
    for site_key, loader_name in (("site_b", "physionet_b"), ("mimic", "mimic"), ("eicu", "eicu")):
        tgt_samples = _load_site_cached(loader_name, args.fraction, args.seed)
        tgt_ds = ICUDataset(tgt_samples)
        tgt_ds.coverage_report()
        tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        ood[site_key] = evaluate_model(model, tgt_loader, TASK, device=device,
                                        split_name=f"{tag} zero-shot OOD ({site_key})")

    elapsed = time.time() - t0
    auroc = {"source": in_domain.get("auroc")}
    auroc.update({k: v.get("auroc") for k, v in ood.items()})
    drift = {k: (auroc[k] - REFERENCE_PCL_AUROC[k]) if auroc[k] is not None else None
             for k in REFERENCE_PCL_AUROC}

    result = {
        "confound": args.confound, "seed": args.seed, "task": TASK,
        "pretrain_epochs": pretrain_epochs, "finetune_epochs": finetune_epochs,
        "fraction": args.fraction, "lambda": LAMBDA_PCL,
        "n_pretrain": len(pretrain_ds), "n_source": len(src_ds),
        "auroc": auroc, "reference_table2_pcl_auroc": REFERENCE_PCL_AUROC,
        "auroc_drift_vs_table2": drift,
        "elapsed_sec": elapsed,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logging.info(f"Saved {out_path} | elapsed={elapsed/60:.1f} min")
    logging.info(f"AUROC drift vs Table 2 PCL row (this - reference): "
                 f"{ {k: round(v, 4) if v is not None else None for k, v in drift.items()} }")
    logging.info(
        f"Projected cost for the other 2 seeds of confound {args.confound} at this pace: "
        f"~{2 * elapsed / 3600:.2f} GPU-hours (excludes data-load caching speedup on reruns)"
    )


if __name__ == "__main__":
    main()
