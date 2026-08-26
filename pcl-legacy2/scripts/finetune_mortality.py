"""
pcl-legacy2, Step 3/4 — mortality task.

Fine-tunes a classification head on top of an already-pretrained encoder
(ERM/PCL/DRO, reused as-is from URTC — see ../README.md) for in-hospital
mortality, one (method, source-site, seed) combination per invocation.

Both directions matter here (see README): MIMIC-IV (single hospital) and
eICU-CRD (200+ hospitals) are structurally different enough that direction is
not interchangeable, so this is run source=mimic and source=eicu separately,
not averaged or treated as one design.

Uses `mortality_hospital`, NOT `mortality` — eICU's `mortality` field is
ICU-unit-level, not hospital-level; using it here would silently compare two
different quantities across sites. See src/data/eicu.py and dataset.py.

Resumable: skips and exits 0 if the result JSON already exists (pass
--overwrite to force a rerun). Run the first combination alone first — its
wall-clock time is the only real cost estimate for the full 18-run sweep
before committing further budget.

Usage (env vars MIMIC_DIR / EICU_DIR must already be exported, e.g. via
runpod_env.sh):
    python finetune_mortality.py --method erm --source mimic --seed 42
"""
import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEGACY2_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_LEGACY2_ROOT)
_CHAT1 = os.path.join(_REPO_ROOT, "chat1_protocol")
sys.path.insert(0, _CHAT1)

# Where the real URTC-era pretrained encoders live — confirmed on the pod
# (13MB each, full-scale, task=sepsis pretraining objective; pretraining
# is task-agnostic per README so that's fine to reuse for mortality).
# Override via env var if the pod ever reorganizes this.
PRETRAIN_CKPT_DIR = os.environ.get(
    "PCL_LEGACY2_PRETRAIN_DIR",
    os.path.join(_REPO_ROOT, "results_lambda17", "ckpt"),
)

OUT_DIR = os.path.join(_LEGACY2_ROOT, "results", "mortality")
FT_CKPT_DIR = os.path.join(OUT_DIR, "ckpt")

TASK = "mortality_hospital"
SITE_DIRS = {"mimic": None, "eicu": None}  # filled after config import (needs env vars set first)


def _load_site(site, fraction, seed):
    from src.data.mimic4 import load_mimic4
    from src.data.eicu import load_eicu
    from config import MIMIC_DIR, EICU_DIR

    if site == "mimic":
        samples, _ = load_mimic4(MIMIC_DIR, fraction=fraction, seed=seed)
    elif site == "eicu":
        samples, _ = load_eicu(EICU_DIR, fraction=fraction, seed=seed)
    else:
        raise ValueError(site)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["erm", "pcl", "dro"], required=True)
    ap.add_argument("--source", choices=["mimic", "eicu"], required=True,
                     help="Fine-tune site. The other site is the zero-shot OOD target.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fraction", type=float, default=1.0,
                     help="Subsample fraction per site, for a cheap dry run before full scale.")
    ap.add_argument("--epochs", type=int, default=None,
                     help="Override FINETUNE_EPOCHS (config default is 30 full-scale). "
                          "Use a small value only to sanity-check plumbing, not for real results.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    target = "eicu" if args.source == "mimic" else "mimic"
    tag = f"{args.method}_{args.source}to{target}_s{args.seed}"
    out_path = os.path.join(OUT_DIR, f"{tag}.json")

    if os.path.exists(out_path) and not args.overwrite:
        logging.info(f"[RESUME] {out_path} already exists — skipping. Pass --overwrite to force.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FT_CKPT_DIR, exist_ok=True)

    import torch
    from torch.utils.data import DataLoader
    from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    from config import FINETUNE_EPOCHS as _CFG_FT_EPOCHS
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset, make_patient_split_loaders
    from src.eval.evaluate_utils import evaluate_model, run_finetuning
    from src.training.train_utils import load_state_dict_flexible

    n_epochs = args.epochs if args.epochs is not None else _CFG_FT_EPOCHS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Run: method={args.method} source={args.source} target={target} "
                 f"seed={args.seed} epochs={n_epochs} device={device}")

    t0 = time.time()

    # ── Source site: fine-tune + in-domain val ─────────────────────────────
    src_samples = _load_site(args.source, args.fraction, args.seed)
    src_ds = ICUDataset(src_samples)
    src_ds.coverage_report()
    n_pos = sum(1 for s in src_ds.samples if s[TASK].item() > 0)
    if n_pos == 0 or n_pos == len(src_ds):
        raise RuntimeError(
            f"{args.source} has {n_pos}/{len(src_ds)} {TASK}+ — single-class, cannot fine-tune. "
            "Check mortality_hospital extraction before proceeding."
        )
    train_loader, val_loader = make_patient_split_loaders(
        src_ds, train_frac=0.8, batch_size=BATCH_SIZE, seed=args.seed, stratify_key=TASK,
    )

    # ── Target site: zero-shot OOD, full site, no split ────────────────────
    tgt_samples = _load_site(target, args.fraction, args.seed)
    tgt_ds = ICUDataset(tgt_samples)
    tgt_ds.coverage_report()
    tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # ── Load pretrained encoder (reused as-is, no retraining) ──────────────
    pretrain_path = os.path.join(PRETRAIN_CKPT_DIR, f"{args.method}_pretrained.pt")
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(
            f"{pretrain_path} not found — expected the URTC-era pretrained checkpoint here. "
            "Do not fall back to training a fresh encoder silently."
        )
    model = fresh_model(seed=args.seed)
    model.load_state_dict(load_state_dict_flexible(pretrain_path, device="cpu"))
    model.add_classification_head(TASK)
    logging.info(f"Loaded pretrained encoder: {pretrain_path}")

    ft_ckpt_path = os.path.join(FT_CKPT_DIR, f"{tag}.pt")
    run_finetuning(
        model, train_loader, val_loader, TASK,
        n_epochs=n_epochs, device=device, save_path=ft_ckpt_path,
    )
    model.load_state_dict(load_state_dict_flexible(ft_ckpt_path, device))

    # ── Evaluate: in-domain (source val) + zero-shot OOD (target) ──────────
    in_domain = evaluate_model(model, val_loader, TASK, device=device,
                                split_name=f"{tag} in-domain ({args.source})")
    ood = evaluate_model(model, tgt_loader, TASK, device=device,
                          split_name=f"{tag} zero-shot OOD ({target})")

    elapsed = time.time() - t0
    result = {
        "method": args.method, "source": args.source, "target": target,
        "seed": args.seed, "task": TASK, "epochs": n_epochs, "fraction": args.fraction,
        "n_source": len(src_ds), "n_target": len(tgt_ds),
        "in_domain": in_domain, "ood": ood,
        "elapsed_sec": elapsed,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logging.info(f"Saved {out_path} | elapsed={elapsed/60:.1f} min")
    logging.info(
        f"Projected cost for remaining 17 runs at this pace: "
        f"~{17 * elapsed / 3600:.2f} GPU-hours (excludes data-load caching speedup on reruns)"
    )


if __name__ == "__main__":
    main()
