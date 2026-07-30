"""
Data-efficiency ablation: OOD AUROC vs training data fraction for ERM vs PCL.

Trains on 10%, 25%, 50%, and 100% of Site A (PhysioNet 2019) and evaluates
zero-shot on Site B. Expected finding: PCL's advantage is larger in low-data
regimes, directly answering the reviewer's attention mechanism criticism.

Usage:
    PCL_TEST_MODE=1 python scripts/data_scaling.py   # quick smoke test
    PCL_TEST_MODE=0 python scripts/data_scaling.py   # full run
"""
import json
import logging
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PHYSIONET_DIR, RESULTS_DIR, CHECKPOINT_DIR,
    BATCH_SIZE, SEED, PRETRAIN_EPOCHS, FINETUNE_EPOCHS,
    LAMBDA_PCL, NUM_WORKERS, PIN_MEMORY, USE_AMP,
    DATA_FRACTION,
)
from src.baselines import fresh_model, run_erm_pretraining
from src.data.physionet2019 import load_physionet2019
from src.data.dataset import ICUDataset
from src.eval.evaluate_utils import run_finetuning, evaluate_model
from src.losses.pcl_loss import PhysiologicalConstraintLoss
from src.training.train_utils import run_pretraining, load_state_dict_flexible

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

FRACTIONS = [0.10, 0.25, 0.50, 1.00]
TASK = "los_3d"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _train_eval(train_indices, site_b_indices, dataset, use_pcl: bool, seed: int, fraction: float):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Stratified train/val split of site A subset
    from sklearn.model_selection import StratifiedShuffleSplit
    labels = np.array([dataset.samples[i][TASK].item() for i in train_indices])
    if len(np.unique(labels)) >= 2 and len(train_indices) > 10:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        idx_arr = np.arange(len(train_indices))
        tr_sub, va_sub = next(sss.split(idx_arr, labels))
        tr_final = [train_indices[i] for i in tr_sub]
        va_final = [train_indices[i] for i in va_sub]
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(train_indices))
        n_val = max(1, int(len(train_indices) * 0.2))
        va_final = [train_indices[i] for i in perm[:n_val]]
        tr_final = [train_indices[i] for i in perm[n_val:]]

    if len(tr_final) < 4 or len(va_final) < 2:
        logging.warning(f"Skipping fraction={fraction}: too few samples (train={len(tr_final)}, val={len(va_final)})")
        return float("nan")

    bs = min(BATCH_SIZE, len(tr_final))
    train_loader = DataLoader(Subset(dataset, tr_final), batch_size=bs, shuffle=True,
                              drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(Subset(dataset, va_final), batch_size=bs, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(Subset(dataset, site_b_indices), batch_size=bs, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = fresh_model(seed=seed).to(DEVICE)

    label_str = "PCL" if use_pcl else "ERM"
    pre_path = os.path.join(RESULTS_DIR, f"scale_{label_str}_{fraction:.2f}_pre.pt")
    ft_path = os.path.join(RESULTS_DIR, f"scale_{label_str}_{fraction:.2f}_ft.pt")

    if use_pcl:
        run_pretraining(
            model, PhysiologicalConstraintLoss(), train_loader, val_loader,
            n_epochs=PRETRAIN_EPOCHS, lam=LAMBDA_PCL, device=DEVICE,
            save_path=pre_path, warmup_fraction=0.4,
        )
    else:
        run_erm_pretraining(
            model, train_loader, val_loader,
            n_epochs=PRETRAIN_EPOCHS, device=DEVICE, save_path=pre_path,
        )

    if os.path.exists(pre_path):
        model.load_state_dict(load_state_dict_flexible(pre_path, DEVICE))

    model.add_classification_head(TASK)
    run_finetuning(
        model, train_loader, val_loader, TASK,
        n_epochs=FINETUNE_EPOCHS, device=DEVICE, save_path=ft_path,
    )

    if os.path.exists(ft_path):
        model.load_state_dict(load_state_dict_flexible(ft_path, DEVICE))

    result = evaluate_model(model, test_loader, TASK, device=DEVICE,
                            split_name=f"{label_str} frac={fraction:.0%} OOD")

    for p in [pre_path, ft_path]:
        if os.path.exists(p):
            os.remove(p)

    return float(result["auroc"]) if result.get("auroc") is not None else float("nan")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logging.info("Loading PhysioNet 2019...")
    samples, _ = load_physionet2019(PHYSIONET_DIR, fraction=DATA_FRACTION)
    dataset = ICUDataset(samples)

    site_a_all = [i for i, s in enumerate(dataset.samples) if s["site_id"].item() == 0]
    site_b_all = [i for i, s in enumerate(dataset.samples) if s["site_id"].item() == 1]

    if not site_b_all:
        logging.warning("No Site B samples — using last 20%% of Site A as OOD proxy")
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(site_a_all))
        n_b = max(1, len(site_a_all) // 5)
        site_b_all = [site_a_all[i] for i in perm[:n_b]]
        site_a_all = [site_a_all[i] for i in perm[n_b:]]

    logging.info(f"Site A: {len(site_a_all)} patients, Site B: {len(site_b_all)} patients")

    xs, erm_curve, pcl_curve = [], [], []

    for frac in FRACTIONS:
        n_use = max(8, int(len(site_a_all) * frac))
        rng = np.random.default_rng(SEED + int(frac * 100))
        chosen = rng.choice(len(site_a_all), size=n_use, replace=False).tolist()
        train_indices = [site_a_all[i] for i in chosen]

        logging.info(f"\n=== Fraction={frac:.0%} ({n_use} patients) ===")
        seed = SEED + int(frac * 1000)

        erm_auroc = _train_eval(train_indices, site_b_all, dataset,
                                use_pcl=False, seed=seed, fraction=frac)
        pcl_auroc = _train_eval(train_indices, site_b_all, dataset,
                                use_pcl=True, seed=seed + 1, fraction=frac)

        logging.info(f"  ERM OOD AUROC: {erm_auroc:.4f}  PCL OOD AUROC: {pcl_auroc:.4f}")
        xs.append(frac)
        erm_curve.append(erm_auroc)
        pcl_curve.append(pcl_auroc)

    out = {"fractions": xs, "erm_ood_auroc": erm_curve, "pcl_ood_auroc": pcl_curve, "task": TASK}
    out_path = os.path.join(RESULTS_DIR, "data_efficiency_ablation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logging.info(f"Results saved to {out_path}")

    pct_labels = [f"{int(f * 100)}%" for f in xs]
    plt.figure(figsize=(7, 5))
    plt.plot(pct_labels, erm_curve, marker="s", label="ERM", color="#c44e52", linewidth=2, markersize=8)
    plt.plot(pct_labels, pcl_curve, marker="o", label="PCL", color="#4c72b0", linewidth=2, markersize=8)
    plt.xlabel("Training data fraction (Site A)")
    plt.ylabel(f"OOD AUROC ({TASK}, Site B)")
    plt.title("Data-Efficiency Ablation: PCL vs ERM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "data_efficiency_ablation.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    logging.info(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
