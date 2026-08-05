"""
OOD stress tests: Gaussian input noise (robustness) + optional mmHg measurement bias (ProjOVERVIEW Exp. 5).
Compares ERM vs PCL after identical pretraining+fine-tuning protocols.
"""
import json
import logging
import os
import sys
import numpy as np
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_DIR,
    FINETUNE_EPOCHS_DEMO,
    NOISE_LEVELS,
    PRETRAIN_EPOCHS_DEMO,
    RESULTS_DIR,
    SEED,
)
from src.ablations import inject_measurement_noise
import copy
from src.baselines import fresh_model, run_erm_pretraining
from src.data_utils import ICUDataset, VARIABLES, attach_labels, load_stays
from src.eval.evaluate_utils import evaluate_model, evaluate_model_noisy_inputs, run_finetuning
from src.losses.pcl_loss import PhysiologicalConstraintLoss
from src.training.train_utils import load_state_dict_flexible, run_pretraining

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def inject_bp_noise(dataset, bias_mmhg: float):
    """Apply explicit signed SBP/DBP bias to simulate a miscalibrated cuff.
    
    Critical: MAP is NOT updated. In real hospitals, MAP is either measured
    independently (arterial line) or computed from a separate oscillometric
    algorithm. A miscalibrated cuff shifts SBP/DBP but leaves MAP unchanged,
    creating a physiological inconsistency that tests whether the model's
    representations are robust to measurement protocol shifts.
    """
    noisy = copy.deepcopy(dataset)
    from src.data_utils import VARIABLES

    i_sbp = VARIABLES.index("SBP")
    i_dbp = VARIABLES.index("DBP")
    sbp_norm_inc = bias_mmhg / (300 - 40)
    dbp_norm_inc = bias_mmhg / (200 - 20)

    for sample in noisy.samples:
        x = sample["x"]
        m = sample["mask"]
        x2 = x.clone()
        
        # Apply bias to SBP and DBP only — MAP is NOT updated
        mask_sbp = m[:, i_sbp]
        mask_dbp = m[:, i_dbp]
        x2[mask_sbp, i_sbp] = torch.clamp(x2[mask_sbp, i_sbp] + sbp_norm_inc, 0.0, 1.0)
        x2[mask_dbp, i_dbp] = torch.clamp(x2[mask_dbp, i_dbp] + dbp_norm_inc, 0.0, 1.0)
            
        sample["x"] = x2
    return noisy

def train_and_finetune_erm(train_loader, val_loader, test_loader, device):
    path_pre = os.path.join(CHECKPOINT_DIR, "erm_pretrained.pt")
    path_ft = os.path.join(CHECKPOINT_DIR, "erm_mortality.pt")
    model = fresh_model(n_vars=len(VARIABLES), seed=SEED).to(device)
    run_erm_pretraining(
        model, train_loader, val_loader, n_epochs=PRETRAIN_EPOCHS_DEMO, save_path=path_pre, device=device
    )
    model.load_state_dict(load_state_dict_flexible(path_pre, device))
    model.add_classification_head("mortality")
    run_finetuning(
        model,
        train_loader,
        val_loader,
        "mortality",
        n_epochs=FINETUNE_EPOCHS_DEMO,
        device=device,
        save_path=path_ft,
    )
    model.load_state_dict(load_state_dict_flexible(path_ft, device))
    clean = evaluate_model(model, test_loader, "mortality", device=device, split_name="ERM clean")
    noisy = evaluate_model_noisy_inputs(
        model, test_loader, "mortality", device=device, noise_std=0.1, split_name="ERM + Gaussian noise"
    )
    return model, clean, noisy


def train_and_finetune_pcl(train_loader, val_loader, test_loader, device):
    path_pre = os.path.join(CHECKPOINT_DIR, "pcl_pretrained.pt")
    path_ft = os.path.join(CHECKPOINT_DIR, "pcl_mortality.pt")
    model = fresh_model(n_vars=len(VARIABLES), seed=SEED + 2).to(device)
    run_pretraining(
        model,
        PhysiologicalConstraintLoss(),
        train_loader,
        val_loader,
        n_epochs=PRETRAIN_EPOCHS_DEMO,
        save_path=path_pre,
        device=device,
    )
    model.load_state_dict(load_state_dict_flexible(path_pre, device))
    model.add_classification_head("mortality")
    run_finetuning(
        model,
        train_loader,
        val_loader,
        "mortality",
        n_epochs=FINETUNE_EPOCHS_DEMO,
        device=device,
        save_path=path_ft,
    )

    model.load_state_dict(load_state_dict_flexible(path_ft, device))
    clean = evaluate_model(model, test_loader, "mortality", device=device, split_name="PCL clean")
    noisy = evaluate_model_noisy_inputs(
        model, test_loader, "mortality", device=device, noise_std=0.1, split_name="PCL + Gaussian noise"
    )
    return model, clean, noisy


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stays = load_stays(DATA_DIR).head(100)
    dataset = ICUDataset(DATA_DIR, stays)
    attach_labels(dataset, stays)

    from sklearn.model_selection import StratifiedShuffleSplit
    labels_all = np.array([s["mortality"].item() for s in dataset.samples])
    n = len(dataset)

    # 70% train, 15% val, 15% test — stratified at each step
    sss_tv = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, tvt_idx = next(sss_tv.split(np.arange(n), labels_all))

    sss_vt = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(sss_vt.split(tvt_idx, labels_all[tvt_idx]))
    val_idx  = tvt_idx[val_idx]
    test_idx = tvt_idx[test_idx]

    n_pos_train = int(labels_all[train_idx].sum())
    n_pos_val   = int(labels_all[val_idx].sum())
    n_pos_test  = int(labels_all[test_idx].sum())
    logging.info(
        f"Stratified split — train: {len(train_idx)} ({n_pos_train} pos) | "
        f"val: {len(val_idx)} ({n_pos_val} pos) | "
        f"test: {len(test_idx)} ({n_pos_test} pos)"
    )
    if n_pos_test == 0:
        logging.warning("Test set has 0 positives — AUROC will be undefined. Increase dataset size.")

    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(Subset(dataset, val_idx.tolist()),   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(Subset(dataset, test_idx.tolist()),  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    _, erm_c, erm_n = train_and_finetune_erm(train_loader, val_loader, test_loader, device)
    _, pcl_c, pcl_n = train_and_finetune_pcl(train_loader, val_loader, test_loader, device)

    def _auroc(d):
        return float(d["auroc"]) if d.get("auroc") is not None else float("nan")

    def _auprc(d):
        return float(d["auprc"]) if d.get("auprc") is not None else float("nan")

    def _brier(d):
        return float(d["brier"]) if d.get("brier") is not None else float("nan")

    metrics = {
        "erm_auroc_clean": _auroc(erm_c),
        "erm_auprc_clean": _auprc(erm_c),
        "erm_brier_clean": _brier(erm_c),
        "erm_auroc_gaussian_noise": _auroc(erm_n),
        "pcl_auroc_clean": _auroc(pcl_c),
        "pcl_auprc_clean": _auprc(pcl_c),
        "pcl_brier_clean": _brier(pcl_c),
        "pcl_auroc_gaussian_noise": _auroc(pcl_n),
        "drop_erm": _auroc(erm_c) - _auroc(erm_n),
        "drop_pcl": _auroc(pcl_c) - _auroc(pcl_n),
    }
    with open(os.path.join(RESULTS_DIR, "metrics_ood.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    labels = ["ERM clean", "ERM noisy", "PCL clean", "PCL noisy"]
    vals = [_auroc(erm_c), _auroc(erm_n), _auroc(pcl_c), _auroc(pcl_n)]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals, color=["#c44e52", "#e7969c", "#4c72b0", "#8c9cb9"])
    plt.ylabel("AUROC (mortality)")
    plt.title("OOD stress test: Gaussian noise on inputs (σ=0.1)")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ood_gaussian_bar.png"), dpi=200)
    plt.close()

    noise_mmhg = list(NOISE_LEVELS)
    curve_erm = []
    curve_pcl = []
    for sig in noise_mmhg:
        if sig == 0:
            noisy_ds = dataset
        else:
            noisy_ds = inject_bp_noise(dataset, float(sig))
        sub = Subset(noisy_ds, test_idx)
        ld = DataLoader(sub, batch_size=BATCH_SIZE)
        erm_model = fresh_model(n_vars=len(VARIABLES), seed=SEED).to(device)
        erm_model.add_classification_head("mortality")
        erm_model.load_state_dict(
            load_state_dict_flexible(os.path.join(CHECKPOINT_DIR, "erm_mortality.pt"), device)
        )
        r_erm = evaluate_model(erm_model, ld, "mortality", device=device, split_name=f"ERM σ={sig}mmHg")
        curve_erm.append(_auroc(r_erm))

        pcl_model = fresh_model(n_vars=len(VARIABLES), seed=SEED + 2).to(device)
        pcl_model.add_classification_head("mortality")
        pcl_model.load_state_dict(
            load_state_dict_flexible(os.path.join(CHECKPOINT_DIR, "pcl_mortality.pt"), device)
        )
        r = evaluate_model(pcl_model, ld, "mortality", device=device, split_name=f"PCL σ={sig}mmHg")
        curve_pcl.append(_auroc(r))

    plt.figure(figsize=(7, 5))
    plt.plot(noise_mmhg, curve_erm, marker="s", color="#c44e52", linewidth=2, label="ERM (fine-tuned)")
    plt.plot(noise_mmhg, curve_pcl, marker="o", color="#4c72b0", linewidth=2, label="PCL (fine-tuned)")
    plt.xlabel("Simulated BP measurement bias (mmHg)")
    plt.ylabel("AUROC")
    plt.title("Measurement noise robustness (PCL model)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "noise_robustness.png"), dpi=200)
    plt.close()

    metrics["noise_curve_mmhg"] = noise_mmhg
    metrics["erm_noise_curve_auroc"] = curve_erm
    metrics["pcl_noise_curve_auroc"] = curve_pcl
    with open(os.path.join(RESULTS_DIR, "metrics_ood.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logging.info("Saved metrics to %s", os.path.join(RESULTS_DIR, "metrics_ood.json"))


if __name__ == "__main__":
    main()
