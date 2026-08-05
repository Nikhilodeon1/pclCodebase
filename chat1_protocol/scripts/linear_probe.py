"""
Mechanistic linear probing (ProjOVERVIEW): train only a linear layer on frozen representations
to predict a held-out physiological variable (pH) — compare ERM vs PCL checkpoints.
"""
import json
import logging
import os
import sys

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BATCH_SIZE, CHECKPOINT_DIR, DATA_DIR, PRETRAIN_EPOCHS_DEMO, RESULTS_DIR, SEED
from src.baselines import fresh_model, run_erm_pretraining
from src.data_utils import ICUDataset, VARIABLES, attach_labels, load_stays
from src.losses.pcl_loss import PhysiologicalConstraintLoss
from src.training.train_utils import load_state_dict_flexible, make_loaders, run_pretraining

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class LinearProbe(nn.Module):
    def __init__(self, d_model: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(d_model, out_features)

    def forward(self, reps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Find index of last observed timestep
        t_obs = mask.any(dim=-1) 
        last_idx = t_obs.sum(dim=1).long() - 1
        last_idx = torch.clamp(last_idx, min=0)
        
        batch_idx = torch.arange(reps.size(0), device=reps.device)
        pooled = reps[batch_idx, last_idx]
        return self.linear(pooled)


def train_probe_mse(
    model: nn.Module,
    loader: DataLoader,
    target_var_name: str,
    device: str,
    epochs: int = 12,
) -> float:
    target_var_idx = VARIABLES.index(target_var_name)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    probe = LinearProbe(model.encoder.d_model, 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    last_loss = 0.0
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            with torch.no_grad():
                reps = model.encode(x, mask)
            
            # Target: last observed value of the target variable
            # Find index of last observed timestep for this specific variable
            v_mask = mask[:, :, target_var_idx]
            last_v_idx = v_mask.sum(dim=1).long() - 1
            last_v_idx = torch.clamp(last_v_idx, min=0)
            
            batch_idx = torch.arange(x.size(0), device=x.device)
            target = x[batch_idx, last_v_idx, target_var_idx].unsqueeze(-1)
            
            pred = probe(reps, mask)
            loss = criterion(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        last_loss = total / max(n, 1)
        logging.info(
            "Probe '%s' | epoch %s/%s | train MSE=%.5f",
            target_var_name,
            epoch + 1,
            epochs,
            last_loss,
        )
    return last_loss


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stays = load_stays(DATA_DIR).head(100)
    dataset = ICUDataset(DATA_DIR, stays)
    attach_labels(dataset, stays)
    train_loader, val_loader = make_loaders(dataset, train_frac=0.8, batch_size=BATCH_SIZE, seed=SEED)

    erm_path = os.path.join(CHECKPOINT_DIR, "erm_pretrained.pt")
    pcl_path = os.path.join(CHECKPOINT_DIR, "pcl_pretrained.pt")

    logging.info("Pretraining ERM baseline (masked prediction only)...")
    erm = fresh_model(n_vars=len(VARIABLES), seed=SEED).to(device)
    run_erm_pretraining(
        erm,
        train_loader,
        val_loader,
        n_epochs=PRETRAIN_EPOCHS_DEMO,
        save_path=erm_path,
        device=device,
    )
    erm.load_state_dict(load_state_dict_flexible(erm_path, device))

    logging.info("Pretraining with PCL (physiological constraints)...")
    pcl = fresh_model(n_vars=len(VARIABLES), seed=SEED + 1).to(device)
    pcl_loss = PhysiologicalConstraintLoss()
    run_pretraining(
        pcl,
        pcl_loss,
        train_loader,
        val_loader,
        n_epochs=PRETRAIN_EPOCHS_DEMO,
        save_path=pcl_path,
        device=device,
    )
    pcl.load_state_dict(load_state_dict_flexible(pcl_path, device))

    probe_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    target = "pH"
    logging.info("Linear probe on frozen representations — target=%s", target)
    mse_erm = train_probe_mse(erm, probe_loader, target, device=device)
    mse_pcl = train_probe_mse(pcl, probe_loader, target, device=device)

    if mse_pcl < mse_erm:
        interp = (
            f"PCL probe MSE ({mse_pcl:.4f}) < ERM ({mse_erm:.4f}): "
            "PCL representations better retain acid-base structure."
        )
    elif mse_pcl > mse_erm:
        interp = (
            f"ERM probe MSE ({mse_erm:.4f}) < PCL ({mse_pcl:.4f}): "
            "No evidence that PCL better encodes acid-base structure in the demo cohort. "
            "This may improve with the full MIMIC-IV dataset where ABG coverage is higher."
        )
    else:
        interp = "ERM and PCL probe MSE are equal — no measurable difference."

    out_metrics = {
        "target_variable": target,
        "mse_erm_frozen_probe": mse_erm,
        "mse_pcl_frozen_probe": mse_pcl,
        "interpretation": interp,
    }
    with open(os.path.join(RESULTS_DIR, "metrics_linear_probe.json"), "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)
    logging.info("Wrote %s", os.path.join(RESULTS_DIR, "metrics_linear_probe.json"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["ERM (frozen)", "PCL (frozen)"], [mse_erm, mse_pcl], color=["#c44e52", "#4c72b0"])
    ax.set_ylabel("Probe MSE (pH)")
    ax.set_title("Mechanistic linear probe: predicting pH from frozen representations")
    plt.tight_layout()
    path_fig = os.path.join(RESULTS_DIR, "linear_probe.png")
    plt.savefig(path_fig, dpi=200)
    plt.close()
    logging.info("Saved %s", path_fig)


if __name__ == "__main__":
    main()