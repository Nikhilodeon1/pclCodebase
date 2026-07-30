"""
Jacobian sensitivity analysis (ProjOVERVIEW): ∂(risk)/∂(HR) via torch.autograd.functional.jacobian.
Delegates to src.eval.interpretability for the core computation.
"""
import logging
import os
import sys

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHECKPOINT_DIR, DATA_DIR, SEED
from src.baselines import fresh_model
from src.data_utils import ICUDataset, VARIABLES, attach_labels, load_stays
from src.eval.interpretability import jacobian_risk_wrt_hr, plot_risk_hr_jacobian_heatmap
from src.training.train_utils import load_state_dict_flexible
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    os.makedirs("results", exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stays = load_stays(DATA_DIR).head(100)
    dataset = ICUDataset(DATA_DIR, stays)
    attach_labels(dataset, stays)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    x = batch["x"].to(device)
    mask = batch["mask"].to(device)

    model = fresh_model(n_vars=len(VARIABLES), seed=SEED).to(device)
    model.add_classification_head("mortality")

    pcl_ckpt = os.path.join(CHECKPOINT_DIR, "pcl_pretrained.pt")
    if os.path.isfile(pcl_ckpt):
        model.load_state_dict(load_state_dict_flexible(pcl_ckpt, device), strict=False)
        logging.info("Loaded PCL pretraining checkpoint for Jacobian analysis.")
    else:
        logging.warning(
            "No PCL checkpoint at %s — using random init (run linear_probe or full train first).",
            pcl_ckpt,
        )

    save_heat = "results/jacobian_sensitivity.png"
    plot_risk_hr_jacobian_heatmap(model, x, mask, save_path=save_heat, task_name="mortality")

    jac = jacobian_risk_wrt_hr(model, x, mask, "mortality")
    plt.figure(figsize=(10, 4))
    plt.bar(range(jac.shape[0]), jac.detach().cpu().numpy(), color="steelblue", alpha=0.85)
    plt.xlabel("Time index")
    plt.ylabel(r"$\partial\, \mathrm{risk} / \partial\, \mathrm{HR}$")
    plt.title("Jacobian sensitivity of mortality risk to heart rate")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("results/jacobian_sensitivity_bars.png", dpi=200)
    plt.close()
    logging.info("Saved bar plot to results/jacobian_sensitivity_bars.png")


if __name__ == "__main__":
    main()
