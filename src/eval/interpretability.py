"""
Interpretability tools for PCL:
  1. Jacobian sensitivity: ∂(risk)/∂(HR_t) for each timestep.
  2. t-SNE/UMAP: side-by-side ERM vs PCL embeddings colored by site_id.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.autograd.functional as AF
import numpy as np

from src.data_utils import VARIABLES


def mortality_risk_scalar(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    task_name: str = "mortality",
) -> torch.Tensor:
    """Sigmoid risk for one batch item (0-dim tensor)."""
    reps = model.encode(x, mask)
    logits = model.classify(reps, task_name, obs_mask=mask)
    return torch.sigmoid(logits.reshape(-1))[0]


def jacobian_risk_wrt_hr(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    task_name: str = "mortality",
) -> torch.Tensor:
    """
    Compute ∂(risk)/∂(HR_t) for each timestep t (batch size must be 1).

    Returns tensor of shape (T,) with derivatives w.r.t. normalized HR channel.
    """
    if x.shape[0] != 1:
        raise ValueError("jacobian_risk_wrt_hr expects batch size 1")

    for p in model.parameters():
        p.requires_grad_(False)

    hr_idx = VARIABLES.index("HR")
    device = x.device

    def risk_from_hr(hr_t: torch.Tensor) -> torch.Tensor:
        x_in = x.clone()
        x_in[0, :, hr_idx] = hr_t
        return mortality_risk_scalar(model, x_in, mask, task_name)

    hr = x[0, :, hr_idx].detach().clone().requires_grad_(True)
    jac = AF.jacobian(risk_from_hr, hr, create_graph=False)
    return jac.reshape(-1).detach()


def jacobian_risk_wrt_hr_batch_mean(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    task_name: str = "mortality",
    max_batch: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Average |∂risk/∂HR| over up to `max_batch` examples for a heatmap-friendly vector (V,).
    For HR we return shape (T,) mean over batch and time aggregation optional.
    """
    model.eval()
    B = min(x.shape[0], max_batch)
    grads = []
    for b in range(B):
        xb = x[b : b + 1]
        mb = mask[b : b + 1]
        g = jacobian_risk_wrt_hr(model, xb, mb, task_name)
        grads.append(g.abs().mean())
    stacked = torch.stack(grads)
    return stacked.mean(), stacked.std(unbiased=False)


def plot_risk_hr_jacobian_heatmap(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    save_path: str,
    task_name: str = "mortality",
) -> None:
    """Save a 1×T heatmap of ∂risk/∂HR (single patient)."""
    import matplotlib.pyplot as plt

    jac = jacobian_risk_wrt_hr(model, x, mask, task_name).cpu().numpy()
    T = jac.shape[0]

    plt.figure(figsize=(max(8, T // 6), 3))
    plt.imshow(jac.reshape(1, -1), aspect="auto", cmap="magma")
    plt.colorbar(label=r"$\partial\, \mathrm{risk} / \partial\, \mathrm{HR}$")
    plt.xlabel("Hour (index)")
    plt.ylabel("")
    plt.yticks([])
    plt.title("Jacobian: risk vs. heart rate (normalized scale)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logging.info("Saved Jacobian heatmap to %s", save_path)


# ── t-SNE VISUALIZATION ───────────────────────────────────────────────────────

SITE_NAMES = {0: "PhysioNet A", 1: "PhysioNet B", 2: "MIMIC-IV", 3: "eICU"}
SITE_COLORS = {0: "#4c72b0", 1: "#c44e52", 2: "#55a868", 3: "#8172b3"}


def _extract_pooled_reps(model, loader, device):
    """Returns (N, d_model) mean-pooled representations and (N,) site_id array."""
    model.to(device).eval()
    reps_list, site_list = [], []
    with torch.no_grad():
        for batch in loader:
            xb = batch["x"].to(device)
            mb = batch["mask"].to(device)
            r = model.encode(xb, mb)
            # Mean-pool over time for 2-D embedding
            pooled = r.mean(dim=1).cpu().numpy()
            reps_list.append(pooled)
            site_list.append(batch["site_id"].numpy())
    return np.concatenate(reps_list), np.concatenate(site_list)


def generate_tsne_visualization(
    erm_model,
    pcl_model,
    loader,
    save_path: str,
    device: str = "cpu",
    seed: int = 42,
    perplexity: int = 30,
) -> None:
    """
    Side-by-side t-SNE of ERM vs PCL representations, colored by site_id.

    If PCL suppresses hospital identity, points from different sites will mix
    in the PCL panel while clustering by site in the ERM panel.

    Args:
        erm_model:  trained ERM PCLModel
        pcl_model:  trained PCL PCLModel
        loader:     DataLoader over the full dataset (all sites)
        save_path:  where to save the PNG
        device:     compute device
        seed:       reproducibility seed for t-SNE
        perplexity: t-SNE perplexity (reduced automatically for small datasets)
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        logging.warning("t-SNE visualization requires sklearn and matplotlib: %s", e)
        return

    logging.info("Extracting ERM representations for t-SNE...")
    erm_reps, site_ids = _extract_pooled_reps(erm_model, loader, device)
    logging.info("Extracting PCL representations for t-SNE...")
    pcl_reps, _ = _extract_pooled_reps(pcl_model, loader, device)

    n = len(erm_reps)
    perp = min(perplexity, max(5, n // 5))

    logging.info("Running t-SNE on %d samples (perplexity=%d)...", n, perp)
    erm_2d = TSNE(n_components=2, perplexity=perp, random_state=seed).fit_transform(erm_reps)
    pcl_2d = TSNE(n_components=2, perplexity=perp, random_state=seed).fit_transform(pcl_reps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for sid, name in SITE_NAMES.items():
        mask = site_ids == sid
        if mask.sum() == 0:
            continue
        axes[0].scatter(erm_2d[mask, 0], erm_2d[mask, 1],
                        c=SITE_COLORS[sid], label=name, alpha=0.6, s=12)
        axes[1].scatter(pcl_2d[mask, 0], pcl_2d[mask, 1],
                        c=SITE_COLORS[sid], label=name, alpha=0.6, s=12)

    for ax, title in zip(axes, ["ERM Representations", "PCL Representations"]):
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=8, markerscale=2)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("t-SNE: Site Clustering in Latent Space", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logging.info("Saved t-SNE visualization to %s", save_path)
