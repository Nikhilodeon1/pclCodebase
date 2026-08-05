import json
import logging
import os
import sys

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BATCH_SIZE, CHECKPOINT_DIR, DATA_DIR, FINETUNE_EPOCHS_DEMO, PRETRAIN_EPOCHS_DEMO, RESULTS_DIR, SEED
from src.baselines import fresh_model
from src.data_utils import ICUDataset, VARIABLES, attach_labels, load_stays, make_ood_loaders_by_unit
from src.eval.evaluate_utils import run_finetuning
from src.losses.pcl_loss import PhysiologicalConstraintLoss
from src.training.train_utils import load_state_dict_flexible, run_pretraining

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def _pooled_representations(model, loader, device):
    model.eval()
    reps_all = []
    if len(loader) == 0:
        logging.warning("Loader is empty!")
        return np.array([])
        
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            m = batch["mask"].to(device)
            reps = model.encode(x, m)
            # Last-timestep pooling (causal — no look-ahead bias)
            t_obs = m.any(dim=-1)
            last_idx = t_obs.sum(dim=1).long() - 1
            last_idx = torch.clamp(last_idx, min=0)
            batch_idx = torch.arange(reps.size(0), device=reps.device)
            pooled = reps[batch_idx, last_idx]
            reps_all.append(pooled.cpu().numpy())
            
    if not reps_all:
        return np.array([])
    return np.concatenate(reps_all, axis=0)

def _predict_and_error(model, loader, device):
    model.eval()
    probs, errs = [], []
    if len(loader) == 0: return np.array([]), np.array([])
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            m = batch["mask"].to(device)
            y = batch["mortality"].cpu().numpy()
            reps = model.encode(x, m)
            logits = model.classify(reps, "mortality", obs_mask=m).squeeze(-1)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.tolist())
            errs.extend(np.abs(p - y).tolist())
    return np.asarray(probs), np.asarray(errs)

def _score_pcl_violation(model, loader, device):
    loss_fn = PhysiologicalConstraintLoss()
    model.eval()
    scores = []
    if len(loader) == 0: return np.array([])
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            m = batch["mask"].to(device)
            c = batch["c_mask"].to(device)
            preds = model.predict(model.encode(x, m))
            for i in range(x.shape[0]):
                score = loss_fn(preds[i : i + 1], c[i : i + 1])["L_PCL"].item()
                scores.append(float(score))
    return np.asarray(scores)

def _fit_mahalanobis(train_reps):
    if train_reps.size == 0:
        return None, None
    mu = train_reps.mean(axis=0)
    cov = np.cov(train_reps, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-3
    inv = np.linalg.pinv(cov)
    return mu, inv

def _score_mahalanobis(reps, mu, inv):
    if mu is None or reps.size == 0:
        return np.zeros(len(reps))
    diff = reps - mu[None, :]
    return np.einsum("bi,ij,bj->b", diff, inv, diff)

def _score_mc_dropout(model, loader, device, n_passes=12):
    scores = []
    if len(loader) == 0: return np.array([])
    for batch in loader:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        pass_probs = []
        for _ in range(n_passes):
            model.train()  # Force dropout active
            with torch.no_grad():
                reps = model.encode(x, m)
                logits = model.classify(reps, "mortality", obs_mask=m).squeeze(-1)
                pass_probs.append(torch.sigmoid(logits).cpu().numpy())
        arr = np.stack(pass_probs, axis=0)
        scores.extend(arr.std(axis=0).tolist())
    model.eval()
    return np.asarray(scores)

def _score_energy(model, loader, device):
    model.eval()
    scores = []
    if len(loader) == 0: return np.array([])
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            m = batch["mask"].to(device)
            reps = model.encode(x, m)
            logits = model.classify(reps, "mortality", obs_mask=m).squeeze(-1)
            two_class = torch.stack([torch.zeros_like(logits), logits], dim=-1)
            energy = -torch.logsumexp(two_class, dim=-1)
            scores.extend(energy.abs().cpu().numpy().tolist())
    return np.asarray(scores)

def _corr(a, b):
    if a.size == 0 or b.size == 0 or len(a) != len(b):
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stays = load_stays(DATA_DIR).head(500) 
    dataset = ICUDataset(DATA_DIR, stays)
    attach_labels(dataset, stays)
    
    tr, va, ood = make_ood_loaders_by_unit(dataset, stays, batch_size=min(BATCH_SIZE, 32))

    pre_path = os.path.join(CHECKPOINT_DIR, "ooddet_pcl_pre.pt")
    ft_path = os.path.join(CHECKPOINT_DIR, "ooddet_pcl_ft.pt")
    
    m = fresh_model(n_vars=len(VARIABLES), seed=SEED + 31).to(device)
    
    run_pretraining(
        m, PhysiologicalConstraintLoss(), tr, va,
        n_epochs=PRETRAIN_EPOCHS_DEMO, save_path=pre_path, device=device,
    )
    
    # Load and add head
    m.load_state_dict(load_state_dict_flexible(pre_path, device))
    m.add_classification_head("mortality")
    
    # Finetuning
    run_finetuning(m, tr, va, "mortality", n_epochs=FINETUNE_EPOCHS_DEMO, device=device, save_path=ft_path)
    m.load_state_dict(load_state_dict_flexible(ft_path, device), strict=False)

    # OOD Evaluations
    train_reps = _pooled_representations(m, tr, device)
    ood_reps = _pooled_representations(m, ood, device)
    _, pred_err = _predict_and_error(m, ood, device)

    if train_reps.size == 0 or ood_reps.size == 0:
        logging.error("Insufficient data in loaders to calculate OOD metrics.")
        return

    mu, inv = _fit_mahalanobis(train_reps)
    
    scores = {
        "PCL violation": _score_pcl_violation(m, ood, device),
        "Mahalanobis": _score_mahalanobis(ood_reps, mu, inv),
        "MC Dropout": _score_mc_dropout(m, ood, device),
        "Energy": _score_energy(m, ood, device),
    }
    
    corrs = {k: _corr(v, pred_err) for k, v in scores.items()}
    logging.info(f"Correlations: {corrs}")

    # Save results
    out = {"correlation_with_prediction_error": corrs}
    with open(os.path.join(RESULTS_DIR, "metrics_ood_detector_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Plotting
    labels = list(corrs.keys())
    vals = [corrs[k] for k in labels]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Pearson correlation with |prediction error|")
    plt.title("OOD detector comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    path_fig = os.path.join(RESULTS_DIR, "ood_detector_comparison.png")
    plt.savefig(path_fig, dpi=200)
    plt.close()
    logging.info("Saved %s", path_fig)

if __name__ == "__main__":
    main()