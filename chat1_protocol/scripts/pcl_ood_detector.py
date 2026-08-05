"""Standalone PCL-violation OOD detector script (Contribution #2)."""
import json
import logging
import os
import sys

os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BATCH_SIZE, CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR, SEED
from src.baselines import fresh_model
from src.data_utils import ICUDataset, VARIABLES, attach_labels, load_stays
from src.eval.evaluate_utils import compute_ood_scores
from src.losses.pcl_loss import PhysiologicalConstraintLoss
from src.training.train_utils import load_state_dict_flexible

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    stays = load_stays(str(DATA_DIR)).head(100)
    ds = ICUDataset(str(DATA_DIR), stays)
    attach_labels(ds, stays)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    ckpt = os.path.join(str(CHECKPOINT_DIR), 'pcl_mortality.pt')
    model = fresh_model(n_vars=len(VARIABLES), seed=SEED + 2).to(device)
    model.add_classification_head('mortality')
    if os.path.isfile(ckpt):
        model.load_state_dict(load_state_dict_flexible(ckpt, device), strict=False)
    else:
        logging.warning('Checkpoint %s not found; scores will be from untrained model.', ckpt)

    loss_fn = PhysiologicalConstraintLoss()
    scores, _ = compute_ood_scores(model, loss_fn, loader, device=device)

    with torch.no_grad():
        errs = []
        for b in loader:
            x = b['x'].to(device)
            m = b['mask'].to(device)
            y = b['mortality'].cpu().numpy()
            p = torch.sigmoid(model.classify(model.encode(x, m), 'mortality', obs_mask=m).squeeze(-1)).cpu().numpy()
            errs.extend(np.abs(p - y).tolist())
    errs = np.asarray(errs)

    corr = float(np.corrcoef(scores, errs)[0, 1]) if np.std(scores) > 0 and np.std(errs) > 0 else 0.0
    out = {'pcl_violation_error_correlation': corr, 'n_samples': int(scores.shape[0])}
    with open(os.path.join(str(RESULTS_DIR), 'metrics_pcl_ood_detector.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    plt.figure(figsize=(6, 5))
    plt.scatter(scores, errs, s=18, alpha=0.7, color='#4c72b0')
    plt.xlabel('PCL violation score')
    plt.ylabel('Absolute prediction error')
    plt.title(f'PCL OOD detector (corr={corr:.3f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig = os.path.join(str(RESULTS_DIR), 'pcl_ood_detector.png')
    plt.savefig(fig, dpi=200)
    plt.close()
    logging.info('Saved %s', fig)


if __name__ == '__main__':
    main()
