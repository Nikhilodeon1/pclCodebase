"""
Feature-engineering demolition ablation:
  1) ERM
  2) ERM + engineered MAP/PP/SI features
  3) PCL
  4) PCL + engineered MAP/PP/SI features

Evaluates on both in-distribution validation and OOD split by care unit.
"""
import copy
import json
import logging
import os
import sys

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BATCH_SIZE, CHECKPOINT_DIR, DATA_DIR, FINETUNE_EPOCHS_DEMO, PRETRAIN_EPOCHS_DEMO, RESULTS_DIR, SEED
from src.baselines import add_engineered_features, fresh_model, run_erm_pretraining
from src.data_utils import ICUDataset, VARIABLES, attach_labels, load_stays, make_ood_loaders_by_unit
from src.eval.evaluate_utils import evaluate_model, run_finetuning
from src.losses.pcl_loss import PhysiologicalConstraintLoss
from src.training.train_utils import load_state_dict_flexible, run_pretraining

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _train_eval_variant(name, train_loader, val_loader, ood_loader, n_vars, use_pcl, device):
    pre_path = os.path.join(CHECKPOINT_DIR, f"fe_ablation_{name}_pre.pt")
    ft_path = os.path.join(CHECKPOINT_DIR, f"fe_ablation_{name}_ft.pt")
    model = fresh_model(n_vars=n_vars, seed=SEED + abs(hash(name)) % 1000).to(device)

    if use_pcl:
        run_pretraining(
            model,
            PhysiologicalConstraintLoss(),
            train_loader,
            val_loader,
            n_epochs=PRETRAIN_EPOCHS_DEMO,
            save_path=pre_path,
            device=device,
        )
    else:
        run_erm_pretraining(
            model,
            train_loader,
            val_loader,
            n_epochs=PRETRAIN_EPOCHS_DEMO,
            save_path=pre_path,
            device=device,
        )

    model.load_state_dict(load_state_dict_flexible(pre_path, device))
    model.add_classification_head("mortality")
    run_finetuning(
        model,
        train_loader,
        val_loader,
        "mortality",
        n_epochs=FINETUNE_EPOCHS_DEMO,
        device=device,
        save_path=ft_path,
    )
    model.load_state_dict(load_state_dict_flexible(ft_path, device), strict=False)
    id_metrics = evaluate_model(model, val_loader, "mortality", device=device, split_name=f"{name} ID")
    ood_metrics = evaluate_model(model, ood_loader, "mortality", device=device, split_name=f"{name} OOD")
    return id_metrics, ood_metrics


def _auroc(m):
    return float(m["auroc"]) if m.get("auroc") is not None else float("nan")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stays = load_stays(DATA_DIR).head(100)
    dataset_base = ICUDataset(DATA_DIR, stays)
    attach_labels(dataset_base, stays)

    dataset_fe = copy.deepcopy(dataset_base)
    add_engineered_features(dataset_fe)

    tr_b, va_b, ood_b = make_ood_loaders_by_unit(dataset_base, stays, batch_size=BATCH_SIZE)
    tr_f, va_f, ood_f = make_ood_loaders_by_unit(dataset_fe, stays, batch_size=BATCH_SIZE)

    results = {}
    id_m, ood_m = _train_eval_variant("erm", tr_b, va_b, ood_b, len(VARIABLES), use_pcl=False, device=device)
    results["ERM"] = {"id_auroc": _auroc(id_m), "ood_auroc": _auroc(ood_m)}

    id_m, ood_m = _train_eval_variant("erm_fe", tr_f, va_f, ood_f, len(VARIABLES) + 3, use_pcl=False, device=device)
    results["ERM+FE"] = {"id_auroc": _auroc(id_m), "ood_auroc": _auroc(ood_m)}

    id_m, ood_m = _train_eval_variant("pcl", tr_b, va_b, ood_b, len(VARIABLES), use_pcl=True, device=device)
    results["PCL"] = {"id_auroc": _auroc(id_m), "ood_auroc": _auroc(ood_m)}

    id_m, ood_m = _train_eval_variant("pcl_fe", tr_f, va_f, ood_f, len(VARIABLES) + 3, use_pcl=True, device=device)
    results["PCL+FE"] = {"id_auroc": _auroc(id_m), "ood_auroc": _auroc(ood_m)}

    with open(os.path.join(RESULTS_DIR, "metrics_feature_engineering_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    labels = list(results.keys())
    id_vals = [results[k]["id_auroc"] for k in labels]
    ood_vals = [results[k]["ood_auroc"] for k in labels]
    x = list(range(len(labels)))

    plt.figure(figsize=(9, 5))
    w = 0.36
    plt.bar([i - w / 2 for i in x], id_vals, width=w, label="ID AUROC", color="#9ecae1")
    plt.bar([i + w / 2 for i in x], ood_vals, width=w, label="OOD AUROC", color="#4c72b0")
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("AUROC")
    plt.title("Feature Engineering Demolition Ablation")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "feature_engineering_ablation.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    logging.info("Saved %s", fig_path)


if __name__ == "__main__":
    main()
