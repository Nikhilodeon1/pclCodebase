"""
IRM failure-mode experiment:
  - IRM-unit: many environments (care units)
  - IRM-hospital: few environments (hospital proxy groups)
  - PCL reference
Evaluates on OOD care-unit split.
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
from src.baselines import IRMFinetuner, attach_environment_labels, fresh_model, run_erm_pretraining
from src.data_utils import ICUDataset, VARIABLES, attach_labels, load_stays, make_ood_loaders_by_unit
from src.eval.evaluate_utils import evaluate_model, run_finetuning
from src.losses.pcl_loss import PhysiologicalConstraintLoss
from src.training.train_utils import load_state_dict_flexible, run_pretraining

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _auroc(m):
    return float(m["auroc"]) if m.get("auroc") is not None else float("nan")


def _attach_hospital_proxy(stays):
    out = stays.copy()

    def to_group(unit_name):
        if not isinstance(unit_name, str):
            return "general"
        lower = unit_name.lower()
        if "cardiac" in lower or "coronary" in lower:
            return "hospital_cardiac"
        if "neuro" in lower:
            return "hospital_neuro"
        return "hospital_general"

    out["hospital_proxy"] = out["first_careunit"].map(to_group)
    return out


def _train_pcl_reference(train_loader, val_loader, ood_loader, device):
    pre_path = os.path.join(CHECKPOINT_DIR, "irmcmp_pcl_pre.pt")
    ft_path = os.path.join(CHECKPOINT_DIR, "irmcmp_pcl_ft.pt")
    m = fresh_model(n_vars=len(VARIABLES), seed=SEED + 11).to(device)
    run_pretraining(
        m,
        PhysiologicalConstraintLoss(),
        train_loader,
        val_loader,
        n_epochs=PRETRAIN_EPOCHS_DEMO,
        save_path=pre_path,
        device=device,
    )
    m.load_state_dict(load_state_dict_flexible(pre_path, device))
    m.add_classification_head("mortality")
    run_finetuning(
        m,
        train_loader,
        val_loader,
        "mortality",
        n_epochs=FINETUNE_EPOCHS_DEMO,
        device=device,
        save_path=ft_path,
    )
    m.load_state_dict(load_state_dict_flexible(ft_path, device), strict=False)
    return evaluate_model(m, ood_loader, "mortality", device=device, split_name="PCL OOD")


def _train_irm(dataset, env_stays, train_loader, val_loader, ood_loader, env_col, label, device):
    pre_path = os.path.join(CHECKPOINT_DIR, f"irmcmp_{label}_pre.pt")
    ft_path = os.path.join(CHECKPOINT_DIR, f"irmcmp_{label}_ft.pt")

    run_erm_pretraining(
        fresh_model(n_vars=len(VARIABLES), seed=SEED + 100).to(device),
        train_loader,
        val_loader,
        n_epochs=PRETRAIN_EPOCHS_DEMO,
        save_path=pre_path,
        device=device,
    )
    m = fresh_model(n_vars=len(VARIABLES), seed=SEED + 100).to(device)
    m.load_state_dict(load_state_dict_flexible(pre_path, device))

    ds = copy.deepcopy(dataset)
    _, env_map = attach_environment_labels(ds, env_stays, env_col=env_col)
    tr_env, va_env, _ = make_ood_loaders_by_unit(ds, env_stays, batch_size=BATCH_SIZE)
    irm = IRMFinetuner(m, "mortality", lambda_irm=1e3, lr=1e-4, device=device)
    irm.run(tr_env, va_env, n_epochs=FINETUNE_EPOCHS_DEMO, save_path=ft_path)

    m.load_state_dict(load_state_dict_flexible(ft_path, device), strict=False)
    metrics = evaluate_model(m, ood_loader, "mortality", device=device, split_name=f"{label} OOD")
    return metrics, len(env_map)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stays = load_stays(DATA_DIR).head(100)
    dataset = ICUDataset(DATA_DIR, stays)
    attach_labels(dataset, stays)
    train_loader, val_loader, ood_loader = make_ood_loaders_by_unit(dataset, stays, batch_size=BATCH_SIZE)

    pcl_ood = _train_pcl_reference(train_loader, val_loader, ood_loader, device)
    irm_unit_ood, n_env_unit = _train_irm(
        dataset=dataset,
        env_stays=stays,
        train_loader=train_loader,
        val_loader=val_loader,
        ood_loader=ood_loader,
        env_col="first_careunit",
        label="irm_unit",
        device=device,
    )

    stays_h = _attach_hospital_proxy(stays)
    irm_hosp_ood, n_env_hosp = _train_irm(
        dataset=dataset,
        env_stays=stays_h,
        train_loader=train_loader,
        val_loader=val_loader,
        ood_loader=ood_loader,
        env_col="hospital_proxy",
        label="irm_hospital",
        device=device,
    )

    out = {
        "PCL": {"ood_auroc": _auroc(pcl_ood)},
        "IRM-unit": {"ood_auroc": _auroc(irm_unit_ood), "n_env": n_env_unit},
        "IRM-hospital": {"ood_auroc": _auroc(irm_hosp_ood), "n_env": n_env_hosp},
    }
    with open(os.path.join(RESULTS_DIR, "metrics_irm_failure_mode.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    labels = ["IRM-hospital", "IRM-unit", "PCL"]
    vals = [out[k]["ood_auroc"] for k in labels]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, vals, color=["#c44e52", "#dd8452", "#4c72b0"])
    plt.ylabel("OOD AUROC (mortality)")
    plt.title("IRM failure mode: few-env vs many-env vs PCL")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "irm_failure_mode.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    logging.info("Saved %s", fig_path)


if __name__ == "__main__":
    main()
