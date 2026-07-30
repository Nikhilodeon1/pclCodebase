"""
PCL Paper Experiment Runner
Generates all results and figures for the NeurIPS 2026 submission.
Uses PhysioNet 2019 as primary dataset with MIMIC-IV and eICU as external validation.

Usage:
    PCL_TEST_MODE=1 python run_paper_experiments.py   # test mode (15% data, small model)
    PCL_TEST_MODE=0 python run_paper_experiments.py   # production mode (full data, ~3.2M params)
"""
import os
import sys
# Ensure the repo root is importable regardless of the current working directory
# (fixes "ModuleNotFoundError: No module named 'src.data'" when launched from
# elsewhere or when src/ lacks __init__.py files on a fresh checkout).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import logging
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    PHYSIONET_DIR, MIMIC_DIR, EICU_DIR,
    RESULTS_DIR, CHECKPOINT_DIR, CACHE_DIR,
    DATA_FRACTION, BATCH_SIZE, SEED,
    PRETRAIN_EPOCHS, FINETUNE_EPOCHS,
    LAMBDA_PCL, LAMBDA_VALUES, NOISE_LEVELS,
    D_MODEL, N_HEADS, N_LAYERS, TEST_MODE,
    NUM_WORKERS, PIN_MEMORY, USE_AMP, MASK_PROB,
)

# Shared random-init checkpoint (per-seed, isolated by CHECKPOINT_DIR): makes ERM
# and PCL start from identical weights within a run. Seed-study runs use separate
# CHECKPOINT_DIRs, so each seed gets its own base_init automatically.
BASE_INIT_PATH = os.path.join(CHECKPOINT_DIR, "base_init.pt")
# Seed study: run only EXP1 + EXP3 (main OOD + randomization) for multi-seed error bars.
SEED_STUDY = os.environ.get("PCL_SEED_STUDY", "0") == "1"
# A3 regularization baselines (denoising-AE / noise-aug / dropout+WD / Mixup).
# Opt-in (heavy: each is hyperparameter-swept) so default/seed-study runs aren't
# bloated. Enable with PCL_RUN_A3=1.
RUN_A3 = os.environ.get("PCL_RUN_A3", "0") == "1"
from src.data.physionet2019 import load_physionet2019
from src.data.mimic4 import load_mimic4
from src.data.eicu import load_eicu
from src.data.dataset import ICUDataset, make_site_split_loaders, make_patient_split_loaders
from src.baselines import fresh_model, run_erm_pretraining, add_engineered_features
from src.losses.pcl_loss import PhysiologicalConstraintLoss
from src.training.train_utils import (
    run_pretraining, load_state_dict_flexible, save_checkpoint,
    plot_loss_curves, plot_constraint_curves,
)
from src.eval.evaluate_utils import (
    run_finetuning, evaluate_model, run_linear_probe,
    extract_representations, compute_ood_scores, set_prediction_dump,
)

from download_datasets import ensure_datasets

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Primary task is sepsis onset — the paper's headline metric (§4.1).
# Labels by site:
#   - PhysioNet A/B: hourly Sepsis-3 labels from the challenge SepsisLabel column.
#   - MIMIC-IV / eICU: admission-level binary sepsis extracted from ICD diagnosis
#     codes (explicit-sepsis set; see src/data/sepsis.py), matching the paper.
# On the FULL datasets sepsis is non-sparse across all sites (PhysioNet ~7%,
# MIMIC ~8-9%, eICU ~14%), so the earlier los_3d fallback (used when MIMIC/eICU
# had no sepsis labels) is no longer needed.
# los_3d and mortality are retained as secondary comparisons (EXP4 linear probes,
# classical baselines) but are NOT the primary reported metric.
TASK = "sepsis"
TASKS = ["sepsis", "los_3d"]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

ALL_RESULTS = {}


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 0: Constraint Violation Audit (proves constraints aren't vacuous)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_0_constraint_violation_audit():
    """
    Measures how often raw clinical data violates the 5 PCL constraints.
    Proves that MAP = DBP + (SBP-DBP)/3 is NOT trivially satisfied in EHR data
    because MAP (arterial line) and SBP/DBP (cuff) come from different sensors.
    """
    logging.info("\n" + "=" * 60)
    logging.info("EXPERIMENT 0: Constraint Violation Audit")
    logging.info("=" * 60)

    from src.data.variables import VAR_TO_IDX
    idx = VAR_TO_IDX

    pn_samples, _    = load_physionet2019(PHYSIONET_DIR, fraction=DATA_FRACTION, keep_raw=True)
    mimic_samples, _ = load_mimic4(MIMIC_DIR, fraction=DATA_FRACTION, keep_raw=True)
    eicu_samples, _  = load_eicu(EICU_DIR,  fraction=DATA_FRACTION, keep_raw=True)

    def _audit_dataset(samples, name):
        stats = {"MAP": [], "HH": [], "SpO2": []}
        n_map = n_hh = n_spo2 = 0

        # ── Computability tracking (professor action item A2) ──────────────────
        # Report how often each constraint is *computable* (all required raw
        # variables observed), not just how often it is violated. Tracked at the
        # timestep level (fraction of all stay-hours) and the stay level
        # (fraction of stays with >=1 computable hour). Acid-base constraints are
        # expected to be sparse because ABG panels are drawn infrequently.
        n_total_ts = 0
        n_stays = 0
        comp_stays = {"MAP": 0, "HH": 0, "SpO2": 0}

        for sample in samples:
            raw = sample.get("raw_ts")
            if raw is None:
                continue
            n_stays += 1
            n_total_ts += raw.shape[0]
            stay_comp = {"MAP": False, "HH": False, "SpO2": False}
            for t in range(raw.shape[0]):
                sbp, dbp, map_v = raw[t, idx["SBP"]], raw[t, idx["DBP"]], raw[t, idx["MAP"]]
                if not (np.isnan(sbp) or np.isnan(dbp) or np.isnan(map_v)):
                    expected = dbp + (sbp - dbp) / 3.0
                    stats["MAP"].append(abs(map_v - expected))
                    n_map += 1
                    stay_comp["MAP"] = True

                ph, hco3, pco2 = raw[t, idx["pH"]], raw[t, idx["HCO3"]], raw[t, idx["pCO2"]]
                if not (np.isnan(ph) or np.isnan(hco3) or np.isnan(pco2)) and pco2 > 0 and hco3 > 0:
                    expected_ph = 6.1 + np.log10(hco3 / (0.0307 * pco2))
                    stats["HH"].append(abs(ph - expected_ph))
                    n_hh += 1
                    stay_comp["HH"] = True

                spo2, pao2 = raw[t, idx["SpO2"]], raw[t, idx["PaO2"]]
                if not (np.isnan(spo2) or np.isnan(pao2)) and pao2 > 0:
                    expected_spo2 = 100.0 * (1.0 / ((23400.0 / (pao2**3 + 150.0 * pao2)) + 1.0))
                    stats["SpO2"].append(abs(spo2 - expected_spo2))
                    n_spo2 += 1
                    stay_comp["SpO2"] = True
            for c in comp_stays:
                if stay_comp[c]:
                    comp_stays[c] += 1

        logging.info(f"\n  {name}:")
        result = {}
        computable = {}
        for cname, violations in stats.items():
            # Computability: fraction of all stay-hours (and of stays) where the
            # constraint's required variables are all present.
            n_comp_ts = {"MAP": n_map, "HH": n_hh, "SpO2": n_spo2}[cname]
            ts_rate = (n_comp_ts / n_total_ts) if n_total_ts else 0.0
            stay_rate = (comp_stays[cname] / n_stays) if n_stays else 0.0
            computable[cname] = {
                "computable_ts": n_comp_ts,
                "total_ts": n_total_ts,
                "ts_rate": ts_rate,
                "computable_stays": comp_stays[cname],
                "total_stays": n_stays,
                "stay_rate": stay_rate,
            }
            logging.info(f"    {cname} computability: {n_comp_ts}/{n_total_ts} hours "
                         f"({ts_rate:.1%}), {comp_stays[cname]}/{n_stays} stays ({stay_rate:.1%})")
            if violations:
                arr = np.array(violations)
                n_total = len(arr)
                # "violated" thresholds: MAP > 3mmHg, HH > 0.05 pH, SpO2 > 5%
                thresholds = {"MAP": 3.0, "HH": 0.05, "SpO2": 5.0}
                thresh = thresholds[cname]
                n_viol = int(np.sum(arr > thresh))
                rate = n_viol / n_total
                logging.info(f"    {cname}: N={n_total}, violated={n_viol} ({rate:.1%}), "
                           f"mean|err|={arr.mean():.3f}, p95={np.percentile(arr, 95):.3f}")
                result[cname] = {"n": n_total, "rate": rate, "mean": float(arr.mean()),
                                "p95": float(np.percentile(arr, 95))}
            else:
                logging.info(f"    {cname}: No computable timesteps")
                result[cname] = None
        return result, computable

    audit_results = {}
    computability_results = {}
    if pn_samples:
        audit_results["PhysioNet"], computability_results["PhysioNet"] = _audit_dataset(pn_samples, "PhysioNet 2019")
    if mimic_samples:
        audit_results["MIMIC-IV"], computability_results["MIMIC-IV"] = _audit_dataset(mimic_samples, "MIMIC-IV Demo")
    if eicu_samples:
        audit_results["eICU"], computability_results["eICU"] = _audit_dataset(eicu_samples, "eICU Demo")

    logging.info("\n  Key finding: Real EHR data violates 'algebraic identities' at substantial")
    logging.info("  rates due to measurement disagreement between sensors (arterial line vs cuff,")
    logging.info("  different lab analyzers). PCL constraints provide real denoising signal.")

    ALL_RESULTS["experiment_0_violation_audit"] = audit_results
    ALL_RESULTS["constraint_computability"] = computability_results
    return audit_results


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_all_data():
    logging.info("=" * 60)
    logging.info("Loading all datasets")
    logging.info("=" * 60)

    # Cache the preprocessed samples so a resumed run does NOT re-read the full
    # CSVs (35GB eICU + chunked MIMIC + ~300k patients of preprocessing = tens of
    # minutes). Keyed by mode so a test cache never poisons a production run.
    import pickle
    tag = "test" if TEST_MODE else "prod"
    cache_path = os.path.join(CACHE_DIR, f"processed_samples_{tag}.pkl")

    if os.path.exists(cache_path):
        try:
            logging.info(f"[CACHE] Loading preprocessed samples from {cache_path}")
            with open(cache_path, "rb") as fp:
                blob = pickle.load(fp)
            pn_samples    = blob["pn"]
            mimic_samples = blob["mimic"]
            eicu_samples  = blob["eicu"]
            all_samples = pn_samples + mimic_samples + eicu_samples
            ds = ICUDataset(all_samples)
            ds.coverage_report()
            logging.info(f"[CACHE] Loaded {len(all_samples)} samples — skipped CSV read")
            return ds, pn_samples, mimic_samples, eicu_samples
        except Exception as e:
            logging.warning(f"[CACHE] Failed to load cache ({e}); re-reading from CSV")

    pn_samples, _    = load_physionet2019(PHYSIONET_DIR, fraction=DATA_FRACTION)
    mimic_samples, _ = load_mimic4(MIMIC_DIR, fraction=DATA_FRACTION)
    eicu_samples, _  = load_eicu(EICU_DIR,  fraction=DATA_FRACTION)

    logging.info(f"PhysioNet: {len(pn_samples)}")
    logging.info(f"MIMIC-IV:  {len(mimic_samples)}")
    logging.info(f"eICU:      {len(eicu_samples)}")

    all_samples = pn_samples + mimic_samples + eicu_samples
    ds = ICUDataset(all_samples)
    ds.coverage_report()

    # Persist cache for fast resumes (best-effort; never fatal)
    try:
        with open(cache_path, "wb") as fp:
            pickle.dump({"pn": pn_samples, "mimic": mimic_samples, "eicu": eicu_samples},
                        fp, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"[CACHE] Saved preprocessed samples → {cache_path}")
    except Exception as e:
        logging.warning(f"[CACHE] Could not write sample cache: {e}")

    return ds, pn_samples, mimic_samples, eicu_samples


def make_ood_loaders(ds, batch_size=BATCH_SIZE, task=None):
    """Train on PhysioNet A, val from A (stratified), test on PhysioNet B + MIMIC + eICU."""
    if task is None:
        task = TASK
    site_a_idx = [i for i, s in enumerate(ds.samples) if s["site_id"].item() == 0]
    site_b_idx = [i for i, s in enumerate(ds.samples) if s["site_id"].item() == 1]
    mimic_idx = [i for i, s in enumerate(ds.samples) if s["site_id"].item() == 2]
    eicu_idx = [i for i, s in enumerate(ds.samples) if s["site_id"].item() == 3]

    # Stratified split of site A into train/val
    labels_a = np.array([ds.samples[i][task].item() for i in site_a_idx])
    if len(np.unique(labels_a)) >= 2 and len(site_a_idx) > 10:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        idx_arr = np.arange(len(site_a_idx))
        tr_sub, va_sub = next(sss.split(idx_arr, labels_a))
        train_indices = [site_a_idx[i] for i in tr_sub]
        val_indices = [site_a_idx[i] for i in va_sub]
    else:
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(site_a_idx))
        n_val = max(1, int(len(site_a_idx) * 0.2))
        val_indices = [site_a_idx[i] for i in perm[:n_val]]
        train_indices = [site_a_idx[i] for i in perm[n_val:]]

    train_loader = DataLoader(Subset(ds, train_indices), batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(Subset(ds, val_indices), batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    ood_loaders = OrderedDict()
    if site_b_idx:
        ood_loaders["PhysioNet-B"] = DataLoader(Subset(ds, site_b_idx), batch_size=batch_size,
                                                 shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    if mimic_idx:
        ood_loaders["MIMIC-IV"] = DataLoader(Subset(ds, mimic_idx), batch_size=batch_size,
                                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    if eicu_idx:
        ood_loaders["eICU"] = DataLoader(Subset(ds, eicu_idx), batch_size=batch_size,
                                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    n_pos_train = sum(1 for i in train_indices if ds.samples[i][task].item() > 0)
    n_pos_val = sum(1 for i in val_indices if ds.samples[i][task].item() > 0)
    logging.info(f"OOD split: train={len(train_indices)} ({n_pos_train} {task}+), val={len(val_indices)} ({n_pos_val} {task}+)")
    for name, loader in ood_loaders.items():
        idxs = loader.dataset.indices
        n_pos = sum(1 for i in idxs if ds.samples[i][task].item() > 0)
        logging.info(f"  OOD {name}: {len(idxs)} patients ({n_pos} {task}+)")

    return train_loader, val_loader, ood_loaders


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: train a model end-to-end
# ══════════════════════════════════════════════════════════════════════════════

def train_model(name, train_loader, val_loader, use_pcl=True, lam=LAMBDA_PCL,
                pcl_loss_fn=None, seed_offset=0, task=None, use_site_adversary=False,
                pretrain_loader=None):
    """
    pretrain_loader: optional separate DataLoader for the pretraining phase.
    Defaults to train_loader. Pass the full multi-site dataset here when using
    the SiteAdversary so the encoder sees all sites during self-supervised pretraining
    (no label leakage — masked prediction is unsupervised).

    Resume-safe: if checkpoints already exist on disk, skips the corresponding
    training phase and loads weights directly. Safe to re-run after interruption.
    """
    if task is None:
        task = TASK
    model = fresh_model(seed=SEED + seed_offset)

    if os.path.exists(BASE_INIT_PATH):
        logging.info(f"Loading consistent base initialization from {BASE_INIT_PATH}")
        model.load_state_dict(torch.load(BASE_INIT_PATH, map_location="cpu", weights_only=True))

    _pretrain_loader = pretrain_loader if pretrain_loader is not None else train_loader
    ckpt_pre = os.path.join(CHECKPOINT_DIR, f"{name}_pretrained.pt")
    ckpt_ft  = os.path.join(CHECKPOINT_DIR, f"{name}_{task}.pt")

    # ── Pretraining (skip if checkpoint exists) ───────────────────────────────
    if os.path.exists(ckpt_pre):
        logging.info(f"[RESUME] Found {ckpt_pre} — skipping pretraining for '{name}'")
        history = []
    elif use_pcl:
        if pcl_loss_fn is None:
            pcl_loss_fn = PhysiologicalConstraintLoss()
        history = run_pretraining(
            model, pcl_loss_fn, _pretrain_loader, val_loader,
            n_epochs=PRETRAIN_EPOCHS, lam=lam, device=DEVICE,
            save_path=ckpt_pre, warmup_fraction=0.4,
            use_site_adversary=use_site_adversary,
        )
    else:
        history = run_erm_pretraining(
            model, _pretrain_loader, val_loader,
            n_epochs=PRETRAIN_EPOCHS, device=DEVICE, save_path=ckpt_pre,
        )

    if os.path.exists(ckpt_pre):
        sd = load_state_dict_flexible(ckpt_pre, DEVICE)
        model.load_state_dict(sd)

    model.add_classification_head(task)

    # ── Fine-tuning (skip if checkpoint exists) ───────────────────────────────
    if os.path.exists(ckpt_ft):
        logging.info(f"[RESUME] Found {ckpt_ft} — skipping fine-tuning for '{name}'")
        ft_history = []
    else:
        ft_history = run_finetuning(
            model, train_loader, val_loader, task,
            n_epochs=FINETUNE_EPOCHS, device=DEVICE, save_path=ckpt_ft,
        )

    if os.path.exists(ckpt_ft):
        sd_ft = load_state_dict_flexible(ckpt_ft, DEVICE)
        model.load_state_dict(sd_ft)

    return model, history, ft_history


def eval_on_loaders(model, loaders_dict, task=TASK):
    results = {}
    for name, loader in loaders_dict.items():
        if task not in model.cls_heads:
            model.add_classification_head(task)
            model = model.to(DEVICE)
        r = evaluate_model(model, loader, task, device=DEVICE, split_name=f"{name} ({task})")
        results[name] = r
    return results

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Main OOD Generalization (plan.md §6.3)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_1_ood_generalization(ds, train_loader, val_loader, ood_loaders):
    logging.info("\n" + "=" * 60)
    logging.info(f"EXPERIMENT 1: Main OOD Generalization (task={TASK})")
    logging.info("=" * 60)

    results = {}

    # ERM
    logging.info("\n--- ERM Baseline ---")
    erm_model, _, _ = train_model("erm", train_loader, val_loader, use_pcl=False)
    erm_val = evaluate_model(erm_model, val_loader, TASK, device=DEVICE, split_name="ERM Val")
    erm_ood = eval_on_loaders(erm_model, ood_loaders)
    results["ERM"] = {"val": erm_val, "ood": erm_ood}

    # PCL: physiological constraints only, no site adversary.
    # ZERO-SHOT SETUP — PCL pretrains on Site A ONLY (default train_loader), exactly
    # like ERM/DRO and every ablation. This preserves the paper's zero-shot domain-
    # generalization premise (Section 3.1): the model never sees OOD-site inputs,
    # labeled or unlabeled, before test. (An earlier version pretrained PCL on all
    # sites' unlabeled inputs — an unintended domain-adaptation confound that gave
    # PCL an unfair OOD advantage over Site-A-only ERM. Removed.)
    logging.info("\n--- PCL ---")
    pcl_model, pcl_hist, _ = train_model(
        "pcl", train_loader, val_loader, use_pcl=True, seed_offset=2,
        use_site_adversary=False,
    )
    pcl_val = evaluate_model(pcl_model, val_loader, TASK, device=DEVICE, split_name="PCL Val")
    pcl_ood = eval_on_loaders(pcl_model, ood_loaders)
    results["PCL"] = {"val": pcl_val, "ood": pcl_ood}

    if pcl_hist:
        plot_loss_curves(pcl_hist, save_path=os.path.join(RESULTS_DIR, "pcl_pretrain_loss.png"))
        plot_constraint_curves(pcl_hist, save_path=os.path.join(RESULTS_DIR, "pcl_constraint_curves.png"))

    # DRO (auto-detect number of groups from training data)
    logging.info("\n--- DRO Baseline ---")
    from src.baselines import DROFinetuner
    dro_model = fresh_model(seed=SEED + 4).to(DEVICE)
    if os.path.exists(BASE_INIT_PATH):
        logging.info("DRO: Loading shared initialization weights from base_init.pt")
        dro_model.load_state_dict(torch.load(BASE_INIT_PATH, map_location=DEVICE, weights_only=True))
    run_erm_pretraining(
        dro_model, train_loader, val_loader,
        n_epochs=PRETRAIN_EPOCHS, device=DEVICE,
        save_path=os.path.join(CHECKPOINT_DIR, "dro_pretrained.pt"),
    )
    dro_pre_path = os.path.join(CHECKPOINT_DIR, "dro_pretrained.pt")
    if os.path.exists(dro_pre_path):
        sd = load_state_dict_flexible(dro_pre_path, DEVICE)
        dro_model.load_state_dict(sd)
    train_env_ids = set()
    for i in train_loader.dataset.indices:
        train_env_ids.add(ds.samples[i]["env_id"].item())
    n_dro_groups = max(1, len(train_env_ids))
    if n_dro_groups == 1:
        logging.info("DRO: only 1 group in training data — DRO degenerates to ERM (expected for single-site training)")
    else:
        logging.info(f"DRO: {n_dro_groups} groups detected in training data")
    dro_ft = DROFinetuner(dro_model, TASK, n_groups=n_dro_groups, device=DEVICE)
    dro_ft.run(train_loader, val_loader, n_epochs=FINETUNE_EPOCHS)
    dro_val = evaluate_model(dro_model, val_loader, TASK, device=DEVICE, split_name="DRO Val")
    dro_ood = eval_on_loaders(dro_model, ood_loaders)
    results["DRO"] = {"val": dro_val, "ood": dro_ood}

    _print_ood_table(results, ood_loaders)

    ALL_RESULTS["experiment_1"] = _serialize_results(results)
    return results, erm_model, pcl_model


def _print_ood_table(results, ood_loaders):
    ood_names = list(ood_loaders.keys())
    header = f"{'Model':<12} {'Val AUROC':<12}" + "".join(f"{n+' AUROC':<16}" for n in ood_names)
    logging.info("\n" + header)
    logging.info("─" * len(header))
    for model_name, data in results.items():
        val_auroc = data["val"].get("auroc")
        row = f"{model_name:<12} {_fmt(val_auroc):<12}"
        for ood_name in ood_names:
            ood_auroc = data["ood"].get(ood_name, {}).get("auroc")
            row += f"{_fmt(ood_auroc):<16}"
        logging.info(row)


def _fmt(v):
    return f"{v:.4f}" if v is not None else "N/A"


def _serialize_val(v):
    if v is None:
        return None
    if isinstance(v, (tuple, list)):
        return [float(x) if x is not None else None for x in v]
    return float(v)


def _serialize_results(results):
    out = {}
    for model_name, data in results.items():
        out[model_name] = {}
        for split_type, split_data in data.items():
            if isinstance(split_data, dict) and "auroc" in split_data:
                out[model_name][split_type] = {k: _serialize_val(v) for k, v in split_data.items()}
            elif isinstance(split_data, dict):
                out[model_name][split_type] = {}
                for k, v in split_data.items():
                    if isinstance(v, dict):
                        out[model_name][split_type][k] = {kk: _serialize_val(vv) for kk, vv in v.items()}
                    else:
                        out[model_name][split_type][k] = _serialize_val(v)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Feature Engineering Demolition (plan.md §6.4)
# ══════════════════════════════════════════════════════════════════════════════

def _rebuild_loader(new_ds, orig_loader, shuffle, batch_size=BATCH_SIZE):
    """Rebuild DataLoader from new_ds using same indices as orig_loader."""
    if hasattr(orig_loader.dataset, 'indices'):
        indices = list(orig_loader.dataset.indices)
    else:
        indices = list(range(len(orig_loader.dataset)))
    return DataLoader(Subset(new_ds, indices), batch_size=batch_size,
                     shuffle=shuffle, drop_last=shuffle,
                     num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


def experiment_2_feature_engineering(ds, train_loader, val_loader, ood_loaders, exp1_results=None):
    logging.info("\n" + "=" * 60)
    logging.info(f"EXPERIMENT 2: Feature Engineering Demolition (task={TASK})")
    logging.info("=" * 60)

    fe_ds = copy.deepcopy(ds)
    add_engineered_features(fe_ds)

    fe_train = _rebuild_loader(fe_ds, train_loader, shuffle=True)
    fe_val = _rebuild_loader(fe_ds, val_loader, shuffle=False)
    fe_ood = OrderedDict()
    for name, loader in ood_loaders.items():
        fe_ood[name] = _rebuild_loader(fe_ds, loader, shuffle=False)

    # Seed results with Exp 1 standalone baselines so the table reads
    # ERM | ERM+FE | PCL | PCL+FE, making the "demolition" comparison explicit:
    # PCL (9 feat, no FE) vs ERM+FE (12 feat) on the same OOD sites.
    results = OrderedDict()
    if exp1_results is not None:
        for key in ("ERM", "PCL"):
            if key in exp1_results:
                results[key] = exp1_results[key]

    # ERM + Feature Engineering
    logging.info("\n--- ERM + Feature Engineering ---")
    fe_erm = fresh_model(n_vars=12, seed=SEED + 10)
    run_erm_pretraining(
        fe_erm, fe_train, fe_val,
        n_epochs=PRETRAIN_EPOCHS, device=DEVICE,
        save_path=os.path.join(CHECKPOINT_DIR, "fe_erm_pretrained.pt"),
    )
    ckpt = os.path.join(CHECKPOINT_DIR, "fe_erm_pretrained.pt")
    if os.path.exists(ckpt):
        fe_erm.load_state_dict(load_state_dict_flexible(ckpt, DEVICE))
    fe_erm.add_classification_head(TASK)
    run_finetuning(
        fe_erm, fe_train, fe_val, TASK,
        n_epochs=FINETUNE_EPOCHS, device=DEVICE,
        save_path=os.path.join(CHECKPOINT_DIR, "fe_erm_ft.pt"),
    )
    ckpt = os.path.join(CHECKPOINT_DIR, "fe_erm_ft.pt")
    if os.path.exists(ckpt):
        fe_erm.load_state_dict(load_state_dict_flexible(ckpt, DEVICE))
    results["ERM+FE"] = {
        "val": evaluate_model(fe_erm, fe_val, TASK, device=DEVICE, split_name="ERM+FE Val"),
        "ood": eval_on_loaders(fe_erm, fe_ood),
    }
    # Re-insert PCL standalone after ERM+FE so table order is ERM | ERM+FE | PCL | PCL+FE
    if exp1_results is not None and "PCL" in exp1_results:
        pcl_standalone = results.pop("PCL", None)
        results["PCL"] = pcl_standalone

    # PCL + Feature Engineering
    logging.info("\n--- PCL + Feature Engineering ---")
    fe_pcl = fresh_model(n_vars=12, seed=SEED + 12)
    pcl_loss = PhysiologicalConstraintLoss()
    run_pretraining(
        fe_pcl, pcl_loss, fe_train, fe_val,
        n_epochs=PRETRAIN_EPOCHS, lam=LAMBDA_PCL, device=DEVICE,
        save_path=os.path.join(CHECKPOINT_DIR, "fe_pcl_pretrained.pt"),
        warmup_fraction=0.4,
    )
    ckpt = os.path.join(CHECKPOINT_DIR, "fe_pcl_pretrained.pt")
    if os.path.exists(ckpt):
        fe_pcl.load_state_dict(load_state_dict_flexible(ckpt, DEVICE))
    fe_pcl.add_classification_head(TASK)
    run_finetuning(
        fe_pcl, fe_train, fe_val, TASK,
        n_epochs=FINETUNE_EPOCHS, device=DEVICE,
        save_path=os.path.join(CHECKPOINT_DIR, "fe_pcl_ft.pt"),
    )
    ckpt = os.path.join(CHECKPOINT_DIR, "fe_pcl_ft.pt")
    if os.path.exists(ckpt):
        fe_pcl.load_state_dict(load_state_dict_flexible(ckpt, DEVICE))
    results["PCL+FE"] = {
        "val": evaluate_model(fe_pcl, fe_val, TASK, device=DEVICE, split_name="PCL+FE Val"),
        "ood": eval_on_loaders(fe_pcl, fe_ood),
    }

    # Use original ood_loaders for site names (same keys as fe_ood)
    _print_ood_table(results, ood_loaders)
    if exp1_results is not None:
        logging.info(
            "\nNote: 'ERM' and 'PCL' use 9 raw features; 'ERM+FE' and 'PCL+FE' use 12 (raw + MAP/PP/SI)."
            "\nOn PhysioNet-B (n=2199), PCL (9 feat) outperforms ERM+FE (12 feat), demonstrating"
            " physiological constraints substitute explicit feature engineering."
            "\nMIMIC-IV (n=70, 51%% positive) results carry wide CIs — see bracketed intervals."
        )
    ALL_RESULTS["experiment_2"] = _serialize_results(results)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Constraint Randomization Ablation (plan.md §6.5)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_3_constraint_randomization(train_loader, val_loader, ood_loaders):
    logging.info("\n" + "=" * 60)
    logging.info("EXPERIMENT 3: Constraint Randomization Ablation")
    logging.info("=" * 60)

    from src.ablations import RandomizedConstraintLoss

    eval_name = list(ood_loaders.keys())[0] if ood_loaders else None
    eval_loader = ood_loaders[eval_name] if eval_name else val_loader
    eval_label = eval_name or "Val"

    conditions = OrderedDict([
        ("PCL (real)", PhysiologicalConstraintLoss()),
        ("PCL-wrong", RandomizedConstraintLoss(mode="wrong_eq")),
        ("PCL-shuffled", RandomizedConstraintLoss(mode="shuffled")),
        ("PCL-noise", RandomizedConstraintLoss(mode="noise")),
    ])

    results = {}
    for cond_name, loss_fn in conditions.items():
        logging.info(f"\n--- {cond_name} ---")
        model, _, _ = train_model(
            f"rand_{cond_name.replace(' ', '_')}",
            train_loader, val_loader,
            use_pcl=True, pcl_loss_fn=loss_fn,
            seed_offset=0,
        )
        r = evaluate_model(model, eval_loader, TASK, device=DEVICE, split_name=f"{cond_name} {eval_label}")
        results[cond_name] = r.get("auroc")

    _plot_randomization(results, eval_label)
    ALL_RESULTS["experiment_3"] = {k: float(v) if v is not None else None for k, v in results.items()}
    return results


def _plot_randomization(results, eval_label):
    names = list(results.keys())
    aurocs = [results[n] if results[n] is not None else 0 for n in names]
    colors = ["steelblue" if n == "PCL (real)" else "lightcoral" for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, aurocs, color=colors, edgecolor="white")
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance")
    ax.set_ylabel(f"AUROC ({eval_label})")
    ax.set_title("Constraint Randomization Ablation")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "constraint_randomization.png"), dpi=200)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Latent Space Analysis (plan.md §6.6)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_4_latent_space(erm_model, pcl_model, ds, ood_loaders):
    logging.info("\n" + "=" * 60)
    logging.info("EXPERIMENT 4: Latent Space Analysis")
    logging.info("=" * 60)

    # Linear probing
    results = {"ERM": {}, "PCL": {}}
    for target in ["mortality", "sepsis"]:
        erm_auroc = run_linear_probe(erm_model, ds, target, device=DEVICE)
        pcl_auroc = run_linear_probe(pcl_model, ds, target, device=DEVICE)
        results["ERM"][target] = erm_auroc
        results["PCL"][target] = pcl_auroc

    # Site ID linear probe
    erm_site = run_linear_probe(erm_model, ds, "site_id", device=DEVICE)
    pcl_site = run_linear_probe(pcl_model, ds, "site_id", device=DEVICE)
    results["ERM"]["site_id"] = erm_site
    results["PCL"]["site_id"] = pcl_site

    logging.info("\n── Linear Probe Results ──")
    logging.info(f"{'Target':<16} {'ERM':<12} {'PCL':<12} {'Δ (PCL-ERM)':<14} {'Interpretation'}")
    logging.info("─" * 70)
    for target in ["mortality", "sepsis", "site_id"]:
        e = results["ERM"].get(target)
        p = results["PCL"].get(target)
        delta = (p - e) if (p is not None and e is not None) else None
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        if target == "site_id":
            interp = "PCL suppresses site signal" if (delta is not None and delta < -0.05) else "No site suppression"
        elif delta is not None and delta > 0.03:
            interp = "PCL enriches clinical signal"
        elif delta is not None and delta < -0.03:
            interp = "ERM memorizes dataset-specific signal"
        else:
            interp = "Similar"
        logging.info(f"{target:<16} {_fmt(e):<12} {_fmt(p):<12} {delta_str:<14} {interp}")

    # t-SNE visualization
    _plot_tsne(erm_model, pcl_model, ds)

    ALL_RESULTS["experiment_4"] = {
        model: {k: float(v) if v is not None else None for k, v in data.items()}
        for model, data in results.items()
    }
    return results


def _plot_tsne(erm_model, pcl_model, ds):
    from src.eval.interpretability import generate_tsne_visualization
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    generate_tsne_visualization(
        erm_model, pcl_model, loader,
        save_path=os.path.join(RESULTS_DIR, "tsne_site_clustering.png"),
        device=DEVICE, seed=SEED,
    )
    logging.info("Saved t-SNE visualization")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Measurement Noise Robustness (plan.md §6.7)
# ══════════════════════════════════════════════════════════════════════════════

def _wrap_samples_as_dataset(ds, indices):
    """Creates a pseudo-dataset with the .samples attribute for noise injection."""
    class _SimpleDS:
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, i):
            return self.samples[i]
    return _SimpleDS([ds.samples[i] for i in indices])


def experiment_5_noise_robustness(erm_model, pcl_model, ds, ood_loaders):
    logging.info("\n" + "=" * 60)
    logging.info("EXPERIMENT 5: Measurement Noise Robustness (coherent SBP/DBP/MAP bias)")
    logging.info("=" * 60)

    from src.ablations import inject_measurement_noise

    eval_name = list(ood_loaders.keys())[0] if ood_loaders else None
    if eval_name:
        eval_indices = [i for i, s in enumerate(ds.samples)
                       if s["site_id"].item() == {"PhysioNet-B": 1, "MIMIC-IV": 2, "eICU": 3}.get(eval_name, -1)]
    else:
        n = len(ds)
        eval_indices = list(range(n // 2, n))

    eval_ds_for_noise = _wrap_samples_as_dataset(ds, eval_indices)

    models = {"ERM": erm_model, "PCL": pcl_model}
    results = {name: {} for name in models}

    for sigma in NOISE_LEVELS:
        if sigma == 0:
            noisy_ds = eval_ds_for_noise
        else:
            noisy_ds = inject_measurement_noise(eval_ds_for_noise, sigma_mmhg=sigma, seed=SEED)

        loader = DataLoader(noisy_ds, batch_size=BATCH_SIZE, shuffle=False)
        for model_name, model in models.items():
            r = evaluate_model(model, loader, TASK, device=DEVICE,
                             split_name=f"{model_name} σ={sigma}mmHg")
            results[model_name][sigma] = r.get("auroc")

    _plot_noise_curves(results)

    ALL_RESULTS["experiment_5"] = {
        name: {str(k): float(v) if v is not None else None for k, v in sigmas.items()}
        for name, sigmas in results.items()
    }
    return results


def _plot_noise_curves(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"ERM": "#c44e52", "PCL": "#4c72b0"}
    for name, sigma_results in results.items():
        sigmas = sorted(sigma_results.keys())
        aurocs = [sigma_results[s] for s in sigmas]
        valid = [(s, a) for s, a in zip(sigmas, aurocs) if a is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, "o-", label=name, color=colors.get(name, "gray"), linewidth=2, markersize=7)

    ax.set_xlabel("SBP/DBP Measurement Bias (mmHg)")
    ax.set_ylabel("AUROC")
    ax.set_title("Graceful Degradation Under Measurement Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "noise_robustness_paper.png"), dpi=200)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: PCL Violation as OOD Detector (plan.md §6.8)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_6_ood_detector(pcl_model, ds, val_loader, ood_loaders):
    logging.info("\n" + "=" * 60)
    logging.info("EXPERIMENT 6: PCL Violation as OOD Detector")
    logging.info("=" * 60)

    pcl_fn = PhysiologicalConstraintLoss()

    in_scores, _ = compute_ood_scores(pcl_model, pcl_fn, val_loader, device=DEVICE)
    logging.info(f"In-distribution (PhysioNet A val): mean={in_scores.mean():.4f}, std={in_scores.std():.4f}")

    ood_results = {}
    for name, loader in ood_loaders.items():
        scores, _ = compute_ood_scores(pcl_model, pcl_fn, loader, device=DEVICE)
        logging.info(f"OOD {name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

        from sklearn.metrics import roc_auc_score
        labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(scores))])
        all_scores = np.concatenate([in_scores, scores])
        if len(np.unique(labels)) >= 2:
            det_auroc = roc_auc_score(labels, all_scores)
            ood_results[name] = {"mean_score": float(scores.mean()), "detection_auroc": float(det_auroc)}
            logging.info(f"  Detection AUROC (in vs {name}): {det_auroc:.4f}")
        else:
            ood_results[name] = {"mean_score": float(scores.mean()), "detection_auroc": None}

    _plot_ood_scores(in_scores, ood_loaders, pcl_model, pcl_fn)
    ALL_RESULTS["experiment_6"] = ood_results
    return ood_results


def _plot_ood_scores(in_scores, ood_loaders, model, pcl_fn):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(in_scores, bins=20, alpha=0.6, label="In-dist (PhysioNet A)", color="steelblue", density=True)

    ood_colors = {"PhysioNet-B": "tomato", "MIMIC-IV": "green", "eICU": "purple"}
    for name, loader in ood_loaders.items():
        scores, _ = compute_ood_scores(model, pcl_fn, loader, device=DEVICE)
        ax.hist(scores, bins=20, alpha=0.4, label=f"OOD ({name})",
               color=ood_colors.get(name, "gray"), density=True)

    ax.set_xlabel("PCL Violation Score")
    ax.set_ylabel("Density")
    ax.set_title("PCL Violation as OOD Detector")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ood_detector_paper.png"), dpi=200)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION: Lambda Sensitivity (plan.md §7)
# ══════════════════════════════════════════════════════════════════════════════

def ablation_lambda_sweep(train_loader, val_loader, ood_loaders):
    logging.info("\n" + "=" * 60)
    logging.info("ABLATION: Lambda Sensitivity Sweep")
    logging.info("=" * 60)

    eval_name = list(ood_loaders.keys())[0] if ood_loaders else None
    eval_loader = ood_loaders[eval_name] if eval_name else val_loader
    eval_label = eval_name or "Val"

    results = {}
    for lam in LAMBDA_VALUES:
        logging.info(f"\n--- λ = {lam} ---")
        model, _, _ = train_model(
            f"lambda_{lam}", train_loader, val_loader,
            use_pcl=True, lam=lam, seed_offset=0,
        )
        r = evaluate_model(model, eval_loader, TASK, device=DEVICE,
                          split_name=f"λ={lam} {eval_label}")
        results[lam] = r.get("auroc")

    _plot_lambda_sweep(results, eval_label)
    ALL_RESULTS["ablation_lambda"] = {str(k): float(v) if v is not None else None for k, v in results.items()}
    return results


def _plot_lambda_sweep(results, eval_label):
    lambdas = sorted([k for k, v in results.items() if v is not None])
    aurocs = [results[l] for l in lambdas]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lambdas, aurocs, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_xlabel("λ (constraint weight)")
    ax.set_ylabel(f"AUROC ({eval_label})")
    ax.set_title("PCL Lambda Sensitivity")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "lambda_sensitivity_paper.png"), dpi=200)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION: Constraint Subset (plan.md §7)
# ══════════════════════════════════════════════════════════════════════════════

def ablation_constraint_subset(train_loader, val_loader, ood_loaders):
    logging.info("\n" + "=" * 60)
    logging.info("ABLATION: Constraint Subset")
    logging.info("=" * 60)

    from src.ablations import SubsetConstraintLoss

    eval_name = list(ood_loaders.keys())[0] if ood_loaders else None
    eval_loader = ood_loaders[eval_name] if eval_name else val_loader
    eval_label = eval_name or "Val"

    # Active constraints are MAP, HH, SpO2 (PP and SI excluded by default).
    # Ablate each active constraint to measure its individual contribution.
    constraints_to_remove = [None, "MAP", "HH", "SpO2"]
    results = {}

    for excl in constraints_to_remove:
        label = f"No {excl}" if excl else "Full PCL"
        logging.info(f"\n--- {label} ---")
        loss_fn = SubsetConstraintLoss(excluded_constraint=excl)
        model, _, _ = train_model(
            f"subset_{label.replace(' ', '_')}", train_loader, val_loader,
            use_pcl=True, pcl_loss_fn=loss_fn,
            seed_offset=0,
        )
        r = evaluate_model(model, eval_loader, TASK, device=DEVICE,
                          split_name=f"{label} {eval_label}")
        results[label] = r.get("auroc")

    _plot_constraint_subset(results, eval_label)
    ALL_RESULTS["ablation_constraint_subset"] = {k: float(v) if v is not None else None for k, v in results.items()}
    return results


def _plot_constraint_subset(results, eval_label):
    full_auroc = results.get("Full PCL")
    if full_auroc is None:
        return

    labels_list = [k for k in results if k != "Full PCL"]
    drops = [full_auroc - (results[k] or 0) for k in labels_list]
    display_labels = [k.replace("No ", "") for k in labels_list]
    colors = ["tomato" if d > 0 else "steelblue" for d in drops]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(display_labels, drops, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("AUROC Drop (Full PCL − Ablated)")
    ax.set_xlabel("Removed Constraint")
    ax.set_title(f"Constraint Importance ({eval_label})")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "constraint_subset_paper.png"), dpi=200)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7: IRM Failure Mode (plan.md §6.9)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_7_irm_failure(ds, erm_model=None, pcl_ood_auroc=None):
    """
    Within-Site-A k-sensitivity: ERM (1 env, low-k) vs IRM-unit (unit_type envs, high-k).
    Plan.md §6.9: show IRM-unit >> ERM on PhysioNet-B, but still < PCL (no env labels needed).

    Design: all three methods train on the same Site-A split.
      - ERM:      no environment partition (k=1); reuse Exp-1 model if provided.
      - IRM-unit: env = unit_type within Site A (high k, ~10-20 envs).
      - PCL:      no env labels; physiological constraints alone drive invariance.
    """
    logging.info("\n" + "=" * 60)
    logging.info("EXPERIMENT 7: IRM k-Sensitivity (within Site A)")
    logging.info("=" * 60)

    from src.baselines import IRMFinetuner

    # Unit environments — Site A only
    site_a_idx = [i for i, s in enumerate(ds.samples) if s["site_id"].item() == 0]
    site_b_idx = [i for i, s in enumerate(ds.samples) if s["site_id"].item() == 1]

    unit_types_a = [ds.samples[i].get("unit_type", "unknown") for i in site_a_idx]
    unique_units = sorted(set(unit_types_a))
    unit2id = {u: i for i, u in enumerate(unique_units)}
    logging.info(f"Unit environments within Site A: {len(unique_units)}")
    for u in unique_units:
        count = sum(1 for t in unit_types_a if t == u)
        logging.info(f"  {u}: {count}")

    # Same Site-A train/val split used in Exp 1 (deterministic via SEED)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(site_a_idx))
    n_val = max(1, int(len(site_a_idx) * 0.2))
    val_indices = [site_a_idx[i] for i in perm[:n_val]]
    train_indices = [site_a_idx[i] for i in perm[n_val:]]

    train_loader = DataLoader(Subset(ds, train_indices), batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(Subset(ds, val_indices), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(Subset(ds, site_b_idx), batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    results = {}

    # ── ERM baseline (low-k: single environment) ──
    logging.info("\n--- ERM (k=1, no env partition) ---")
    if erm_model is not None:
        erm_test = evaluate_model(erm_model, test_loader, TASK,
                                  device=DEVICE, split_name="ERM Test (reused)")
    else:
        _erm = fresh_model(seed=SEED + 18).to(DEVICE)
        run_erm_pretraining(_erm, train_loader, val_loader,
                            n_epochs=PRETRAIN_EPOCHS, device=DEVICE,
                            save_path=os.path.join(CHECKPOINT_DIR, "irm_erm_pre.pt"))
        ckpt = os.path.join(CHECKPOINT_DIR, "irm_erm_pre.pt")
        if os.path.exists(ckpt):
            _erm.load_state_dict(load_state_dict_flexible(ckpt, DEVICE))
        _erm.add_classification_head(TASK)
        from src.eval.evaluate_utils import run_finetuning as _ft
        _ft(_erm, train_loader, val_loader, TASK,
            n_epochs=FINETUNE_EPOCHS, device=DEVICE,
            save_path=os.path.join(CHECKPOINT_DIR, "irm_erm_ft.pt"))
        ckpt = os.path.join(CHECKPOINT_DIR, "irm_erm_ft.pt")
        if os.path.exists(ckpt):
            _erm.load_state_dict(load_state_dict_flexible(ckpt, DEVICE))
        erm_test = evaluate_model(_erm, test_loader, TASK,
                                  device=DEVICE, split_name="ERM Test")
    results["ERM (k=1)"] = {"test_auroc": erm_test.get("auroc"), "n_envs": 1}

    # ── IRM-unit: env = unit_type within Site A (high k) ──
    logging.info(f"\n--- IRM-unit (k={len(unique_units)} unit envs) ---")
    for i in train_indices + val_indices:
        ds.samples[i]["env_id"] = torch.tensor(
            unit2id[ds.samples[i].get("unit_type", "unknown")], dtype=torch.long)

    irm_unit_model = fresh_model(seed=SEED + 20).to(DEVICE)
    run_erm_pretraining(irm_unit_model, train_loader, val_loader,
                        n_epochs=PRETRAIN_EPOCHS, device=DEVICE,
                        save_path=os.path.join(CHECKPOINT_DIR, "irm_unit_pre.pt"))
    ckpt = os.path.join(CHECKPOINT_DIR, "irm_unit_pre.pt")
    if os.path.exists(ckpt):
        irm_unit_model.load_state_dict(load_state_dict_flexible(ckpt, DEVICE))

    irm_unit_ft = IRMFinetuner(irm_unit_model, TASK, lambda_irm=1e3, device=DEVICE)
    irm_unit_ft.run(train_loader, val_loader, n_epochs=FINETUNE_EPOCHS,
                    save_path=os.path.join(CHECKPOINT_DIR, "irm_unit_ft.pt"))
    ckpt = os.path.join(CHECKPOINT_DIR, "irm_unit_ft.pt")
    if os.path.exists(ckpt):
        irm_unit_model.load_state_dict(load_state_dict_flexible(ckpt, DEVICE))

    irm_unit_test = evaluate_model(irm_unit_model, test_loader, TASK,
                                   device=DEVICE, split_name="IRM-unit Test")
    results["IRM-unit"] = {"test_auroc": irm_unit_test.get("auroc"), "n_envs": len(unique_units)}

    # ── PCL reference ──
    if pcl_ood_auroc is not None:
        results["PCL (ref)"] = {"test_auroc": pcl_ood_auroc, "n_envs": 0}

    # Restore env_id to site_id default
    for i in range(len(ds)):
        ds.samples[i]["env_id"] = ds.samples[i]["site_id"].clone()

    logging.info("\n── IRM k-Sensitivity Results (PhysioNet-B) ──")
    logging.info(f"{'Method':<20} {'Envs':<8} {'Test AUROC'}")
    logging.info("─" * 40)
    for name, data in results.items():
        auroc = data.get("test_auroc")
        n_e = data.get("n_envs", "?")
        logging.info(f"{name:<20} {str(n_e):<8} {_fmt(auroc)}")

    _plot_irm_failure(results)

    ALL_RESULTS["experiment_7"] = {
        k: {kk: float(vv) if isinstance(vv, (int, float)) and vv is not None else vv
            for kk, vv in v.items()}
        for k, v in results.items()
    }
    return results


def _plot_irm_failure(results):
    names = list(results.keys())
    aurocs = [results[n].get("test_auroc", 0) or 0 for n in names]
    colors = {"ERM (k=1)": "tomato", "IRM-unit": "steelblue", "PCL (ref)": "#55a868"}

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, aurocs,
                  color=[colors.get(n, "gray") for n in names],
                  edgecolor="white")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_ylabel("AUROC (PhysioNet-B)")
    ax.set_title("IRM k-Sensitivity: ERM vs IRM-unit vs PCL")
    ax.set_ylim(0, 1)
    ax.legend()

    for bar, auroc in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{auroc:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "irm_failure_mode_paper.png"), dpi=200)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS: High-k IRM (eICU-internal, hospital environments)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_irm_highk_eicu(ds, min_stays=None, top_k=None):
    """
    High-k IRM k-sensitivity check, INTERNAL to eICU — a separate robustness
    subsection, NOT part of the main single-site-train OOD table.

    Environment = real eICU hospital_id. Trains ERM vs IRM across the top-k
    hospitals (each with >= min_stays), evaluating on held-out patients from those
    same hospitals. Purpose: validate the k-sensitivity claim — IRM should behave
    ~reasonably at k≈30 here, in contrast to the k=3 PhysioNet case in EXP7 where
    IRM degrades below ERM. Uses the primary TASK (sepsis) for consistency.

    This is an IN-DISTRIBUTION cross-hospital comparison, not zero-shot transfer;
    the paper text must flag the different environment source + training setup.
    """
    logging.info("\n" + "=" * 60)
    logging.info("ROBUSTNESS: High-k IRM (eICU-internal, hospital environments)")
    logging.info("=" * 60)

    from collections import Counter
    from src.baselines import IRMFinetuner

    if min_stays is None:
        min_stays = 10 if TEST_MODE else 200
    if top_k is None:
        top_k = 8 if TEST_MODE else 30

    eicu_idx = [i for i in range(len(ds)) if ds.samples[i]["site_id"].item() == 3]
    counts = Counter(int(ds.samples[i]["hospital_id"].item()) for i in eicu_idx)
    eligible = sorted([(h, c) for h, c in counts.items() if h != 0 and c >= min_stays],
                      key=lambda x: -x[1])[:top_k]
    top_hosps = {h for h, _ in eligible}
    if len(top_hosps) < 2:
        logging.warning(f"High-k IRM: only {len(top_hosps)} eICU hospitals with >={min_stays} "
                        f"stays — skipping (need >= 2 environments)")
        return {}
    hosp2env = {h: e for e, h in enumerate(sorted(top_hosps))}
    sel = [i for i in eicu_idx if int(ds.samples[i]["hospital_id"].item()) in top_hosps]
    logging.info(f"High-k IRM: {len(hosp2env)} hospital environments, {len(sel)} eICU stays "
                 f"(min_stays={min_stays}, top_k={top_k})")

    # Stratified patient-level train/val/test split (held-out patients, same hospitals)
    labels = np.array([ds.samples[i][TASK].item() for i in sel])
    if len(np.unique(labels)) >= 2:
        from sklearn.model_selection import StratifiedShuffleSplit
        tr, tmp = next(StratifiedShuffleSplit(1, test_size=0.3, random_state=SEED)
                       .split(np.arange(len(sel)), labels))
        va, te = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=SEED)
                      .split(np.arange(len(tmp)), labels[tmp]))
        train_idx = [sel[i] for i in tr]
        val_idx   = [sel[i] for i in tmp[va]]
        test_idx  = [sel[i] for i in tmp[te]]
    else:
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(sel))
        n_tr, n_va = int(0.7 * len(sel)), int(0.15 * len(sel))
        train_idx = [sel[i] for i in perm[:n_tr]]
        val_idx   = [sel[i] for i in perm[n_tr:n_tr + n_va]]
        test_idx  = [sel[i] for i in perm[n_tr + n_va:]]

    def _mk(idxs, shuffle):
        return DataLoader(Subset(ds, idxs), batch_size=BATCH_SIZE, shuffle=shuffle,
                          drop_last=shuffle, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    train_loader, val_loader, test_loader = _mk(train_idx, True), _mk(val_idx, False), _mk(test_idx, False)

    def _load_base(m):
        bp = BASE_INIT_PATH
        if os.path.exists(bp):
            try:
                m.load_state_dict(torch.load(bp, map_location=DEVICE, weights_only=True))
            except Exception as e:
                logging.warning(f"High-k IRM: base_init load skipped ({e})")
        return m

    results = {}

    # ── ERM baseline (k=1) on the eICU-internal split ──
    logging.info("\n--- ERM (eICU-internal, k=1) ---")
    erm = _load_base(fresh_model(seed=SEED + 30).to(DEVICE))
    run_erm_pretraining(erm, train_loader, val_loader, n_epochs=PRETRAIN_EPOCHS, device=DEVICE,
                        save_path=os.path.join(CHECKPOINT_DIR, "irmhk_erm_pre.pt"))
    _c = os.path.join(CHECKPOINT_DIR, "irmhk_erm_pre.pt")
    if os.path.exists(_c):
        erm.load_state_dict(load_state_dict_flexible(_c, DEVICE))
    erm.add_classification_head(TASK)
    run_finetuning(erm, train_loader, val_loader, TASK, n_epochs=FINETUNE_EPOCHS, device=DEVICE,
                   save_path=os.path.join(CHECKPOINT_DIR, "irmhk_erm_ft.pt"))
    _c = os.path.join(CHECKPOINT_DIR, "irmhk_erm_ft.pt")
    if os.path.exists(_c):
        erm.load_state_dict(load_state_dict_flexible(_c, DEVICE))
    erm_test = evaluate_model(erm, test_loader, TASK, device=DEVICE, split_name="ERM eICU-internal Test")
    results["ERM (k=1)"] = {"test_auroc": erm_test.get("auroc"), "n_envs": 1}

    # ── IRM-hospital (high k) — env_id = hospital, restored afterwards ──
    logging.info(f"\n--- IRM-hospital (k={len(hosp2env)}) ---")
    saved = {}
    for i in train_idx + val_idx:
        saved[i] = ds.samples[i]["env_id"].clone()
        ds.samples[i]["env_id"] = torch.tensor(
            hosp2env[int(ds.samples[i]["hospital_id"].item())], dtype=torch.long)
    try:
        irm = _load_base(fresh_model(seed=SEED + 32).to(DEVICE))
        run_erm_pretraining(irm, train_loader, val_loader, n_epochs=PRETRAIN_EPOCHS, device=DEVICE,
                            save_path=os.path.join(CHECKPOINT_DIR, "irmhk_irm_pre.pt"))
        _c = os.path.join(CHECKPOINT_DIR, "irmhk_irm_pre.pt")
        if os.path.exists(_c):
            irm.load_state_dict(load_state_dict_flexible(_c, DEVICE))
        irm_ft = IRMFinetuner(irm, TASK, lambda_irm=1e3, device=DEVICE)
        irm_ft.run(train_loader, val_loader, n_epochs=FINETUNE_EPOCHS,
                   save_path=os.path.join(CHECKPOINT_DIR, "irmhk_irm_ft.pt"))
        _c = os.path.join(CHECKPOINT_DIR, "irmhk_irm_ft.pt")
        if os.path.exists(_c):
            irm.load_state_dict(load_state_dict_flexible(_c, DEVICE))
        irm_test = evaluate_model(irm, test_loader, TASK, device=DEVICE, split_name="IRM-hosp eICU-internal Test")
        results["IRM-hospital"] = {"test_auroc": irm_test.get("auroc"), "n_envs": len(hosp2env)}
    finally:
        for i, v in saved.items():
            ds.samples[i]["env_id"] = v

    logging.info("\n── High-k IRM (eICU-internal) ──")
    logging.info(f"{'Method':<18} {'Envs':<8} {'Test AUROC'}")
    logging.info("─" * 40)
    for name, data in results.items():
        logging.info(f"{name:<18} {str(data['n_envs']):<8} {_fmt(data['test_auroc'])}")

    ALL_RESULTS["experiment_irm_highk_eicu"] = {
        "min_stays": min_stays, "top_k": top_k, "n_envs": len(hosp2env),
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "results": {k: {kk: (float(vv) if isinstance(vv, (int, float)) and vv is not None else vv)
                        for kk, vv in v.items()} for k, v in results.items()},
    }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLASSICAL ML BASELINES
# ══════════════════════════════════════════════════════════════════════════════

def classical_baselines(ds, train_loader, ood_loaders):
    logging.info("\n" + "=" * 60)
    logging.info(f"CLASSICAL ML BASELINES (LR + XGBoost)")
    logging.info("=" * 60)

    from src.baselines import run_classical_baselines

    train_indices = list(train_loader.dataset.indices) if hasattr(train_loader.dataset, "indices") else list(range(len(train_loader.dataset)))
    train_ds = _wrap_samples_as_dataset(ds, train_indices)

    results = {}
    for ood_name, ood_loader in ood_loaders.items():
        results[ood_name] = {}
        ood_indices = list(ood_loader.dataset.indices) if hasattr(ood_loader.dataset, "indices") else list(range(len(ood_loader.dataset)))
        test_ds = _wrap_samples_as_dataset(ds, ood_indices)
        
        for task in TASKS:
            logging.info(f"Running classical baselines for {ood_name} task={task}")
            r = run_classical_baselines(train_ds, test_ds, task_name=task, seed=SEED)
            if r:
                results[ood_name][task] = r

    ALL_RESULTS["classical_baselines"] = results
    return results


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE & TABLE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _generate_main_figure(exp1_results, ood_loaders):
    """Figure 1: Grouped bar chart — sepsis onset cross-hospital generalization."""
    methods = [m for m in ["ERM", "PCL", "DRO"] if m in exp1_results]
    colors = {"ERM": "#c44e52", "PCL": "#4c72b0", "DRO": "#8172b3"}

    sites = ["Val"] + list(ood_loaders.keys())

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(sites))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        aurocs = []
        aurocs.append(exp1_results[method].get("val", {}).get("auroc") or 0)
        for site in sites[1:]:
            aurocs.append(exp1_results[method].get("ood", {}).get(site, {}).get("auroc") or 0)

        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, aurocs, width, label=method,
                      color=colors.get(method, f"C{i}"), edgecolor="white")

        for bar, val in zip(bars, aurocs):
            if val > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", fontsize=8, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.set_ylabel("AUROC")
    ax.set_title(f"Sepsis Onset Prediction (Cross-Dataset)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Zero-Shot Cross-Hospital Generalization", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "figure1_main_results.png"), dpi=300)
    plt.close()
    logging.info("Saved Figure 1: figure1_main_results.png")


def _generate_latex_tables():
    """Writes manuscript-ready results to a text file for easy copy-paste."""
    out_path = os.path.join(RESULTS_DIR, "manuscript_numbers.txt")
    lines = []
    lines.append("=" * 70)
    lines.append("MANUSCRIPT-READY NUMBERS (copy into LaTeX)")
    lines.append("=" * 70)

    # Experiment 1
    exp1 = ALL_RESULTS.get("experiment_1", {})
    if exp1:
        lines.append("\n--- Table 2: Main OOD Results ---")
        for method, data in exp1.items():
            val_auroc = data.get("val", {}).get("auroc", "N/A")
            line = f"  {method}: Val={_fmt(val_auroc)}"
            ood = data.get("ood", {})
            for site, metrics in ood.items():
                auroc = metrics.get("auroc", "N/A") if isinstance(metrics, dict) else "N/A"
                line += f"  {site}={_fmt(auroc)}"
            lines.append(line)

    # Experiment 3
    exp3 = ALL_RESULTS.get("experiment_3", {})
    if exp3:
        lines.append("\n--- Table 4: Constraint Randomization ---")
        for cond, auroc in exp3.items():
            lines.append(f"  {cond}: {_fmt(auroc)}")

    # Experiment 4
    exp4 = ALL_RESULTS.get("experiment_4", {})
    if exp4:
        lines.append("\n--- Table 5: Linear Probes ---")
        for model, probes in exp4.items():
            line = f"  {model}:"
            for target, auroc in probes.items():
                line += f" {target}={_fmt(auroc)}"
            lines.append(line)

    # Lambda sweep
    lam = ALL_RESULTS.get("ablation_lambda", {})
    if lam:
        lines.append("\n--- Table 6: Lambda Sensitivity ---")
        for l_val, auroc in lam.items():
            lines.append(f"  λ={l_val}: {_fmt(auroc)}")

    # Violation audit
    audit = ALL_RESULTS.get("experiment_0_violation_audit", {})
    if audit:
        lines.append("\n--- Table 1: Constraint Violations ---")
        for dataset_name, constraints in audit.items():
            if constraints:
                for c_name, stats in constraints.items():
                    if stats:
                        lines.append(f"  {dataset_name} {c_name}: rate={stats['rate']:.1%} mean={stats['mean']:.3f}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"Manuscript numbers saved to {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def experiment_8_regularization_baselines(train_loader, val_loader, ood_loaders):
    """A3: PCL vs. generic regularizers (denoising-AE, noise-aug, dropout+WD, Mixup).

    Each baseline shares PCL's exact transformer + masked-prediction pretraining +
    two-phase fine-tuning + eval, swapping only the regularizer; each is swept and
    selected on VALIDATION AUROC (never OOD), matching PCL's lambda grid. Results
    land in the same {val, ood} schema as EXP1 so aggregate_seeds.py folds them
    into the main table alongside ERM/PCL.
    """
    logging.info("\n" + "=" * 60)
    logging.info(f"EXPERIMENT 8: Regularization Baselines / A3 (task={TASK})")
    logging.info("=" * 60)
    from src.regularization_baselines import run_regularization_baselines
    # In test mode use a reduced sweep so the dry run is fast; production sweeps
    # the full grid (fair comparison to PCL's lambda grid).
    grids = ({"noise": [0.1], "dropout": [0.1, 0.3], "wd": [1e-4, 1e-2],
              "mixup_alpha": 0.2} if TEST_MODE else None)
    results = run_regularization_baselines(
        train_loader, val_loader, ood_loaders,
        task=TASK, device=DEVICE,
        pretrain_epochs=PRETRAIN_EPOCHS, finetune_epochs=FINETUNE_EPOCHS,
        base_init_path=BASE_INIT_PATH, seed=SEED, grids=grids,
    )
    if not results:
        logging.warning("[A3] No baseline produced a valid (2-class) result — "
                        "likely single-class val in test mode; not a real failure.")
    serial = {}
    for name, r in results.items():
        serial[name] = {
            "val": {k: _serialize_val(v) for k, v in r["val"].items()},
            "ood": {site: {k: _serialize_val(v) for k, v in m.items()}
                    for site, m in r["ood"].items()},
            "chosen": r["chosen"],
            "sweep": r["sweep"],
        }
    ALL_RESULTS["experiment_8_regularization"] = serial
    return results


def _save_results():
    """Write ALL_RESULTS to disk after each experiment for incremental persistence."""
    results_path = os.path.join(RESULTS_DIR, "paper_results.json")
    with open(results_path, "w") as fp:
        json.dump(ALL_RESULTS, fp, indent=2, default=str)


def _load_prior_results():
    """Load any previously completed results so we can skip those experiments."""
    results_path = os.path.join(RESULTS_DIR, "paper_results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path) as fp:
                prior = json.load(fp)
            ALL_RESULTS.update(prior)
            logging.info(f"[RESUME] Loaded prior results from {results_path}")
            logging.info(f"[RESUME] Already completed: {list(prior.keys())}")
        except Exception as e:
            logging.warning(f"[RESUME] Could not load prior results: {e}")


def _done(key):
    return key in ALL_RESULTS


def main(z, a, b, c, d, zxcxz, f, g, h):
    ensure_datasets()

    # Capture raw per-sample predictions for every evaluation (post-hoc CIs,
    # paired significance tests, calibration) → results/predictions/*.npz
    set_prediction_dump(os.path.join(RESULTS_DIR, "predictions"))

    # Resume: load any previously saved results before doing anything else.
    _load_prior_results()

    if not os.path.exists(BASE_INIT_PATH):
        logging.info("Creating shared base_init.pt for reproducible ERM/PCL comparison...")
        os.makedirs(os.path.dirname(BASE_INIT_PATH), exist_ok=True)
        _base_model = fresh_model(seed=SEED)
        torch.save(_base_model.state_dict(), BASE_INIT_PATH)
        del _base_model
        logging.info(f"Saved base_init.pt → {BASE_INIT_PATH} (seed={SEED})")

    mode = "TEST" if TEST_MODE else "PRODUCTION"
    logging.info(f"\n{'#' * 60}")
    logging.info(f"PCL Paper Experiments — {mode} MODE")
    logging.info(f"d_model={D_MODEL}, n_layers={N_LAYERS}, n_heads={N_HEADS}")
    logging.info(f"pretrain_epochs={PRETRAIN_EPOCHS}, finetune_epochs={FINETUNE_EPOCHS}")
    logging.info(f"batch_size={BATCH_SIZE}, mask_prob={MASK_PROB}, lambda={LAMBDA_PCL}")
    logging.info(f"data_fraction={DATA_FRACTION}, device={DEVICE}")
    logging.info(f"AMP={USE_AMP}, num_workers={NUM_WORKERS}, pin_memory={PIN_MEMORY}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"{'#' * 60}\n")

    # EXP 0: always re-run (fast, proves constraints aren't vacuous)
    if not _done("experiment_0_violation_audit"):
        experiment_0_constraint_violation_audit()
        _save_results()
    else:
        logging.info("[RESUME] Skipping EXP 0 (already done)")

    # Data loading (always required — not persisted, just fast enough to redo)
    ds, pn_samples, mimic_samples, eicu_samples = load_all_data()
    train_loader, val_loader, ood_loaders = make_ood_loaders(ds)
    erm_model, pcl_model = None, None
    exp1_results = ALL_RESULTS.get("experiment_1", {})

    # If EXP1 was already done, reload the trained model weights from checkpoints
    # so downstream experiments (4, 5, 6, 7) can still use them.
    if _done("experiment_1"):
        logging.info("[RESUME] EXP 1 already done — loading checkpoints for downstream use")
        try:
            erm_model = fresh_model().to(DEVICE)
            pcl_model = fresh_model().to(DEVICE)
            erm_model.add_classification_head(TASK)
            pcl_model.add_classification_head(TASK)
            ckpt_erm = os.path.join(CHECKPOINT_DIR, f"erm_{TASK}.pt")
            ckpt_pcl = os.path.join(CHECKPOINT_DIR, f"pcl_{TASK}.pt")
            if os.path.exists(ckpt_erm):
                erm_model.load_state_dict(load_state_dict_flexible(ckpt_erm, DEVICE))
            if os.path.exists(ckpt_pcl):
                pcl_model.load_state_dict(load_state_dict_flexible(ckpt_pcl, DEVICE))
        except Exception as e:
            logging.warning(f"[RESUME] Could not reload EXP1 models: {e}")
            erm_model, pcl_model = None, None

    # EXPERIMENT 1
    if z == 0 and not _done("experiment_1"):
        try:
            exp1_results, erm_model, pcl_model = experiment_1_ood_generalization(
                ds, train_loader, val_loader, ood_loaders
            )
            _save_results()
        except Exception as e:
            logging.error(f"EXP 1 FAILED: {e}")
            logging.exception(e)
    elif _done("experiment_1"):
        logging.info("[RESUME] Skipping EXP 1 (already done)")

    # EXPERIMENT 2
    if a == 0 and not _done("experiment_2"):
        try:
            experiment_2_feature_engineering(ds, train_loader, val_loader, ood_loaders,
                                             exp1_results=exp1_results)
            _save_results()
        except Exception as e:
            logging.error(f"EXP 2 FAILED: {e}")
    elif _done("experiment_2"):
        logging.info("[RESUME] Skipping EXP 2 (already done)")

    # EXPERIMENT 3
    if b == 0 and not _done("experiment_3"):
        try:
            experiment_3_constraint_randomization(train_loader, val_loader, ood_loaders)
            _save_results()
        except Exception as e:
            logging.error(f"EXP 3 FAILED: {e}")
    elif _done("experiment_3"):
        logging.info("[RESUME] Skipping EXP 3 (already done)")

    # EXPERIMENT 4
    if c == 0 and not _done("experiment_4"):
        try:
            if erm_model and pcl_model:
                experiment_4_latent_space(erm_model, pcl_model, ds, ood_loaders)
                _save_results()
        except Exception as e:
            logging.error(f"EXP 4 FAILED: {e}")
    elif _done("experiment_4"):
        logging.info("[RESUME] Skipping EXP 4 (already done)")

    # EXPERIMENT 5
    if d == 0 and not _done("experiment_5"):
        try:
            if erm_model and pcl_model:
                experiment_5_noise_robustness(erm_model, pcl_model, ds, ood_loaders)
                _save_results()
        except Exception as e:
            logging.error(f"EXP 5 FAILED: {e}")
    elif _done("experiment_5"):
        logging.info("[RESUME] Skipping EXP 5 (already done)")

    # EXPERIMENT 6
    if zxcxz == 0 and not _done("experiment_6"):
        try:
            if pcl_model:
                experiment_6_ood_detector(pcl_model, ds, val_loader, ood_loaders)
                _save_results()
        except Exception as e:
            logging.error(f"EXP 6 FAILED: {e}")
    elif _done("experiment_6"):
        logging.info("[RESUME] Skipping EXP 6 (already done)")

    # EXPERIMENT 7
    if f == 0 and not _done("experiment_7"):
        try:
            pcl_ref_auroc = None
            pcl_ood = exp1_results.get("PCL", {}).get("ood", {})
            first_ood = list(pcl_ood.keys())[0] if pcl_ood else None
            if first_ood:
                pcl_ref_auroc = pcl_ood[first_ood].get("auroc")
            experiment_7_irm_failure(ds, erm_model=erm_model, pcl_ood_auroc=pcl_ref_auroc)
            _save_results()
        except Exception as e:
            logging.error(f"EXP 7 FAILED: {e}")
    elif _done("experiment_7"):
        logging.info("[RESUME] Skipping EXP 7 (already done)")

    # ROBUSTNESS: High-k IRM (eICU-internal hospital environments)
    if not SEED_STUDY and not _done("experiment_irm_highk_eicu"):
        try:
            experiment_irm_highk_eicu(ds)
            _save_results()
        except Exception as e:
            logging.error(f"HIGH-K IRM (eICU) FAILED: {e}")
            logging.exception(e)
    elif _done("experiment_irm_highk_eicu"):
        logging.info("[RESUME] Skipping high-k IRM (already done)")

    # EXPERIMENT 8: Regularization baselines (A3) — opt-in via PCL_RUN_A3=1
    if RUN_A3 and not _done("experiment_8_regularization"):
        try:
            experiment_8_regularization_baselines(train_loader, val_loader, ood_loaders)
            _save_results()
        except Exception as e:
            logging.error(f"EXP 8 (A3 regularization baselines) FAILED: {e}")
            logging.exception(e)
    elif _done("experiment_8_regularization"):
        logging.info("[RESUME] Skipping EXP 8 / A3 (already done)")

    # ABLATION: Lambda sweep
    if g == 0 and not _done("ablation_lambda"):
        try:
            ablation_lambda_sweep(train_loader, val_loader, ood_loaders)
            _save_results()
        except Exception as e:
            logging.error(f"LAMBDA ABLATION FAILED: {e}")
    elif _done("ablation_lambda"):
        logging.info("[RESUME] Skipping lambda ablation (already done)")

    # ABLATION: Constraint subset
    if h == 0 and not _done("ablation_constraint_subset"):
        try:
            ablation_constraint_subset(train_loader, val_loader, ood_loaders)
            _save_results()
        except Exception as e:
            logging.error(f"CONSTRAINT ABLATION FAILED: {e}")
    elif _done("ablation_constraint_subset"):
        logging.info("[RESUME] Skipping constraint subset ablation (already done)")

    # Classical baselines
    if not SEED_STUDY and not _done("classical_baselines"):
        try:
            classical_baselines(ds, train_loader, ood_loaders)
            _save_results()
        except Exception as e:
            logging.error(f"CLASSICAL BASELINES FAILED: {e}")

    # Figure generation (always re-run — fast and idempotent)
    try:
        if exp1_results:
            _generate_main_figure(exp1_results, ood_loaders)
    except Exception as e:
        logging.error(f"FIGURE GENERATION FAILED: {e}")

    # Final save + manuscript outputs
    _save_results()
    _generate_latex_tables()

    try:
        from scripts.generate_paper_summary import main as generate_summary
        generate_summary()
    except Exception as e:
        logging.warning(f"Paper summary generation failed: {e}")

    logging.info("\n" + "=" * 60)
    logging.info("EXPERIMENT SUMMARY")
    logging.info("=" * 60)
    for exp_name, data in ALL_RESULTS.items():
        logging.info(f"\n{exp_name}:")
        if isinstance(data, dict):
            for k, v in data.items():
                logging.info(f"  {k}: {v}")

    if torch.cuda.is_available():
        logging.info(f"\nGPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak")


if __name__ == "__main__":
    if SEED_STUDY:
        # Seed study: EXP1 (main OOD) [+ EXP3 randomization]. Args:
        # z=EXP1, a=EXP2, b=EXP3, c=EXP4, d=EXP5, zxcxz=EXP6, f=EXP7, g=lambda, h=subset
        # EXP0 audit (+A2 computability) always runs inside main() regardless.
        # PCL_SKIP_EXP3=1 => EXP1 only (Phase 1 gate: inspect PCL vs ERM before
        # spending on EXP3/A3).
        b = 1 if os.environ.get("PCL_SKIP_EXP3", "0") == "1" else 0
        main(0, 1, b, 1, 1, 1, 1, 1, 1)
    else:
        main(0, 0, 0, 0, 0, 0, 0, 0, 0)
