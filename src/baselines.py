import logging
logging.basicConfig(level=logging.INFO)
"""
baselines.py
Baseline model implementations for comparison against PCL.

Baselines:
  1. ERM — standard training, no PCL (lambda=0)
  2. FE — ERM + computed MAP/PP/SI as explicit input features
  3. IRM-unit — Invariant Risk Minimization with care unit environments (high k)
  4. IRM-hosp — IRM with hospital-level environments (low k, failure regime)
  5. DRO — Distributionally Robust Optimization over groups

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import copy

from config import USE_AMP


# ════════════════════════════════════════════════════════════════════════════
# 1. ERM BASELINE
# ════════════════════════════════════════════════════════════════════════════

def run_erm_pretraining(
    model,
    train_loader,
    val_loader,
    n_epochs = 20,
    lr = 3e-4,
    mask_prob = None,
    device = "cpu",
    save_path = "erm_pretrained.pt",
    input_noise = 0.0,
    weight_decay = 1e-4,
):
    """
    Standard ERM pretraining — masked prediction loss only, no PCL.
    Identical to PCL pretraining but with lambda=0.

    This is your primary baseline. Everything else is compared against this.

    A3 regularization baselines (professor action item A3) reuse this loop:
      * input_noise > 0 adds zero-mean Gaussian noise (std=input_noise, in
        normalized [0,1] space) to the OBSERVED CONTEXT positions the encoder
        sees — i.e. positions that were NOT masked out. The reconstruction
        target stays the clean value, so the model must denoise. Corrupting the
        context (not just the already-masked targets) is what makes a denoising
        autoencoder / noise-augmentation baseline genuinely distinct from the
        plain masked-prediction ERM/PCL objective.
      * weight_decay is exposed so the "stronger weight decay" baseline can be
        swept without editing this function.
    Defaults (input_noise=0.0, weight_decay=1e-4) reproduce the original ERM
    behavior exactly.
    """
    from src.training.train_utils import masked_prediction_loss, save_checkpoint
    from src.models.backbone import apply_random_mask
    from config import MASK_PROB
    if mask_prob is None:
        mask_prob = MASK_PROB

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    logging.info(f"ERM Pretraining — {n_epochs} epochs | device={device}")
    logging.info(f"{'Epoch':<8} {'Train Loss':<14} {'Val Loss'}")
    logging.info("─" * 40)

    for epoch in range(1, n_epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            x_masked, pretrain_mask = apply_random_mask(x, mask, mask_prob)
            if input_noise > 0.0:
                # Corrupt the observed CONTEXT (observed AND not-masked); the
                # target `x` stays clean → denoising signal. Masked positions are
                # already 0 and are the prediction targets, so we leave them.
                context = mask & (~pretrain_mask)
                noise = torch.randn_like(x_masked) * input_noise
                x_masked = x_masked + noise * context.to(x_masked.dtype)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                reps = model.encode(x_masked, mask)
                preds = model.predict(reps)
                loss = masked_prediction_loss(preds, x, pretrain_mask)

            if torch.isnan(loss):
                logging.warning(f" [ERM Epoch {epoch}] NaN loss detected. Skipping batch.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        # ── Val ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                x_masked, pretrain_mask = apply_random_mask(x, mask, mask_prob)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    reps = model.encode(x_masked, mask)
                    preds = model.predict(reps)
                    loss = masked_prediction_loss(preds, x, pretrain_mask)
                val_losses.append(loss.item())

        scheduler.step()
        t = np.mean(train_losses)
        v = np.mean(val_losses)
        history["train_loss"].append(t)
        history["val_loss"].append(v)
        logging.info(f"{epoch:<8} {t:<14.6f} {v:.6f}")

        if v < best_val:
            best_val = v
            torch.save(model.state_dict(), save_path)

    logging.info(f"\nERM pretraining complete. Best val loss: {best_val:.6f}")
    return history


# ════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING BASELINE
# ════════════════════════════════════════════════════════════════════════════

def add_engineered_features(dataset):
    """
    Adds computed MAP, PP, and SI as explicit extra input columns to each sample.
    Modifies dataset in-place. n_vars goes from 9 → 12.

    This is the FE baseline — gives the model the physics as features.
    The FE Demolition Ablation will show PCL >> FE on OOD.

    New columns appended:
      index 9 : MAP_computed = DBP + (SBP - DBP) / 3
      index 10 : PP_computed = SBP - DBP
      index 11 : SI_computed = HR / SBP
    """
    from src.data_utils import VARIABLES, PLAUS

    idx = {v: i for i, v in enumerate(VARIABLES)}
    n_added = 0

    for sample in dataset.samples:
        x = sample["x"]
        mask = sample["mask"]

        T = x.shape[0]
        fe_cols = torch.zeros(T, 3)
        fe_masks = torch.zeros(T, 3, dtype=torch.bool)

        SBP = x[:, idx["SBP"]] * (300 - 40) + 40
        DBP = x[:, idx["DBP"]] * (200 - 20) + 20
        HR = x[:, idx["HR"]] * (300 - 20) + 20

        m_sbp = mask[:, idx["SBP"]]
        m_dbp = mask[:, idx["DBP"]]
        m_hr = mask[:, idx["HR"]]

        map_valid = m_sbp & m_dbp
        map_raw = DBP + (SBP - DBP) / 3.0
        map_norm = (map_raw - 20) / (200 - 20)
        fe_cols[:, 0] = torch.where(map_valid, map_norm, torch.zeros(T))
        fe_masks[:, 0] = map_valid

        pp_valid = m_sbp & m_dbp
        pp_raw = SBP - DBP
        pp_norm = torch.clamp(pp_raw / 200.0, 0, 1)
        fe_cols[:, 1] = torch.where(pp_valid, pp_norm, torch.zeros(T))
        fe_masks[:, 1] = pp_valid

        si_valid = m_hr & m_sbp
        si_raw = HR / (SBP + 1e-6)
        si_norm = torch.clamp(si_raw / 5.0, 0, 1)
        fe_cols[:, 2] = torch.where(si_valid, si_norm, torch.zeros(T))
        fe_masks[:, 2] = si_valid

        sample["x"] = torch.cat([x, fe_cols], dim=-1)
        sample["mask"] = torch.cat([mask, fe_masks], dim=-1)
        n_added += 1

    logging.info(f"Feature engineering applied to {n_added} stays")
    logging.info(f"New x shape: {dataset.samples[0]['x'].shape}  (was (48, 9), now (48, 12))")
    logging.info(f"Added columns: MAP_computed (idx 9), PP_computed (idx 10), SI_computed (idx 11)")
    return dataset




# ════════════════════════════════════════════════════════════════════════════
# 3. IRM BASELINE
# ════════════════════════════════════════════════════════════════════════════

def irm_penalty(logits, labels):
    """
    Computes the IRM penalty for one environment.
    Penalty = ||grad_{w|w=1} (loss over env)||^2
    where w is a scalar weight on the logits.

    Following Arjovsky et al. 2019 / the official IRM implementation.

    FIX: The scale dummy variable must be initialised to 1.0 (not -1.0 or 0.0).
    A sign-flip here inverts the gradient direction and causes the penalty to
    push the model *away* from invariance, producing AUROC < 0.5 (worse than
    random). The gradient is squared so the sign of the penalty itself is
    always non-negative — the bug manifests in the ERM loss term direction.
    """
    scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits.detach() * scale, labels.float())
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)




class IRMFinetuner:
    """
    Fine-tunes a model using IRM on a binary classification task.
    Requires environment labels (sample["env_id"]) in dataset.

    IRM loss = ERM_loss + lambda_irm * sum_e(IRM_penalty_e)
    """
    def __init__(self, model, task_name, lambda_irm=1e3, lr=1e-4, device="cpu"):
        self.model = model.to(device)
        self.task_name = task_name
        self.lambda_irm = lambda_irm
        self.device = device

        if task_name not in model.cls_heads:
            model.add_classification_head(task_name)
        self.model = self.model.to(device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-4
        )

    def train_epoch(self, loader):
        self.model.train()
        epoch_losses = []

        for batch in loader:
            self.optimizer.zero_grad()

            x = batch["x"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch[self.task_name].to(self.device)
            envs = batch["env_id"].to(self.device)

            reps = self.model.encode(x, mask)
            logits = self.model.classify(
                reps, self.task_name, obs_mask=mask
            ).squeeze(-1)

            erm_loss = F.binary_cross_entropy_with_logits(logits, labels)

            penalty = torch.tensor(0.0, device=self.device)
            unique_envs = envs.unique()
            for env_id in unique_envs:
                env_mask = (envs == env_id)
                if env_mask.sum() < 2:
                    continue
                env_logits = logits[env_mask]
                env_labels = labels[env_mask]
                if len(env_labels.unique()) < 2:
                    continue
                penalty = penalty + irm_penalty(env_logits, env_labels)

            loss = erm_loss + self.lambda_irm * penalty
            
            if torch.isnan(loss):
                logging.warning(" [IRM] NaN loss detected. Skipping batch.")
                self.optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            epoch_losses.append(erm_loss.item())

        return np.mean(epoch_losses)

    def run(self, train_loader, val_loader, n_epochs=20, save_path=None):
        history = {"train_loss": [], "val_auroc": []}
        best_auroc = 0.0

        logging.info(f"\nIRM Fine-tuning: {self.task_name} | λ_IRM={self.lambda_irm}")
        logging.info(f"{'Epoch':<8} {'Train Loss':<14} {'Val AUROC'}")
        logging.info("─" * 38)

        for epoch in range(1, n_epochs + 1):
            t_loss = self.train_epoch(train_loader)

            self.model.eval()
            probs, labels_all = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(self.device)
                    mask = batch["mask"].to(self.device)
                    lbl = batch[self.task_name].cpu().numpy()
                    reps = self.model.encode(x, mask)
                    logits = self.model.classify(
                        reps, self.task_name, obs_mask=mask
                    ).squeeze(-1)
                    probs.extend(torch.sigmoid(logits).cpu().numpy())
                    labels_all.extend(lbl)

            labels_arr = np.array(labels_all)
            probs_arr = np.array(probs)
            history["train_loss"].append(t_loss)

            if len(np.unique(labels_arr)) < 2:
                logging.info(f"{epoch:<8} {t_loss:<14.6f} N/A (1 class)")
                continue

            auroc = roc_auc_score(labels_arr, probs_arr)
            history["val_auroc"].append(auroc)
            logging.info(f"{epoch:<8} {t_loss:<14.6f} {auroc:.6f}")

            if auroc > best_auroc:
                best_auroc = auroc
                if save_path:
                    torch.save(self.model.state_dict(), save_path)

        logging.info(f"\nBest IRM val AUROC: {best_auroc:.6f}")
        return history


# ════════════════════════════════════════════════════════════════════════════
# 4. DRO BASELINE
# ════════════════════════════════════════════════════════════════════════════

class DROFinetuner:
    """
    Group DRO — minimizes the worst-group loss.
    Groups defined by env_id (same as IRM environments).
    Requires sample["env_id"] in dataset.

    DRO loss = max_g { ERM_loss_g } (soft version via exponential reweighting)
    """
    def __init__(self, model, task_name, n_groups, eta=0.01, lr=1e-4, device="cpu"):
        self.model = model.to(device)
        self.task_name = task_name
        self.n_groups = n_groups
        self.eta = eta
        self.device = device

        if task_name not in model.cls_heads:
            model.add_classification_head(task_name)
        self.model = self.model.to(device)

        self.group_weights = torch.ones(n_groups) / n_groups

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-4
        )

    def train_epoch(self, loader):
        self.model.train()
        epoch_losses = []

        for batch in loader:
            self.optimizer.zero_grad()

            x = batch["x"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch[self.task_name].to(self.device)
            envs = batch["env_id"]

            reps = self.model.encode(x, mask)
            logits = self.model.classify(
                reps, self.task_name, obs_mask=mask
            ).squeeze(-1)

            per_sample_loss = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )
            group_losses = torch.zeros(self.n_groups, device=self.device)
            group_counts = torch.zeros(self.n_groups, device=self.device)

            for g in range(self.n_groups):
                g_mask = (envs == g)
                if g_mask.sum() > 0:
                    group_losses[g] = per_sample_loss[g_mask].mean()
                    group_counts[g] = g_mask.sum().float()

            w = self.group_weights.to(self.device)
            w = w * torch.exp(self.eta * group_losses.detach())
            w = w / (w.sum() + 1e-8)
            self.group_weights = w.cpu()
            loss = (w * group_losses).sum()

            if torch.isnan(loss):
                logging.warning(" [DRO] NaN loss detected. Skipping batch.")
                self.optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            epoch_losses.append(loss.item())

        return np.mean(epoch_losses)

    def run(self, train_loader, val_loader, n_epochs=20, save_path=None):
        history = {"train_loss": [], "val_auroc": []}
        best_auroc = 0.0

        logging.info(f"\nDRO Fine-tuning: {self.task_name} | n_groups={self.n_groups}")
        logging.info(f"{'Epoch':<8} {'Train Loss':<14} {'Val AUROC'}")
        logging.info("─" * 38)

        for epoch in range(1, n_epochs + 1):
            t_loss = self.train_epoch(train_loader)

            self.model.eval()
            probs, labels_all = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(self.device)
                    mask = batch["mask"].to(self.device)
                    lbl = batch[self.task_name].cpu().numpy()
                    reps = self.model.encode(x, mask)
                    logits = self.model.classify(
                        reps, self.task_name, obs_mask=mask
                    ).squeeze(-1)
                    probs.extend(torch.sigmoid(logits).cpu().numpy())
                    labels_all.extend(lbl)

            labels_arr = np.array(labels_all)
            probs_arr = np.array(probs)
            history["train_loss"].append(t_loss)

            if len(np.unique(labels_arr)) < 2:
                logging.info(f"{epoch:<8} {t_loss:<14.6f} N/A (1 class)")
                continue

            auroc = roc_auc_score(labels_arr, probs_arr)
            history["val_auroc"].append(auroc)
            logging.info(f"{epoch:<8} {t_loss:<14.6f} {auroc:.6f}")

            if auroc > best_auroc:
                best_auroc = auroc
                if save_path:
                    torch.save(self.model.state_dict(), save_path)

        logging.info(f"\nBest DRO val AUROC: {best_auroc:.6f}")
        return history




# ════════════════════════════════════════════════════════════════════════════
# 6. FRESH MODEL FACTORY
# ════════════════════════════════════════════════════════════════════════════

def fresh_model(n_vars=None, seed=42, dropout=None):
    from src.models.backbone import PCLModel
    from config import D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, N_VARS, DROPOUT
    if n_vars is None:
        n_vars = N_VARS
    if dropout is None:
        dropout = DROPOUT
    torch.manual_seed(seed)
    return PCLModel(
        n_vars=n_vars, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=dropout,
    )


# ════════════════════════════════════════════════════════════════════════════
# 7. CLASSICAL ML BASELINES (Logistic Regression + XGBoost)
# ════════════════════════════════════════════════════════════════════════════

def _aggregate_features(dataset, task_name="mortality"):
    """
    Aggregates each variable's time series into mean, min, max, and
    observation rate (fraction of timesteps observed).
    Returns (X, y) numpy arrays suitable for sklearn.

    This is the standard tabular representation used in clinical ML
    baselines — if the Transformer cannot beat this, the architecture
    is not justified.
    """
    from src.data_utils import VARIABLES
    X_rows, y_rows = [], []
    for sample in dataset.samples:
        x = sample["x"].numpy()       # (T, V)
        mask = sample["mask"].numpy()  # (T, V)
        row = []
        for v in range(x.shape[1]):
            obs = x[mask[:, v], v]
            if len(obs) == 0:
                row.extend([0.0, 0.0, 0.0, 0.0])  # mean, min, max, obs_rate
            else:
                row.extend([obs.mean(), obs.min(), obs.max(), mask[:, v].mean()])
        X_rows.append(row)
        label = sample.get(task_name)
        y_rows.append(float(label.item()) if label is not None else 0.0)
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


def run_classical_baselines(
    train_dataset,
    test_dataset,
    task_name="mortality",
    seed=42,
):
    """
    Trains Logistic Regression and (optionally) XGBoost on aggregated
    statistics (mean/min/max per variable) and evaluates on test_dataset.

    Returns dict of {model_name: {"auroc": float, "auprc": float, "brier": float}}

    These results should be included in the paper's results table to
    justify the Transformer architecture. If PCL does not outperform
    LR on AUROC, the architecture choice needs re-examination.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_train, y_train = _aggregate_features(train_dataset, task_name)
    X_test, y_test = _aggregate_features(test_dataset, task_name)

    if len(np.unique(y_test)) < 2:
        logging.warning(
            "Classical baselines: test set has only one class — "
            "AUROC undefined. Use a larger or stratified test split."
        )
        return {}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # ── Logistic Regression ──
    lr_clf = LogisticRegression(max_iter=2000, random_state=seed, C=1.0, class_weight="balanced")
    lr_clf.fit(X_train_s, y_train)
    lr_probs = lr_clf.predict_proba(X_test_s)[:, 1]
    results["LR (agg features)"] = {
        "auroc": roc_auc_score(y_test, lr_probs),
        "auprc": average_precision_score(y_test, lr_probs),
        "brier": brier_score_loss(y_test, lr_probs),
    }
    logging.info(
        f"LR baseline — AUROC: {results['LR (agg features)']['auroc']:.4f} | "
        f"AUPRC: {results['LR (agg features)']['auprc']:.4f} | "
        f"Brier: {results['LR (agg features)']['brier']:.4f}"
    )

    # ── XGBoost (optional — skipped gracefully if not installed) ──
    try:
        from xgboost import XGBClassifier
        xgb_clf = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )
        xgb_clf.fit(X_train_s, y_train)
        xgb_probs = xgb_clf.predict_proba(X_test_s)[:, 1]
        results["XGBoost (agg features)"] = {
            "auroc": roc_auc_score(y_test, xgb_probs),
            "auprc": average_precision_score(y_test, xgb_probs),
            "brier": brier_score_loss(y_test, xgb_probs),
        }
        logging.info(
            f"XGBoost baseline — AUROC: {results['XGBoost (agg features)']['auroc']:.4f} | "
            f"AUPRC: {results['XGBoost (agg features)']['auprc']:.4f} | "
            f"Brier: {results['XGBoost (agg features)']['brier']:.4f}"
        )
    except ImportError:
        logging.info("XGBoost not installed — skipping. Install with: pip install xgboost")

    return results