import logging
logging.basicConfig(level=logging.INFO)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import copy

import os
from config import USE_AMP


# ── PER-SAMPLE PREDICTION DUMP ────────────────────────────────────────────────
# When a directory is registered here, every evaluate_model() call also writes
# its raw per-sample probs + labels to disk. This preserves the data needed for
# post-hoc analysis (recomputing CIs with different settings, paired PCL-vs-ERM
# significance / DeLong tests, calibration) which cannot be recovered after the
# run. A monotonic counter keeps filenames collision-free across experiments.
_PRED_DUMP = {"dir": None, "n": 0}


def set_prediction_dump(directory):
    if directory:
        os.makedirs(directory, exist_ok=True)
    _PRED_DUMP["dir"] = directory
    _PRED_DUMP["n"] = 0


def _dump_predictions(probs, labels, task_name, split_name):
    d = _PRED_DUMP["dir"]
    if not d:
        return
    _PRED_DUMP["n"] += 1
    safe = "".join(c if c.isalnum() else "_" for c in f"{task_name}__{split_name}")
    path = os.path.join(d, f"{_PRED_DUMP['n']:03d}__{safe}.npz")
    np.savez(path, probs=np.asarray(probs), labels=np.asarray(labels),
             task=task_name, split=split_name)


def bootstrap_ci(labels, probs, metric_fn, n_bootstrap=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = labels[idx], probs[idx]
        if len(np.unique(y_b)) < 2:
            continue
        scores.append(metric_fn(y_b, p_b))
    if not scores:
        return None, None
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return lo, hi


# ── FINE-TUNING LOOP ──────────────────────────────────────────────────────────

def run_finetuning(
    model,
    train_loader,
    val_loader,
    task_name,
    n_epochs=20,
    lr=1e-3,
    device="cpu",
    save_path=None,
    input_noise=0.0,
    mixup_alpha=0.0,
    weight_decay=1e-4,
):
    """
    Two-phase fine-tuning:
      Phase 1 (epochs 1 .. freeze_epochs): encoder frozen, head trains at lr.
      Phase 2 (remaining epochs): encoder unfrozen, discriminative LR
                                  (encoder at lr*0.1, head at lr).

    Key fixes vs previous version:
      - lr default raised 1e-4 → 1e-3. A randomly-initialised head on a frozen
        encoder needs a higher LR to escape the flat loss region quickly.
      - The optimizer is rebuilt at unfreeze time and explicitly lists ALL
        trainable param groups. The old filter(requires_grad) snapshot taken
        before the loop silently excluded encoder params after unfreezing.
      - pred_head is always included in the optimizer so its params are never
        orphaned when the encoder unfreezes.
      - best_state is seeded from initial weights so load_state_dict never
        receives None when AUROC never improves (single-class val set).
    """
    model = model.to(device)

    if task_name not in model.cls_heads:
        model.add_classification_head(task_name)
        logging.info(f"Added classification head for task: {task_name}")
    model = model.to(device)

    best_state = copy.deepcopy(model.state_dict())
    last_state = best_state  # updated every epoch; fallback when AUROC never improves
    best_val_auroc = 0.0

    # ── Phase 1: freeze encoder, train head only ──────────────────────────────
    freeze_epochs = max(1, n_epochs // 3)
    for param in model.encoder.parameters():
        param.requires_grad = False

    def _head_params():
        return (
            list(model.cls_heads[task_name].parameters())
            + list(model.pred_head.parameters())
        )

    optimizer = torch.optim.AdamW(_head_params(), lr=lr, weight_decay=weight_decay)
    encoder_unfrozen = False

    history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_auprc": [], "val_brier": []}

    logging.info(f"\nFine-tuning: {task_name} | {n_epochs} epochs | lr={lr} | device={device}")
    logging.info(f"  Phase 1 (frozen encoder): epochs 1–{freeze_epochs}")
    logging.info(f"  Phase 2 (full model):     epochs {freeze_epochs+1}–{n_epochs}")
    logging.info(f"{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<12} {'Val AUROC':<12} {'Val AUPRC':<12} {'Val Brier'}")
    logging.info("─" * 75)

    # Calculate pos_weight from the entire dataset to avoid shuffle/drop_last non-determinism
    all_train_labels = []
    if hasattr(train_loader.dataset, "indices"):
        # Subset dataset
        ds = train_loader.dataset.dataset
        idxs = train_loader.dataset.indices
        all_train_labels = [ds.samples[i][task_name].item() for i in idxs]
    else:
        # Full dataset
        all_train_labels = [s[task_name].item() for s in train_loader.dataset.samples]
        
    num_pos = float(np.sum(all_train_labels))
    num_neg = float(len(all_train_labels) - num_pos)
    # Safety: avoid division by zero and cap extreme weights
    pos_weight_val = num_neg / (num_pos + 1e-6)
    pos_weight_val = min(pos_weight_val, 100.0)
    pos_weight = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)
    logging.info(f"  pos_weight = {pos_weight.item():.2f}")

    ft_scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    for epoch in range(1, n_epochs + 1):

        # ── Phase 2 transition: unfreeze encoder ─────────────────────────────
        if epoch == freeze_epochs + 1 and not encoder_unfrozen:
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.encoder.parameters(),          "lr": lr * 0.1},
                    {"params": model.cls_heads[task_name].parameters(), "lr": lr},
                    {"params": model.pred_head.parameters(),        "lr": lr * 0.1},
                ],
                weight_decay=weight_decay,
            )
            encoder_unfrozen = True
            logging.info(f" [Epoch {epoch}] Encoder unfrozen — discriminative LR (encoder={lr*0.1:.1e}, head={lr:.1e})")

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            x      = batch["x"].to(device, non_blocking=True)
            mask   = batch["mask"].to(device, non_blocking=True)
            labels = batch[task_name].to(device, non_blocking=True)

            # A3 baselines (professor action item A3), train-time only:
            #   input_noise: Gaussian noise on observed inputs (noise-augmentation).
            #   mixup_alpha: convex interpolation of inputs and (soft) labels.
            # Defaults 0.0 => original ERM/PCL fine-tuning, unchanged.
            target = labels.float()
            if input_noise > 0.0:
                x = x + torch.randn_like(x) * input_noise * mask.to(x.dtype)
            if mixup_alpha > 0.0 and x.size(0) > 1:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(x.size(0), device=x.device)
                x = lam * x + (1.0 - lam) * x[perm]
                mask = mask | mask[perm]
                target = lam * target + (1.0 - lam) * target[perm]

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                reps   = model.encode(x, mask)
                logits = model.classify(reps, task_name, obs_mask=mask).squeeze(-1)
                loss   = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

            # CRITICAL: Skip step if loss is NaN to protect weights
            if torch.isnan(loss):
                logging.warning(f" [Epoch {epoch}] NaN loss detected in finetuning batch. Skipping update.")
                optimizer.zero_grad()
                continue

            ft_scaler.scale(loss).backward()
            ft_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            ft_scaler.step(optimizer)
            ft_scaler.update()
            train_losses.append(loss.item())

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_losses, val_probs, val_labels_all = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x      = batch["x"].to(device, non_blocking=True)
                mask   = batch["mask"].to(device, non_blocking=True)
                labels = batch[task_name].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    reps   = model.encode(x, mask)
                    logits = model.classify(reps, task_name, obs_mask=mask).squeeze(-1)
                    loss   = F.binary_cross_entropy_with_logits(logits, labels.float())
                val_losses.append(loss.item())
                val_probs.extend(torch.sigmoid(logits).float().cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())

        t_loss = np.mean(train_losses)
        v_loss = np.mean(val_losses)
        last_state = copy.deepcopy(model.state_dict())
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        val_labels_arr = np.array(val_labels_all)
        val_probs_arr  = np.array(val_probs)

        if len(np.unique(val_labels_arr)) < 2:
            logging.info(f"{epoch:<8} {t_loss:<14.6f} {v_loss:<12.6f} N/A (single class in val — use stratified split)")
        else:
            # Filter out any NaNs that would crash sklearn
            valid_mask = ~np.isnan(val_probs_arr)
            if not np.any(valid_mask):
                logging.warning(f" [Epoch {epoch}] ALL val predictions were NaN.")
                continue

            v_probs = val_probs_arr[valid_mask]
            v_labels = val_labels_arr[valid_mask]

            auroc = roc_auc_score(v_labels, v_probs)
            auprc = average_precision_score(v_labels, v_probs)
            brier = brier_score_loss(v_labels, v_probs)
            history["val_auroc"].append(auroc)
            history["val_auprc"].append(auprc)
            history["val_brier"].append(brier)
            logging.info(
                f"{epoch:<8} {t_loss:<14.6f} {v_loss:<12.6f} "
                f"{auroc:<12.6f} {auprc:<12.6f} {brier:.6f}"
            )
            if auroc > best_val_auroc:
                best_val_auroc = auroc
                best_state = copy.deepcopy(model.state_dict())
                if save_path:
                    torch.save(best_state, save_path)

    if best_val_auroc > 0.0:
        model.load_state_dict(best_state)
    else:
        # AUROC was never computable (single-class val) — use last epoch, not initial weights
        model.load_state_dict(last_state)
        if save_path:
            torch.save(last_state, save_path)
        logging.info(" [Fine-tuning] AUROC never improved (single-class val) — using last epoch state.")
    logging.info(f"\nBest val AUROC: {best_val_auroc:.6f}")
    if save_path and best_val_auroc > 0.0:
        logging.info(f"Best model saved to: {save_path}")
    return history


# ── FULL EVALUATION ───────────────────────────────────────────────────────────

def evaluate_model(model, loader, task_name, device="cpu", split_name="Val"):
    """
    Computes AUROC, AUPRC, and Brier Score on a given loader with bootstrap CIs.
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch[task_name].cpu().numpy()

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                reps = model.encode(x, mask)
                logits = model.classify(reps, task_name, obs_mask=mask).squeeze(-1)
            probs = torch.sigmoid(logits).float().cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Persist raw per-sample predictions before any aggregation (no-op unless a
    # dump dir was registered via set_prediction_dump).
    _dump_predictions(all_probs, all_labels, task_name, split_name)

    num_pos = int(np.sum(all_labels))
    num_neg = len(all_labels) - num_pos

    if len(np.unique(all_labels)) < 2:
        logging.info(
            f" [{split_name}] Warning: Only one class present "
            f"(Pos:{num_pos}, Neg:{num_neg}) — cannot compute AUROC"
        )
        return {"auroc": None, "auprc": None, "brier": None,
                "auroc_ci": (None, None), "auprc_ci": (None, None)}

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    brier = brier_score_loss(all_labels, all_probs)

    auroc_ci = bootstrap_ci(all_labels, all_probs, roc_auc_score)
    auprc_ci = bootstrap_ci(all_labels, all_probs, average_precision_score)

    ci_str = ""
    if auroc_ci[0] is not None:
        ci_str = f" [CI: {auroc_ci[0]:.3f}-{auroc_ci[1]:.3f}]"
    logging.info(
        f" [{split_name}] AUROC: {auroc:.4f}{ci_str} | AUPRC: {auprc:.4f} | "
        f"Brier: {brier:.4f} | Dist: {num_pos}pos/{num_neg}neg"
    )
    return {"auroc": auroc, "auprc": auprc, "brier": brier,
            "auroc_ci": auroc_ci, "auprc_ci": auprc_ci}




# ── LINEAR PROBING ────────────────────────────────────────────────────────────

def extract_representations(model, loader, device="cpu"):
    """
    Extracts mean-pooled frozen representations from the encoder.
    Returns (N, d_model) numpy array and corresponding stay_ids.
    """
    model.eval()
    all_reps, all_ids = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            sids = batch["stay_id"]

            reps = model.encode(x, mask)
            t_obs = mask.any(dim=-1)
            last_idx = (t_obs.size(1) - 1 - t_obs.flip(dims=[1]).long().argmax(dim=1)).clamp(min=0)
            
            batch_idx = torch.arange(reps.size(0), device=reps.device)
            pooled = reps[batch_idx, last_idx]

            all_reps.extend(pooled.cpu().numpy())
            all_ids.extend(sids.numpy() if hasattr(sids, 'numpy') else sids)

    return np.array(all_reps), np.array(all_ids)


def run_linear_probe(model, dataset, probe_target, device="cpu", test_frac=0.3, seed=42):
    """
    Trains a linear classifier on frozen representations to predict probe_target.
    Handles both binary and multi-class targets (e.g. site_id with 4 classes).
    Returns probe AUROC — near 0.5 (binary) or 1/K (multi-class) means
    representations don't encode that attribute.
    """
    full_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    reps, ids = extract_representations(model, full_loader, device=device)
    labels = np.array([s[probe_target].item() for s in dataset.samples])

    n_classes = len(np.unique(labels))
    if n_classes < 2:
        logging.info(f" Probe '{probe_target}': only one class present — skipping")
        return None

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(sss.split(reps, labels))

    X_train, y_train = reps[train_idx], labels[train_idx]
    X_test, y_test = reps[test_idx], labels[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    clf.fit(X_train, y_train)

    if n_classes == 2:
        preds = clf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, preds)
        if auroc < 0.5:
            auroc = 1.0 - auroc
    else:
        preds = clf.predict_proba(X_test)
        n_test_classes = len(np.unique(y_test))
        if n_test_classes < 2:
            logging.info(f" Probe '{probe_target}': test set has only 1 class — skipping")
            return None
        auroc = roc_auc_score(y_test, preds, multi_class="ovr", average="macro")

    accuracy = clf.score(X_test, y_test)
    logging.info(f" Linear probe '{probe_target}': AUROC = {auroc:.4f} | Acc = {accuracy:.4f} ({n_classes} classes)")
    return auroc




# ── PCL VIOLATION AS OOD DETECTOR ────────────────────────────────────────────

def compute_ood_scores(model, pcl_loss_fn, loader, device="cpu"):
    """
    Computes per-sample PCL violation scores at test time (no gradient updates).
    High score = model's physiological assumptions are violated = potential OOD.

    Returns array of (N,) violation scores and corresponding stay_ids.
    """
    model.eval()
    all_scores, all_ids = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            c_mask = batch["c_mask"].to(device)

            reps = model.encode(x, mask)
            preds = model.predict(reps)

            B = x.shape[0]
            for i in range(B):
                sample_preds = preds[i:i+1]
                sample_cmask = c_mask[i:i+1]
                sample_out = pcl_loss_fn(sample_preds, sample_cmask)
                # Normalize by number of active constraints to get a stable 0-1 score
                valid_losses = [v.item() for k, v in sample_out["losses"].items() if sample_out["active"][k] > 0 and k != "SI"]
                score = np.mean(valid_losses) if valid_losses else 0.0
                all_scores.append(score)

            sids = batch["stay_id"]
            all_ids.extend(sids.numpy() if hasattr(sids, 'numpy') else sids)

    return np.array(all_scores), np.array(all_ids)


