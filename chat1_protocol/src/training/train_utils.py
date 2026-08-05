import logging
logging.basicConfig(level=logging.INFO)

import torch
import numpy as np

from config import LAMBDA_PCL, MASK_PROB, USE_AMP


# ── PCL LOSS BALANCER ────────────────────────────────────────────────────────

class PCLLossBalancer:
    """Normalizes L_PCL relative to L_masked via EMA ratio tracking.

    target_ratio controls how large the balanced PCL loss should be relative
    to L_masked.  With target_ratio=1.0 and λ=1.0:
        L_total = L_masked + 1.0 * L_masked = 2.0 * L_masked
    giving PCL equal footing with reconstruction.  Each of the 4 active
    constraints contributes ~25% of L_masked, enough to meaningfully shape
    encoder representations.  target_ratio=0.5 was found too conservative —
    constraint gradients were smaller than reconstruction gradients and the
    encoder learned ERM-equivalent representations.
    """

    def __init__(self, momentum=0.95, target_ratio=0.5):
        self.momentum = momentum
        self.target_ratio = target_ratio
        self.ratio_ema = None

    def balance(self, L_pcl, L_masked):
        pcl_val = L_pcl.item()
        masked_val = L_masked.item()
        if torch.isnan(L_pcl) or torch.isnan(L_masked):
            return L_pcl
        if pcl_val < 1e-6 or masked_val < 1e-6:
            return L_pcl
        
        # Scale so that balanced_pcl ≈ target_ratio * L_masked
        ratio = (self.target_ratio * masked_val) / pcl_val
        # Cap at 20× — keeps PCL meaningful when constraint positions are sparse
        # (30% masking × sparse lab co-occurrence). Gradient clipping (max_norm=1.0
        # in pretrain_step) handles any per-step magnitude spikes. 5× was too
        # conservative: PCL barely influenced encoder relative to masked prediction.
        ratio = min(ratio, 20.0)
        
        if self.ratio_ema is None:
            self.ratio_ema = ratio
        else:
            self.ratio_ema = self.momentum * self.ratio_ema + (1 - self.momentum) * ratio
        return L_pcl * self.ratio_ema


# ── MASKED PREDICTION LOSS ───────────────────────────────────────────────────

def masked_prediction_loss(preds, x_original, pretrain_mask):
    """
    MSE only on the positions that were masked.

    preds         : (B, T, V) — model output
    x_original    : (B, T, V) — original (unmasked) normalized values
    pretrain_mask : (B, T, V) bool — True at positions that were masked
    """
    if pretrain_mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device)
    diff = (preds - x_original)[pretrain_mask]
    return (diff ** 2).mean()


# ── TRAINING STEP ─────────────────────────────────────────────────────────────

def pretrain_step(batch, model, pcl_loss_fn, optimizer, lam=LAMBDA_PCL, mask_prob=MASK_PROB,
                   device="cpu", scaler=None, site_adversary=None, lam_site=0.5,
                   pcl_balancer=None):
    """
    Single pretraining step combining masked prediction + PCL + optional site adversarial.

    Returns dict of scalar loss values for logging.
    """
    from src.models.backbone import apply_random_mask

    model.train()
    optimizer.zero_grad()

    x = batch["x"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    c_mask = batch["c_mask"].to(device, non_blocking=True)

    x_masked, pretrain_mask = apply_random_mask(x, mask, mask_prob=mask_prob)

    with torch.amp.autocast("cuda", enabled=USE_AMP):
        reps = model.encode(x_masked, mask)
        preds = model.predict(reps)
        L_masked = masked_prediction_loss(preds, x, pretrain_mask)

    # The physiological constraints denormalize to physical units — PaO2**3
    # (up to ~700^3 ≈ 3.4e8), denormalized MAP, and Henderson-Hasselbalch log10 —
    # whose magnitudes overflow float16 (max ~65504). Under AMP that produces
    # inf/NaN and corrupts the model within one epoch. So L_PCL, the balancer, and
    # the loss combination are ALWAYS computed in float32 (autocast disabled).
    with torch.amp.autocast("cuda", enabled=False):
        L_masked = L_masked.float()
        pcl_out = pcl_loss_fn(preds.float(), c_mask, pretrain_mask=pretrain_mask)
        L_pcl = pcl_out["L_PCL"]
        L_pcl_balanced = pcl_balancer.balance(L_pcl, L_masked) if pcl_balancer is not None else L_pcl
        L_total = L_masked + lam * L_pcl_balanced

        # CRITICAL: If loss is NaN, skip this step to protect model weights
        if torch.isnan(L_total):
            optimizer.zero_grad()
            return {k: 0.0 for k in ["L_total", "L_masked", "L_pcl", "L_site", "L_MAP", "L_PP", "L_SI", "L_HH", "L_SpO2"]}

        L_site_val = 0.0
        if site_adversary is not None and "site_id" in batch:
            site_labels = batch["site_id"].to(device, non_blocking=True).long()
            site_logits = site_adversary(reps.float(), mask, alpha=lam_site)
            L_site = torch.nn.functional.cross_entropy(site_logits, site_labels)
            L_total = L_total + lam_site * L_site
            L_site_val = L_site.item()

    if scaler is not None:
        scaler.scale(L_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {
        "L_total": L_total.item(),
        "L_masked": L_masked.item(),
        "L_pcl": L_pcl.item(),
        "L_site": L_site_val,
        "L_MAP": pcl_out["losses"]["MAP"].item(),
        "L_PP": pcl_out["losses"]["PP"].item(),
        "L_SI": pcl_out["losses"]["SI"].item(),
        "L_HH": pcl_out["losses"]["HH"].item(),
        "L_SpO2": pcl_out["losses"]["SpO2"].item(),
    }


# ── VALIDATION STEP ───────────────────────────────────────────────────────────

def pretrain_val_step(batch, model, pcl_loss_fn, lam=LAMBDA_PCL, mask_prob=MASK_PROB, device="cpu"):
    """Same as pretrain_step but no gradient updates."""
    from src.models.backbone import apply_random_mask

    model.eval()
    with torch.no_grad():
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        c_mask = batch["c_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            x_masked, pretrain_mask = apply_random_mask(x, mask, mask_prob=mask_prob)
            reps = model.encode(x_masked, mask)
            preds = model.predict(reps)
            L_masked = masked_prediction_loss(preds, x, pretrain_mask)
        # L_PCL in float32 (see pretrain_step) — float16 overflow → NaN otherwise.
        with torch.amp.autocast("cuda", enabled=False):
            pcl_out = pcl_loss_fn(preds.float(), c_mask, pretrain_mask=pretrain_mask)
            L_pcl = pcl_out["L_PCL"]
            L_total = L_masked.float() + lam * L_pcl

    return {
        "L_total": L_total.item(),
        "L_masked": L_masked.item(),
        "L_pcl": L_pcl.item(),
    }


# ── PRETRAINING LOOP ──────────────────────────────────────────────────────────

def run_pretraining(
    model,
    pcl_loss_fn,
    train_loader,
    val_loader,
    n_epochs = 20,
    lr = 3e-4,
    lam = LAMBDA_PCL,
    mask_prob = MASK_PROB,
    device = "cpu",
    save_path = "pcl_pretrained.pt",
    log_every = 1,
    warmup_fraction = 0.3,
    use_site_adversary = False,
    lam_site = 0.1,
):
    """
    Full pretraining loop with constraint warmup.
    λ linearly ramps from 0 to target over first warmup_fraction of epochs.
    Returns history dict with loss curves.
    """
    model = model.to(device)

    site_adversary = None
    if use_site_adversary and lam > 0:
        from src.models.backbone import SiteAdversary
        site_adversary = SiteAdversary(model.d_model, n_sites=4).to(device)

    all_params = list(model.parameters())
    if site_adversary is not None:
        all_params += list(site_adversary.parameters())

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.1
    )

    pcl_balancer = PCLLossBalancer(target_ratio=1.0)

    # Guarantee at least 2 warmup epochs so that short runs (TEST_MODE)
    # still get a meaningful ramp.  With only 1 warmup epoch the full λ
    # hits on step 1, before the model learns basic reconstruction.
    warmup_epochs = max(2, int(n_epochs * warmup_fraction))

    history = {
        "train_total": [], "train_masked": [], "train_pcl": [],
        "val_total": [], "val_masked": [], "val_pcl": [],
        "train_MAP": [], "train_PP": [], "train_SI": [], "train_HH": [], "train_SpO2": [],
        "effective_lam": [], "balance_ratio": [],
    }

    # Checkpoint on L_masked only — mirrors ERM criterion and ensures we pick the best
    # reconstructive encoder, not the one that merely minimises the PCL penalty.
    best_val_masked = float("inf")

    logging.info(f"Starting pretraining — {n_epochs} epochs, λ={lam} (warmup over {warmup_epochs} epochs), device={device}")
    logging.info(f"{'Epoch':<8} {'Train Total':<14} {'Train Masked':<14} {'Train PCL':<12} {'Val Total':<12} {'λ_eff':<10} {'LR'}")
    logging.info("─" * 85)

    for epoch in range(1, n_epochs + 1):

        if epoch <= warmup_epochs:
            lam_eff = lam * (epoch / warmup_epochs)
        else:
            lam_eff = lam

        # ── Train ──
        train_metrics = {"L_total": [], "L_masked": [], "L_pcl": [], "L_site": [],
                         "L_MAP": [], "L_PP": [], "L_SI": [], "L_HH": [], "L_SpO2": []}
        for batch in train_loader:
            step = pretrain_step(batch, model, pcl_loss_fn, optimizer,
                                 lam=lam_eff, mask_prob=mask_prob, device=device,
                                 scaler=scaler, site_adversary=site_adversary, lam_site=lam_site,
                                 pcl_balancer=pcl_balancer)
            for k in train_metrics:
                train_metrics[k].append(step[k])

        # ── Validate ──
        val_metrics = {"L_total": [], "L_masked": [], "L_pcl": []}
        for batch in val_loader:
            step = pretrain_val_step(batch, model, pcl_loss_fn,
                                     lam=lam, mask_prob=mask_prob, device=device)
            for k in val_metrics:
                val_metrics[k].append(step[k])

        scheduler.step()

        # -- Aggregate --
        t_total = np.mean(train_metrics["L_total"]) if train_metrics["L_total"] else np.nan
        t_masked = np.mean(train_metrics["L_masked"]) if train_metrics["L_masked"] else np.nan
        t_pcl = np.mean(train_metrics["L_pcl"]) if train_metrics["L_pcl"] else np.nan
        v_total = np.mean(val_metrics["L_total"]) if val_metrics["L_total"] else np.nan
        v_masked = np.mean(val_metrics["L_masked"]) if val_metrics["L_masked"] else np.nan
        v_pcl = np.mean(val_metrics["L_pcl"]) if val_metrics["L_pcl"] else np.nan
        
        if np.isnan(t_total):
            logging.warning(f"Epoch {epoch}: Train loss is NaN. This usually means the DataLoader is empty (check BATCH_SIZE) or the model exploded.")
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_total"].append(t_total)
        history["train_masked"].append(t_masked)
        history["train_pcl"].append(t_pcl)
        history["val_total"].append(v_total)
        history["val_masked"].append(v_masked)
        history["val_pcl"].append(v_pcl)
        history["effective_lam"].append(lam_eff)
        history["balance_ratio"].append(pcl_balancer.ratio_ema if pcl_balancer.ratio_ema else 1.0)

        for cname in ["MAP", "PP", "SI", "HH", "SpO2"]:
            key = f"train_{cname}"
            history[key].append(np.nanmean(train_metrics[f"L_{cname}"]) if train_metrics[f"L_{cname}"] else np.nan)

        if epoch % log_every == 0:
            logging.info(f"{epoch:<8} {t_total:<14.6f} {t_masked:<14.6f} {t_pcl:<12.6f} {v_total:<12.6f} {lam_eff:<10.4f} {cur_lr:.2e}")

            if epoch % 5 == 0 and train_metrics['L_MAP']:
                logging.info(f" Constraints → MAP:{np.nanmean(train_metrics['L_MAP']):.5f} "
                      f"PP:{np.nanmean(train_metrics['L_PP']):.5f} "
                      f"SI:{np.nanmean(train_metrics['L_SI']):.5f} "
                      f"HH:{np.nanmean(train_metrics['L_HH']):.5f} "
                      f"SpO2:{np.nanmean(train_metrics['L_SpO2']):.5f}")

        if v_masked < best_val_masked:
            best_val_masked = v_masked
            save_checkpoint(model, optimizer, epoch, history, save_path)

    logging.info(f"\nPretraining complete. Best val masked loss: {best_val_masked:.6f}")
    logging.info(f"Checkpoint saved to: {save_path}")
    return history




# ── CHECKPOINT UTILS ──────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, history, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
    }, path)


def load_checkpoint(model, path, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    logging.info(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    return ckpt["history"]


def load_state_dict_flexible(path, device="cpu"):
    """
    Loads either a raw `state_dict` (ERM pretraining) or a training checkpoint dict
    with key `model_state` (PCL pretraining via save_checkpoint).
    """
    o = torch.load(path, map_location=device, weights_only=False)
    if isinstance(o, dict) and "model_state" in o:
        return o["model_state"]
    return o




# ── LOSS CURVE PLOT ───────────────────────────────────────────────────────────

def plot_loss_curves(history, save_path=None):
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_total"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_total"], label="Train")
    axes[0].plot(epochs, history["val_total"], label="Val")
    axes[0].set_title("Total Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, history["train_masked"], label="Train")
    axes[1].plot(epochs, history["val_masked"], label="Val")
    axes[1].set_title("Masked Prediction Loss"); axes[1].legend(); axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, history["train_pcl"], label="Train")
    axes[2].plot(epochs, history["val_pcl"], label="Val")
    axes[2].set_title("PCL Loss"); axes[2].legend(); axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_constraint_curves(history, save_path=None):
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_total"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for cname in ["MAP", "PP", "SI", "HH", "SpO2"]:
        key = f"train_{cname}"
        if key in history and history[key]:
            axes[0].plot(epochs, history[key], label=cname)
    axes[0].set_title("Per-Constraint Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].set_yscale("log")

    if "effective_lam" in history and history["effective_lam"]:
        axes[1].plot(epochs, history["effective_lam"], "k-", linewidth=2)
        axes[1].set_title("Effective λ (Warmup Schedule)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("λ")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()