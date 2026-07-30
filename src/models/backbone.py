import logging
logging.basicConfig(level=logging.INFO)

import torch
import torch.nn as nn
import math

from config import D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, N_VARS, DROPOUT, HOURS


# ── INPUT EMBEDDING ───────────────────────────────────────────────────────────

class ClinicalEmbedding(nn.Module):
    """
    Projects each (timestep, variable) pair into d_model space.
    Adds learned variable embeddings + sinusoidal positional embeddings.

    Input : x    (B, T, V) — normalized time series
            mask (B, T, V) — True where observed
    Output: (B, T, d_model)
    """
    def __init__(self, n_vars: int, d_model: int, max_len: int = 48):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model

        self.input_proj = nn.Linear(n_vars, d_model)
        self.var_embed = nn.Embedding(n_vars + 1, d_model, padding_idx=0)
        self.obs_embed = nn.Embedding(2, d_model)
        self.register_buffer("pos_enc", self._sinusoidal(max_len, d_model))
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def _sinusoidal(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x, mask):
        B, T, V = x.shape
        out = self.input_proj(x)
        out = out + self.pos_enc[:, :T, :]

        obs_flag = mask.long()
        var_ids = torch.arange(1, V + 1, device=x.device)
        var_ids = var_ids.unsqueeze(0).unsqueeze(0).expand(B, T, V)
        var_ids = var_ids * obs_flag
        v_emb = self.var_embed(var_ids).sum(dim=2)
        out = out + v_emb

        out = self.layer_norm(out)
        out = self.dropout(out)
        return out


# ── TRANSFORMER ENCODER ───────────────────────────────────────────────────────

class ClinicalTransformer(nn.Module):
    """
    6-layer transformer encoder over hourly ICU time series.

    Input : x      (B, T, V)
            mask   (B, T, V) — observation mask
    Output: representations (B, T, d_model)
    """
    def __init__(
        self,
        n_vars: int = N_VARS,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        ffn_dim: int = FFN_DIM,
        max_len: int = HOURS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embedding = ClinicalEmbedding(n_vars, d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(self, x, mask):
        emb = self.embedding(x, mask)
        # Create causal mask (triangular)
        sz = emb.size(1)
        causal_mask = torch.triu(torch.full((sz, sz), float('-inf'), device=x.device), diagonal=1)
        reps = self.encoder(emb, mask=causal_mask, is_causal=True)
        return reps

    def encode(self, x, mask):
        """Alias so ClinicalTransformer can be used standalone."""
        return self.forward(x, mask)


# ── MASKED PREDICTION HEAD (pretraining) ─────────────────────────────────────

class MaskedPredictionHead(nn.Module):
    """
    Predicts masked variable values from transformer representations.

    Input : reps (B, T, d_model)
    Output: preds (B, T, V)
    """
    def __init__(self, d_model: int, n_vars: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_vars),
        )

    def forward(self, reps):
        return self.head(reps)


# ── CLASSIFICATION HEAD (fine-tuning) ─────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Extracts the transformer representation at the LAST observed timestep
    for classification. This ensures zero look-ahead bias.

    Input : reps (B, T, d_model)
            mask (B, T, V) — used to find the last valid timestep
    Output: logits (B, 1)
    """
    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, reps, obs_mask=None):
        if obs_mask is not None:
            # Find index of last observed timestep for each sample in batch
            # obs_mask is (B, T, V), t_obs is (B, T)
            t_obs = obs_mask.any(dim=-1)
            # Flip along time axis; argmax finds first True in flipped = last True in original.
            # sum()-1 was wrong: counts observed timesteps, not the index of the last one.
            last_idx = (t_obs.size(1) - 1 - t_obs.flip(dims=[1]).long().argmax(dim=1)).clamp(min=0)
            
            # Gather representations at last_idx
            # pooled shape: (B, d_model)
            batch_idx = torch.arange(reps.size(0), device=reps.device)
            pooled = reps[batch_idx, last_idx]
        else:
            # Fallback to last timestep if no mask provided
            pooled = reps[:, -1, :]
        return self.head(pooled)


# ── FULL PCL MODEL ────────────────────────────────────────────────────────────

class PCLModel(nn.Module):
    """
    Full model used during pretraining and fine-tuning.
    Combines encoder + masked prediction head.
    The PCL loss is applied externally (in PhysiologicalConstraintLoss).

    API:
        model   = PCLModel()
        reps    = model.encode(x, mask)              # get representations
        reps    = model.get_representations(x, mask) # alias for encode
        preds   = model.predict(reps)                # for masked prediction loss
        logits  = model.classify(reps, task, mask)   # for fine-tuning
    """
    def __init__(
        self,
        n_vars: int = N_VARS,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        ffn_dim: int = FFN_DIM,
        max_len: int = HOURS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.encoder = ClinicalTransformer(
            n_vars=n_vars, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, ffn_dim=ffn_dim, max_len=max_len, dropout=dropout
        )
        self.pred_head = MaskedPredictionHead(d_model, n_vars)
        self.cls_heads = nn.ModuleDict()
        self.d_model = d_model

    def encode(self, x, mask):
        """Returns (B, T, d_model) representations. Primary API method."""
        return self.encoder(x, mask)

    def get_representations(self, x, mask):
        """Alias for encode() — kept for evaluate_utils compatibility."""
        return self.encode(x, mask)

    def predict(self, reps):
        """Returns (B, T, V) predictions for masked variable prediction loss."""
        return self.pred_head(reps)

    def add_classification_head(self, task_name: str, dropout: float = 0.2):
        """
        Adds a classification head for a downstream task.
        Queries the model's current device and moves the new head there
        immediately so mixed-device crashes cannot occur.
        """
        device = next(self.parameters()).device
        self.cls_heads[task_name] = ClassificationHead(self.d_model, dropout).to(device)

    def classify(self, reps, task_name: str, obs_mask=None):
        """Returns (B, 1) logits for a given task."""
        if task_name not in self.cls_heads:
            raise KeyError(
                f"No classification head for task '{task_name}'. "
                "Call model.add_classification_head(task_name) first."
            )
        return self.cls_heads[task_name](reps, obs_mask)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Trainable parameters: {total:,} ({total/1e6:.1f}M)")
        return total


# ── GRADIENT REVERSAL (site adversarial) ─────────────────────────────────────

class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class SiteAdversary(nn.Module):
    """Small MLP with gradient reversal for site-invariant pretraining."""
    def __init__(self, d_model, n_sites=4):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, n_sites),
        )

    def forward(self, reps, obs_mask, alpha=1.0):
        t_obs = obs_mask.any(dim=-1)
        last_idx = (t_obs.size(1) - 1 - t_obs.flip(dims=[1]).long().argmax(dim=1)).clamp(min=0)
        batch_idx = torch.arange(reps.size(0), device=reps.device)
        pooled = reps[batch_idx, last_idx]
        reversed_pooled = _GradientReversal.apply(pooled, alpha)
        return self.clf(reversed_pooled)


# ── MASKING UTILITY ───────────────────────────────────────────────────────────

def apply_random_mask(x: torch.Tensor, obs_mask: torch.Tensor, mask_prob: float = 0.30):
    """
    Randomly masks mask_prob fraction (default 30%) of observed (variable, timestep) pairs.

    Returns:
        x_masked     : (B, T, V) — masked input (0 at masked positions)
        pretrain_mask: (B, T, V) bool — True at positions that were masked
    """
    rand = torch.rand_like(x)
    pretrain_mask = (rand < mask_prob) & obs_mask
    x_masked = x.clone()
    x_masked[pretrain_mask] = 0.0
    return x_masked, pretrain_mask
