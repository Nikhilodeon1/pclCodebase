import torch
import torch.nn as nn

try:
    from src.data.variables import CANONICAL_VARIABLES as VARIABLES
except ImportError:
    from src.data_utils import VARIABLES


def severinghaus_sao2_fraction(PaO2_mmhg: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable Severinghaus-style oxyhemoglobin dissociation.
    SaO2 = inner / (inner + 23400)  where  inner = PaO2^3 + 150*PaO2  (PaO2 in mmHg).
    Returns saturation in [0, 1].
    """
    p = torch.clamp(PaO2_mmhg, min=1.0)
    inner = p ** 3 + 150.0 * p
    return inner / (inner + 23400.0 + eps)


class PhysiologicalConstraintLoss(nn.Module):
    """
    Five physiological constraints applied to model predictions (all normalized [0,1]):

      0  MAP equation consistency        MAP = DBP + (SBP-DBP)/3
      1  Pulse pressure consistency      PP_direct == PP_via_MAP
      2  Shock index equality            MSE(HR/SBP, 0.7 reference)  [audit only, si_weight=0.0]
      3  Henderson-Hasselbalch           pH = 6.1 + log10(HCO3 / 0.0307*pCO2)
      4  SpO2-PaO2 Severinghaus          SpO2 ~ f(PaO2)

    Clinical note on constraint 3 (Henderson-Hasselbalch):
      Mechanical ventilation and bicarbonate infusion decouple the natural
      acid-base equilibrium. Pass hh_weight < 1.0 (e.g. 0.1-0.2) for patients
      on a ventilator or bicarbonate drip to avoid penalising iatrogenic
      decoupling of natural acid-base chemistry.
    """

    def __init__(self, eps: float = 1e-8, si_weight: float = 0.0, pp_weight: float = 0.0):
        super().__init__()
        self.eps = eps
        self.si_weight = si_weight
        # PP is algebraically equivalent to MAP (MAP=DBP+(SBP-DBP)/3 ↔ PP=3*(MAP-DBP)).
        # Including both double-penalizes BP variables relative to HH and SpO2.
        # Keep computed for audit/ablation; excluded from L_PCL by default (pp_weight=0.0).
        self.pp_weight = pp_weight
        self.idx = {v: i for i, v in enumerate(VARIABLES)}

    def _get(self, x: torch.Tensor, var: str) -> torch.Tensor:
        return x[:, :, self.idx[var]]

    def _mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return ((pred - target)[mask] ** 2).mean()

    def _l1(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return torch.abs((pred - target)[mask]).mean()

    def _constraint_mask(self, c_mask_k, pretrain_mask, var_indices):
        """Returns positions where the constraint gradient is informative.

        Restricts to timesteps where ≥1 constraint-relevant variable was masked.
        At those positions the model cannot copy the input — it must impute using
        cross-variable physiological relationships, making L_PCL a real invariance
        signal that shapes the encoder.

        At non-masked positions: masked_prediction_loss is silent (no reconstruction
        pressure), so preds there are unconstrained. The pred_head can satisfy the
        constraint trivially without the encoder learning cross-variable structure.
        Including those positions dilutes the gradient and causes PCL < ERM.
        """
        if pretrain_mask is None:
            return c_mask_k
        any_var_masked = torch.zeros_like(c_mask_k)
        for vi in var_indices:
            any_var_masked = any_var_masked | pretrain_mask[:, :, vi]
        return c_mask_k & any_var_masked

    def forward(
        self,
        x: torch.Tensor,
        c_mask: torch.Tensor,
        pretrain_mask: torch.Tensor = None,
        hh_weight: float = 1.0,
    ) -> dict:
        """
        Args:
            x             : (B, T, V) model predictions, normalized to [0, 1]
            c_mask        : (B, T, 5) bool — True where each constraint is computable
            pretrain_mask : (B, T, V) bool — True at masked positions (optional).
                            When provided, constraints only apply at positions where
                            at least one relevant variable was masked, preventing
                            trivial satisfaction by input copying.
            hh_weight     : scalar or (B,) tensor — downweight HH for ventilated patients
        Returns:
            dict with keys "L_PCL", "losses" (per-constraint), "active" (counts)
        """
        idx = self.idx
        SBP  = self._get(x, "SBP")
        DBP  = self._get(x, "DBP")
        MAP  = self._get(x, "MAP")
        HR   = self._get(x, "HR")
        SpO2 = self._get(x, "SpO2")
        pH   = self._get(x, "pH")
        HCO3 = self._get(x, "HCO3")
        pCO2 = self._get(x, "pCO2")
        PaO2 = self._get(x, "PaO2")

        losses: dict = {}
        active: dict = {}

        SBP_d = SBP * (300 - 40) + 40
        DBP_d = DBP * (200 - 20) + 20
        MAP_d = MAP * (200 - 20) + 20
        HR_d  = HR  * (300 - 20) + 20

        # ── 0: MAP equation consistency (L1) ─────────────────────────────────
        m0 = self._constraint_mask(c_mask[:, :, 0], pretrain_mask,
                                   [idx["SBP"], idx["DBP"], idx["MAP"]])
        predicted_map     = DBP_d + (SBP_d - DBP_d) / 3.0
        predicted_map_norm = (predicted_map - 20) / (200 - 20 + self.eps)
        losses["MAP"] = self._l1(MAP, predicted_map_norm, m0)
        active["MAP"] = m0.sum().item()

        # ── 1: Pulse pressure consistency ────────────────────────────────────
        m1 = self._constraint_mask(c_mask[:, :, 1], pretrain_mask,
                                   [idx["SBP"], idx["DBP"], idx["MAP"]])
        PP_direct  = SBP_d - DBP_d
        PP_via_map = 3.0 * (MAP_d - DBP_d)
        losses["PP"] = self._mse(PP_direct / 260.0, PP_via_map / 260.0, m1)
        active["PP"] = m1.sum().item()

        # ── 2: Shock index equality (MSE on HR/SBP ratio) ───────────────────
        # Converted from hinge [0.3, 2.0] to MSE for gradient consistency with
        # other equality constraints. Reference SI=0.7 (normal resting midpoint).
        # Excluded from L_PCL by default (si_weight=0.0); retained for audit logging.
        m2 = self._constraint_mask(c_mask[:, :, 2], pretrain_mask,
                                   [idx["HR"], idx["SBP"]])
        SBP_floor = torch.clamp(SBP_d, min=40.0)
        pred_si = HR_d / (SBP_floor + self.eps)
        target_si = torch.full_like(pred_si, 0.7)
        losses["SI"] = self._mse(pred_si / 3.0, target_si / 3.0, m2)
        active["SI"] = m2.sum().item()

        # ── 3: Henderson-Hasselbalch acid-base equilibrium ────────────────────
        m3 = self._constraint_mask(c_mask[:, :, 3], pretrain_mask,
                                   [idx["pH"], idx["HCO3"], idx["pCO2"]])
        HCO3_raw = HCO3 * (60 - 5) + 5
        pCO2_raw = pCO2 * (150 - 10) + 10

        # Stability: clamp inputs to log to avoid log(0) or log(neg)
        H_conc      = HCO3_raw / (0.0307 * pCO2_raw + self.eps)
        H_conc_safe = torch.clamp(H_conc, min=0.01, max=1000.0) 
        ph_pred     = 6.1 + torch.log10(H_conc_safe)

        ph_pred_norm = (ph_pred - 6.5) / (7.9 - 6.5 + self.eps)
        ph_pred_norm_clamped = torch.clamp(ph_pred_norm, 0.0, 1.0)

        hh_raw_norm = self._mse(pH, ph_pred_norm_clamped, m3)
        if isinstance(hh_weight, torch.Tensor):
            hh_w = hh_weight.mean()
        else:
            hh_w = float(hh_weight)
        losses["HH"] = hh_raw_norm * hh_w
        active["HH"] = m3.sum().item()

        # ── 4: SpO2-PaO2 Severinghaus dissociation ───────────────────────────
        m4 = self._constraint_mask(c_mask[:, :, 4], pretrain_mask,
                                   [idx["SpO2"], idx["PaO2"]])
        PaO2_raw     = torch.clamp(PaO2 * (700 - 20) + 20, min=1.0)
        sao2         = severinghaus_sao2_fraction(PaO2_raw, eps=self.eps)
        spo2_pred_pct  = sao2 * 100.0
        spo2_pred_norm = torch.clamp((spo2_pred_pct - 50.0) / 50.0, 0.0, 1.0)
        losses["SpO2"] = self._mse(SpO2, spo2_pred_norm, m4)
        active["SpO2"] = m4.sum().item()

        # ── Total PCL loss ────────────────────────────────────────────────────
        # SI excluded by default (si_weight=0.0): ablation showed it hurt OOD performance.
        # PP excluded by default (pp_weight=0.0): algebraically redundant with MAP —
        # both enforce MAP=DBP+(SBP-DBP)/3, doubling BP gradient vs HH/SpO2.
        # Both retained for audit logging and ablation via SubsetConstraintLoss.
        L_PCL = (
            losses["MAP"] + self.pp_weight * losses["PP"] + self.si_weight * losses["SI"]
            + losses["HH"] + losses["SpO2"]
        )
        
        # Ensure L_PCL is a valid scalar even if no constraints were active in this batch
        if torch.isnan(L_PCL):
            L_PCL = torch.tensor(0.0, device=x.device, requires_grad=True)

        return {"L_PCL": L_PCL, "losses": losses, "active": active}

PCLoss = PhysiologicalConstraintLoss
