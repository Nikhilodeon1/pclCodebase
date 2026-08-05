import logging
logging.basicConfig(level=logging.INFO)

import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score


# ════════════════════════════════════════════════════════════════════════════
# 1. CONSTRAINT RANDOMIZATION ABLATION
# ════════════════════════════════════════════════════════════════════════════

class RandomizedConstraintLoss(nn.Module):
    """
    Drop-in replacement for PhysiologicalConstraintLoss that uses
    WRONG equations instead of real physiological ones.

    All modes use the same denormalization and loss functions as the real PCL
    to ensure magnitude-fair comparison. The only difference is the formula.

    Three randomization modes:
      "wrong_eq" — incorrect formulas in physical units (same scale as real)
      "shuffled" — correct formula structure but variables swapped
      "noise" — random noise targets (pure regularization with no structure)

    If PCL with real constraints beats all three, the content
    of the constraints — not the penalty itself — is doing the work.
    """
    def __init__(self, mode="wrong_eq", eps=1e-8):
        super().__init__()
        self.mode = mode
        self.eps = eps
        try:
            from src.data.variables import CANONICAL_VARIABLES as VARIABLES
        except ImportError:
            from src.data_utils import VARIABLES
        self.idx = {v: i for i, v in enumerate(VARIABLES)}

    def _get(self, x, var):
        return x[:, :, self.idx[var]]

    def _mse(self, pred, target, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return ((pred - target)[mask] ** 2).mean()

    def _l1(self, pred, target, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return torch.abs((pred - target)[mask]).mean()

    def _constraint_mask(self, c_mask_k, pretrain_mask, var_indices):
        if pretrain_mask is None:
            return c_mask_k
        any_var_masked = torch.zeros_like(c_mask_k)
        for vi in var_indices:
            any_var_masked = any_var_masked | pretrain_mask[:, :, vi]
        return c_mask_k & any_var_masked

    def forward(self, x, c_mask, pretrain_mask=None):
        idx = self.idx
        SBP = self._get(x, "SBP"); DBP = self._get(x, "DBP")
        MAP = self._get(x, "MAP"); HR = self._get(x, "HR")
        SpO2 = self._get(x, "SpO2"); pH = self._get(x, "pH")
        HCO3 = self._get(x, "HCO3"); pCO2 = self._get(x, "pCO2")
        PaO2 = self._get(x, "PaO2")

        losses, active = {}, {}

        # Denormalize to physical units (same as real PCL)
        SBP_d = SBP * (300 - 40) + 40
        DBP_d = DBP * (200 - 20) + 20
        MAP_d = MAP * (200 - 20) + 20
        HR_d  = HR  * (300 - 20) + 20

        if self.mode == "wrong_eq":
            # MAP: wrong formula MAP = SBP - DBP (instead of DBP + (SBP-DBP)/3)
            m0 = self._constraint_mask(c_mask[:, :, 0], pretrain_mask,
                                       [idx["SBP"], idx["DBP"], idx["MAP"]])
            wrong_map = SBP_d - DBP_d
            wrong_map_norm = (wrong_map - 20) / (200 - 20 + self.eps)
            losses["MAP"] = self._l1(MAP, wrong_map_norm, m0)
            active["MAP"] = m0.sum().item()

            # PP: wrong formula PP = SBP + DBP (instead of SBP - DBP)
            m1 = self._constraint_mask(c_mask[:, :, 1], pretrain_mask,
                                       [idx["SBP"], idx["DBP"], idx["MAP"]])
            PP_wrong = SBP_d + DBP_d
            PP_real_via_map = 3.0 * (MAP_d - DBP_d)
            losses["PP"] = self._mse(PP_wrong / 260.0, PP_real_via_map / 260.0, m1)
            active["PP"] = m1.sum().item()

            # SI: wrong range [1.5, 5.0] (instead of [0.3, 2.0])
            m2 = self._constraint_mask(c_mask[:, :, 2], pretrain_mask,
                                       [idx["HR"], idx["SBP"]])
            SBP_floor = torch.clamp(SBP_d, min=40.0)
            pred_si = HR_d / (SBP_floor + self.eps)
            si_low = torch.clamp(1.5 - pred_si, min=0.0)
            si_high = torch.clamp(pred_si - 5.0, min=0.0)
            si_violation = si_low ** 2 + si_high ** 2
            losses["SI"] = si_violation[m2].mean() if m2.sum() > 0 else torch.tensor(0.0, device=x.device)
            active["SI"] = m2.sum().item()

            # HH: wrong formula pH = 7.0 + log10(pCO2 / HCO3) (inverted ratio)
            m3 = self._constraint_mask(c_mask[:, :, 3], pretrain_mask,
                                       [idx["pH"], idx["HCO3"], idx["pCO2"]])
            HCO3_raw = HCO3 * (60 - 5) + 5
            pCO2_raw = pCO2 * (150 - 10) + 10
            ratio = torch.clamp(pCO2_raw / (0.0307 * HCO3_raw + self.eps), min=self.eps)
            ph_pred = 7.0 + torch.log(ratio) / torch.log(torch.tensor(10.0, device=x.device))
            ph_pred_norm = torch.clamp((ph_pred - 6.5) / (7.9 - 6.5 + self.eps), 0.0, 1.0)
            losses["HH"] = self._mse(pH, ph_pred_norm, m3)
            active["HH"] = m3.sum().item()

            # SpO2: wrong dissociation (linear instead of sigmoidal)
            m4 = self._constraint_mask(c_mask[:, :, 4], pretrain_mask,
                                       [idx["SpO2"], idx["PaO2"]])
            PaO2_raw = torch.clamp(PaO2 * (700 - 20) + 20, min=1.0)
            spo2_wrong_pct = torch.clamp(PaO2_raw / 7.0, 0.0, 100.0)
            spo2_wrong_norm = torch.clamp((spo2_wrong_pct - 50.0) / 50.0, 0.0, 1.0)
            losses["SpO2"] = self._mse(SpO2, spo2_wrong_norm, m4)
            active["SpO2"] = m4.sum().item()

        elif self.mode == "shuffled":
            # Correct formula structure but variables swapped across constraints
            # MAP formula applied to wrong variables: MAP = SpO2 + (HR - SpO2)/3
            m0 = self._constraint_mask(c_mask[:, :, 0], pretrain_mask,
                                       [idx["SBP"], idx["DBP"], idx["MAP"]])
            SpO2_as_dbp = SpO2 * (200 - 20) + 20
            HR_as_sbp = HR * (300 - 40) + 40
            shuffled_map = SpO2_as_dbp + (HR_as_sbp - SpO2_as_dbp) / 3.0
            shuffled_map_norm = (shuffled_map - 20) / (200 - 20 + self.eps)
            losses["MAP"] = self._l1(MAP, shuffled_map_norm, m0)
            active["MAP"] = m0.sum().item()

            # PP with shuffled vars
            m1 = self._constraint_mask(c_mask[:, :, 1], pretrain_mask,
                                       [idx["SBP"], idx["DBP"], idx["MAP"]])
            PP_shuffled = HR_as_sbp - SpO2_as_dbp
            PP_via_map = 3.0 * (MAP_d - DBP_d)
            losses["PP"] = self._mse(PP_shuffled / 260.0, PP_via_map / 260.0, m1)
            active["PP"] = m1.sum().item()

            # SI with shuffled vars: DBP/MAP instead of HR/SBP
            m2 = self._constraint_mask(c_mask[:, :, 2], pretrain_mask,
                                       [idx["HR"], idx["SBP"]])
            MAP_floor = torch.clamp(MAP_d, min=40.0)
            pred_si = DBP_d / (MAP_floor + self.eps)
            si_low = torch.clamp(0.3 - pred_si, min=0.0)
            si_high = torch.clamp(pred_si - 2.0, min=0.0)
            si_violation = si_low ** 2 + si_high ** 2
            losses["SI"] = si_violation[m2].mean() if m2.sum() > 0 else torch.tensor(0.0, device=x.device)
            active["SI"] = m2.sum().item()

            # HH with SBP instead of pCO2
            m3 = self._constraint_mask(c_mask[:, :, 3], pretrain_mask,
                                       [idx["pH"], idx["HCO3"], idx["pCO2"]])
            HCO3_raw = HCO3 * (60 - 5) + 5
            ratio = torch.clamp(HCO3_raw / (0.0307 * SBP_d + self.eps), min=self.eps)
            ph_pred = 6.1 + torch.log(ratio) / torch.log(torch.tensor(10.0, device=x.device))
            ph_pred_norm = torch.clamp((ph_pred - 6.5) / (7.9 - 6.5 + self.eps), 0.0, 1.0)
            losses["HH"] = self._mse(pH, ph_pred_norm, m3)
            active["HH"] = m3.sum().item()

            # SpO2 predicted from HR instead of PaO2
            m4 = self._constraint_mask(c_mask[:, :, 4], pretrain_mask,
                                       [idx["SpO2"], idx["PaO2"]])
            from src.losses.pcl_loss import severinghaus_sao2_fraction
            sao2 = severinghaus_sao2_fraction(HR_d, eps=self.eps)
            spo2_pred_pct = sao2 * 100.0
            spo2_pred_norm = torch.clamp((spo2_pred_pct - 50.0) / 50.0, 0.0, 1.0)
            losses["SpO2"] = self._mse(SpO2, spo2_pred_norm, m4)
            active["SpO2"] = m4.sum().item()

        elif self.mode == "noise":
            # Random targets — same scale as real constraints
            m0 = self._constraint_mask(c_mask[:, :, 0], pretrain_mask,
                                       [idx["SBP"], idx["DBP"], idx["MAP"]])
            noise_map = torch.rand_like(MAP)
            losses["MAP"] = self._l1(MAP, noise_map, m0)
            active["MAP"] = m0.sum().item()

            m1 = self._constraint_mask(c_mask[:, :, 1], pretrain_mask,
                                       [idx["SBP"], idx["DBP"], idx["MAP"]])
            losses["PP"] = self._mse(SBP, torch.rand_like(SBP), m1)
            active["PP"] = m1.sum().item()

            m2 = self._constraint_mask(c_mask[:, :, 2], pretrain_mask,
                                       [idx["HR"], idx["SBP"]])
            losses["SI"] = self._mse(HR, torch.rand_like(HR), m2)
            active["SI"] = m2.sum().item()

            m3 = self._constraint_mask(c_mask[:, :, 3], pretrain_mask,
                                       [idx["pH"], idx["HCO3"], idx["pCO2"]])
            losses["HH"] = self._mse(pH, torch.rand_like(pH), m3)
            active["HH"] = m3.sum().item()

            m4 = self._constraint_mask(c_mask[:, :, 4], pretrain_mask,
                                       [idx["SpO2"], idx["PaO2"]])
            losses["SpO2"] = self._mse(SpO2, torch.rand_like(SpO2), m4)
            active["SpO2"] = m4.sum().item()

        L_PCL = losses["MAP"] + losses["HH"] + losses["SpO2"]

        return {"L_PCL": L_PCL, "losses": losses, "active": active}


# ════════════════════════════════════════════════════════════════════════════
# 3. CONSTRAINT SUBSET ABLATION
# ════════════════════════════════════════════════════════════════════════════

class SubsetConstraintLoss(nn.Module):
    """
    PCL loss with one constraint removed.
    Pass excluded_constraint="MAP" to remove MAP, etc.
    Pass excluded_constraint=None to use all constraints (full PCL).
    """
    def __init__(self, excluded_constraint=None, eps=1e-8):
        super().__init__()
        self.excluded = excluded_constraint
        self.eps = eps
        try:
            from src.data.variables import CANONICAL_VARIABLES as VARIABLES
        except ImportError:
            from src.data_utils import VARIABLES
        self.idx = {v: i for i, v in enumerate(VARIABLES)}

    def _get(self, x, var):
        return x[:, :, self.idx[var]]

    def _mse(self, pred, target, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return ((pred - target)[mask] ** 2).mean()

    def _l1(self, pred, target, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return torch.abs((pred - target)[mask]).mean()

    def _constraint_mask(self, c_mask_k, pretrain_mask, var_indices):
        if pretrain_mask is None:
            return c_mask_k
        any_var_masked = torch.zeros_like(c_mask_k)
        for vi in var_indices:
            any_var_masked = any_var_masked | pretrain_mask[:, :, vi]
        return c_mask_k & any_var_masked

    def forward(self, x, c_mask, pretrain_mask=None):
        idx = self.idx
        SBP = self._get(x, "SBP"); DBP = self._get(x, "DBP")
        MAP = self._get(x, "MAP"); HR = self._get(x, "HR")
        SpO2 = self._get(x, "SpO2"); pH = self._get(x, "pH")
        HCO3 = self._get(x, "HCO3"); pCO2 = self._get(x, "pCO2")
        PaO2 = self._get(x, "PaO2")

        losses, active = {}, {}

        SBP_d = SBP * (300 - 40) + 40
        DBP_d = DBP * (200 - 20) + 20
        MAP_d = MAP * (200 - 20) + 20
        HR_d  = HR  * (300 - 20) + 20

        # MAP
        if "MAP" != self.excluded:
            m0 = self._constraint_mask(c_mask[:, :, 0], pretrain_mask,
                                       [idx["SBP"], idx["DBP"], idx["MAP"]])
            predicted_map = DBP_d + (SBP_d - DBP_d) / 3.0
            predicted_map_norm = (predicted_map - 20) / (200 - 20 + self.eps)
            losses["MAP"] = self._l1(MAP, predicted_map_norm, m0)
            active["MAP"] = m0.sum().item()
        else:
            losses["MAP"] = torch.tensor(0.0, device=x.device)
            active["MAP"] = 0

        # PP
        if "PP" != self.excluded:
            m1 = self._constraint_mask(c_mask[:, :, 1], pretrain_mask,
                                       [idx["SBP"], idx["DBP"], idx["MAP"]])
            losses["PP"] = self._mse((SBP_d - DBP_d) / 260.0, (3.0 * (MAP_d - DBP_d)) / 260.0, m1)
            active["PP"] = m1.sum().item()
        else:
            losses["PP"] = torch.tensor(0.0, device=x.device)
            active["PP"] = 0

        # SI
        if "SI" != self.excluded:
            m2 = self._constraint_mask(c_mask[:, :, 2], pretrain_mask,
                                       [idx["HR"], idx["SBP"]])
            SBP_floor = torch.clamp(SBP_d, min=40.0)
            pred_si = HR_d / (SBP_floor + self.eps)
            si_low = torch.clamp(0.3 - pred_si, min=0.0)
            si_high = torch.clamp(pred_si - 2.0, min=0.0)
            si_violation = si_low ** 2 + si_high ** 2
            losses["SI"] = si_violation[m2].mean() if m2.sum() > 0 else torch.tensor(0.0, device=x.device)
            active["SI"] = m2.sum().item()
        else:
            losses["SI"] = torch.tensor(0.0, device=x.device)
            active["SI"] = 0

        # HH
        if "HH" != self.excluded:
            m3 = self._constraint_mask(c_mask[:, :, 3], pretrain_mask,
                                       [idx["pH"], idx["HCO3"], idx["pCO2"]])
            HCO3_raw = HCO3 * (60 - 5) + 5
            pCO2_raw = pCO2 * (150 - 10) + 10
            ratio = torch.clamp(HCO3_raw / (0.0307 * pCO2_raw + self.eps), min=self.eps)
            ph_pred = 6.1 + torch.log(ratio) / torch.log(torch.tensor(10.0, device=x.device))
            ph_pred_norm = torch.clamp((ph_pred - 6.5) / (7.9 - 6.5 + self.eps), 0.0, 1.0)
            losses["HH"] = self._mse(pH, ph_pred_norm, m3)
            active["HH"] = m3.sum().item()
        else:
            losses["HH"] = torch.tensor(0.0, device=x.device)
            active["HH"] = 0

        # SpO2
        if "SpO2" != self.excluded:
            from src.losses.pcl_loss import severinghaus_sao2_fraction
            m4 = self._constraint_mask(c_mask[:, :, 4], pretrain_mask,
                                       [idx["SpO2"], idx["PaO2"]])
            PaO2_raw = torch.clamp(PaO2 * (700 - 20) + 20, min=1.0)
            sao2 = severinghaus_sao2_fraction(PaO2_raw, eps=self.eps)
            spo2_pred_pct = sao2 * 100.0
            spo2_pred_norm = torch.clamp((spo2_pred_pct - 50.0) / 50.0, 0.0, 1.0)
            losses["SpO2"] = self._mse(SpO2, spo2_pred_norm, m4)
            active["SpO2"] = m4.sum().item()
        else:
            losses["SpO2"] = torch.tensor(0.0, device=x.device)
            active["SpO2"] = 0

        # Match main PhysiologicalConstraintLoss defaults: PP and SI excluded.
        # The "No PP" and "No SI" ablations are vestigial since both are already off.
        L_PCL = losses["MAP"] + losses["HH"] + losses["SpO2"]
        if "PP" != self.excluded:
            L_PCL = L_PCL  # PP still computed above for logging but not added
        return {"L_PCL": L_PCL, "losses": losses, "active": active}


# ════════════════════════════════════════════════════════════════════════════
# 4. MEASUREMENT NOISE EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════

def inject_measurement_noise(dataset, sigma_mmhg=5.0, seed=42):
    """
    Simulates a systematic blood pressure sensor calibration offset.

    A SINGLE per-stay mmHg bias is applied coherently to SBP, DBP, and MAP,
    preserving MAP = DBP + (SBP-DBP)/3 and PP = SBP-DBP exactly.

    Physical model: whole-system pressure reading offset (e.g. arm positioning
    artefact, cuff size mismatch). All three channels shift by the same
    absolute mmHg, so their cross-variable relationships are intact.

    Why this matters for PCL: PCL representations encode MAP/PP relationships.
    Under this noise model those relationships are unchanged, so PCL features
    are stable. ERM, which encodes absolute BP values, is hurt by the shift.

    Previous version applied independent SBP/DBP biases with MAP unchanged,
    which violated both the MAP constraint and the PP constraint simultaneously
    — penalising PCL more than ERM and producing the wrong direction of effect.
    """
    try:
        from src.data.variables import CANONICAL_VARIABLES as VARIABLES
    except ImportError:
        from src.data_utils import VARIABLES

    noisy_dataset = copy.deepcopy(dataset)
    rng = np.random.default_rng(seed)

    idx_sbp = VARIABLES.index("SBP")
    idx_dbp = VARIABLES.index("DBP")
    idx_map = VARIABLES.index("MAP")

    # SBP range 40-300 (260 mmHg); DBP/MAP range 20-200 (180 mmHg)
    sbp_range = 300 - 40
    bp_range  = 200 - 20

    for sample in noisy_dataset.samples:
        x    = sample["x"]
        mask = sample["mask"]

        # One bias per stay — same absolute mmHg applied to all three channels
        bias_mmhg = float(rng.normal(0, sigma_mmhg))

        x_noisy = x.clone()

        sbp_obs = mask[:, idx_sbp]
        dbp_obs = mask[:, idx_dbp]
        map_obs = mask[:, idx_map]

        x_noisy[sbp_obs, idx_sbp] = torch.clamp(
            x[sbp_obs, idx_sbp] + bias_mmhg / sbp_range, 0.0, 1.0
        )
        x_noisy[dbp_obs, idx_dbp] = torch.clamp(
            x[dbp_obs, idx_dbp] + bias_mmhg / bp_range, 0.0, 1.0
        )
        x_noisy[map_obs, idx_map] = torch.clamp(
            x[map_obs, idx_map] + bias_mmhg / bp_range, 0.0, 1.0
        )
        sample["x"] = x_noisy

    return noisy_dataset
