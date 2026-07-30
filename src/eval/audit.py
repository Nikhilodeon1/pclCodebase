import torch
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_utils import VARIABLES

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class PhysiologicalAudit:
    """
    Evaluates model predictions against hard physiological laws.
    
    Only evaluates at OBSERVED positions (using the mask) so the score
    reflects the model's actual clinical predictions, not its guesses
    at unobserved timesteps.
    """
    def __init__(self):
        self.idx = {v: i for i, v in enumerate(VARIABLES)}
    
    def check_consistency(self, x_pred, obs_mask):
        """
        Calculates Consistency Score given normalized predictions and observation mask.
        x_pred:   (B, T, V) — model predictions, normalized [0, 1]
        obs_mask: (B, T, V) — True where variable was originally observed
        
        Returns dict with overall score and per-rule breakdown.
        """
        from src.data_utils import PLAUS
        
        # Denormalize predictions to physical units
        SBP = x_pred[:, :, self.idx["SBP"]] * (300 - 40) + 40
        DBP = x_pred[:, :, self.idx["DBP"]] * (200 - 20) + 20
        MAP = x_pred[:, :, self.idx["MAP"]] * (200 - 20) + 20
        HR  = x_pred[:, :, self.idx["HR"]]  * (300 - 20) + 20
        pH  = x_pred[:, :, self.idx["pH"]]  * (7.9 - 6.5) + 6.5
        
        m_sbp = obs_mask[:, :, self.idx["SBP"]]
        m_dbp = obs_mask[:, :, self.idx["DBP"]]
        m_map = obs_mask[:, :, self.idx["MAP"]]
        m_hr  = obs_mask[:, :, self.idx["HR"]]
        m_ph  = obs_mask[:, :, self.idx["pH"]]
        
        results = {}
        
        # Rule 1: SBP >= DBP (at positions where both are observed)
        m_bp = m_sbp & m_dbp
        if m_bp.sum() > 0:
            rule1 = (SBP[m_bp] >= DBP[m_bp] - 1.0)  # 1mmHg tolerance
            results["sbp_geq_dbp"] = rule1.float().mean().item() * 100.0
        else:
            results["sbp_geq_dbp"] = 100.0
        
        # Rule 2: MAP equation consistency — |MAP - (DBP + (SBP-DBP)/3)| < 10 mmHg
        m_all_bp = m_sbp & m_dbp & m_map
        if m_all_bp.sum() > 0:
            map_pred = DBP[m_all_bp] + (SBP[m_all_bp] - DBP[m_all_bp]) / 3.0
            map_err = torch.abs(MAP[m_all_bp] - map_pred)
            rule2 = (map_err < 10.0)  # within 10 mmHg
            results["map_consistency"] = rule2.float().mean().item() * 100.0
            results["map_mae"] = map_err.mean().item()
        else:
            results["map_consistency"] = 100.0
            results["map_mae"] = 0.0
        
        # Rule 3: HR in plausible range [20, 300]
        if m_hr.sum() > 0:
            rule3 = (HR[m_hr] >= 20) & (HR[m_hr] <= 300)
            results["hr_plausible"] = rule3.float().mean().item() * 100.0
        else:
            results["hr_plausible"] = 100.0
        
        # Rule 4: pH in plausible range [6.5, 7.9]
        if m_ph.sum() > 0:
            rule4 = (pH[m_ph] >= 6.5) & (pH[m_ph] <= 7.9)
            results["ph_plausible"] = rule4.float().mean().item() * 100.0
        else:
            results["ph_plausible"] = 100.0
        
        # Overall: weighted average of applicable rules
        overall = (results["sbp_geq_dbp"] + results["map_consistency"] + 
                   results["hr_plausible"] + results["ph_plausible"]) / 4.0
        results["overall"] = overall
        
        return results


def audit_model_consistency(model, loader, x=None, mask=None, device="cpu"):
    """
    Computes consistency score over a loader or a direct tensor.
    Returns the overall consistency percentage.
    """
    model.to(device)
    model.eval()
    audit = PhysiologicalAudit()
    
    # Handle direct tensor input
    if x is not None:
        m = mask.to(device) if mask is not None else torch.ones_like(x, dtype=torch.bool)
        with torch.no_grad():
            reps = model.encode(x.to(device), m)
            x_pred = model.predict(reps)
            result = audit.check_consistency(x_pred.cpu(), m.cpu())
            return result

    # Handle standard DataLoader input
    all_results = []
    with torch.no_grad():
        for batch in loader:
            xb = batch["x"].to(device)
            mb = batch["mask"].to(device)
            reps = model.encode(xb, mb)
            x_pred = model.predict(reps)
            batch_result = audit.check_consistency(x_pred.cpu(), mb.cpu())
            all_results.append(batch_result)
    
    if not all_results:
        return {"overall": 0.0}
    
    # Average across batches
    avg_result = {}
    for key in all_results[0]:
        avg_result[key] = sum(r[key] for r in all_results) / len(all_results)
    
    return avg_result

def compute_violation_divergence_per_site(
    erm_model,
    pcl_model,
    site_loaders: dict,
    device: str = "cpu",
) -> dict:
    """
    Computes mean MAP constraint violation magnitude per site for frozen ERM and PCL
    representations. Replaces linear probing as the primary invariance evidence.

    ERM representations should show high, site-variable violations (encoder memorizes
    site-specific signal instead of physical laws). PCL representations should show
    low, consistent violations across sites.

    Args:
        erm_model:    trained ERM model (PCLModel)
        pcl_model:    trained PCL model (PCLModel)
        site_loaders: dict mapping site name → DataLoader
        device:       compute device

    Returns:
        dict with keys "ERM" and "PCL", each mapping site_name → mean_map_violation (mmHg)
    """
    idx = {v: i for i, v in enumerate(VARIABLES)}

    def _site_violation(model, loader):
        model.to(device).eval()
        errors = []
        with torch.no_grad():
            for batch in loader:
                xb = batch["x"].to(device)
                mb = batch["mask"].to(device)
                reps = model.encode(xb, mb)
                preds = model.predict(reps)

                SBP = preds[:, :, idx["SBP"]] * (300 - 40) + 40
                DBP = preds[:, :, idx["DBP"]] * (200 - 20) + 20
                MAP = preds[:, :, idx["MAP"]] * (200 - 20) + 20

                m_sbp = mb[:, :, idx["SBP"]]
                m_dbp = mb[:, :, idx["DBP"]]
                m_map = mb[:, :, idx["MAP"]]
                valid = m_sbp & m_dbp & m_map

                if valid.sum() > 0:
                    map_pred = DBP[valid] + (SBP[valid] - DBP[valid]) / 3.0
                    err = torch.abs(MAP[valid] - map_pred)
                    errors.append(err.cpu())

        if errors:
            all_err = torch.cat(errors)
            return float(all_err.mean()), float(all_err.std())
        return float("nan"), float("nan")

    results = {"ERM": {}, "PCL": {}}
    for site_name, loader in site_loaders.items():
        erm_mean, erm_std = _site_violation(erm_model, loader)
        pcl_mean, pcl_std = _site_violation(pcl_model, loader)
        results["ERM"][site_name] = {"mean_violation_mmhg": erm_mean, "std": erm_std}
        results["PCL"][site_name] = {"mean_violation_mmhg": pcl_mean, "std": pcl_std}
        logging.info(
            f"  {site_name}: ERM MAP err={erm_mean:.3f}±{erm_std:.3f} mmHg | "
            f"PCL MAP err={pcl_mean:.3f}±{pcl_std:.3f} mmHg"
        )

    return results


if __name__ == "__main__":
    audit = PhysiologicalAudit()
    dummy_preds = torch.rand(16, 48, len(VARIABLES))
    dummy_mask = torch.ones(16, 48, len(VARIABLES), dtype=torch.bool)
    result = audit.check_consistency(dummy_preds, dummy_mask)
    print(f"Medical Reality Check completed. Consistency Score: {result['overall']:.2f}%")
