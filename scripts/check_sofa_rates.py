"""
Cheap A1 validation: compute SOFA-based Sepsis-3 positive rates on the FULL
MIMIC-IV / eICU data WITHOUT any model training. Run this on the pod to check
the rate lands in a believable range before wiring the label into the pipeline.

Usage (pod, with real data dirs):
    MIMIC_DIR=/workspace/mimic-iv EICU_DIR=/workspace/eicu \
        python scripts/check_sofa_rates.py

Notes:
  * Cohort here is age>=18 & LOS>=24h (the loaders' filter minus the
    has-hemodynamic-window step, which needs the vitals read); the rate is a
    close approximation of the modeled cohort's rate.
  * Reads are chunked+filtered, so memory stays bounded even on full eICU.
"""
import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from config import MIMIC_DIR, EICU_DIR, MIN_LOS_H, RESULTS_DIR
from src.data.mimic4 import load_stays
from src.data.sofa_sepsis import mimic_sofa_sepsis_labels, eicu_sofa_sepsis_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _eicu_cohort(data_path):
    p = pd.read_csv(os.path.join(data_path, "patient.csv.gz"),
                    usecols=["patientunitstayid", "age", "unitdischargeoffset"],
                    encoding_errors="replace", low_memory=False)
    p = p[p["age"].notna()]
    p["age_n"] = pd.to_numeric(p["age"].replace("> 89", 90), errors="coerce")
    p = p[(p["age_n"] >= 18) & (p["unitdischargeoffset"] / 60.0 >= MIN_LOS_H)]
    return p.reset_index(drop=True)


def main():
    out = {}
    print("=" * 60)
    if os.path.isdir(MIMIC_DIR):
        stays = load_stays(MIMIC_DIR)
        _, md = mimic_sofa_sepsis_labels(MIMIC_DIR, stays)
        out["MIMIC-IV"] = md
        print(f"MIMIC-IV : window={md['positive_rate_window']:.1%}  "
              f"single-point={md['positive_rate_singlepoint']:.1%}  "
              f"(n={md['n_stays']}, infection {md['n_suspected_infection']})")
        print(f"  cardio-alone frac of pos: {md['frac_pos_cardio_alone']:.2f} | "
              f"3+ organs imputed frac of pos: {md['frac_pos_3plus_organs_imputed']:.2f}")
        print("  observed_fraction:", {k: round(v, 2) for k, v in md["observed_fraction"].items()})
    else:
        print(f"MIMIC_DIR not found: {MIMIC_DIR}")

    if os.path.isdir(EICU_DIR):
        pats = _eicu_cohort(EICU_DIR)
        _, ed = eicu_sofa_sepsis_labels(EICU_DIR, pats)
        out["eICU"] = ed
        print(f"eICU     : window={ed['positive_rate_window']:.1%}  "
              f"single-point={ed['positive_rate_singlepoint']:.1%}  "
              f"(n={ed['n_stays']}, infection {ed['n_suspected_infection']})")
        print(f"  cardio-alone frac of pos: {ed['frac_pos_cardio_alone']:.2f} | "
              f"3+ organs imputed frac of pos: {ed['frac_pos_3plus_organs_imputed']:.2f}")
        print("  observed_fraction:", {k: round(v, 2) for k, v in ed["observed_fraction"].items()})
    else:
        print(f"EICU_DIR not found: {EICU_DIR}")
    print("=" * 60)
    print("Target: roughly 8-20% (eICU) / 20-40% (MIMIC full, long stays lower it).")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "sofa_rate_check.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("Wrote", path)


if __name__ == "__main__":
    main()
