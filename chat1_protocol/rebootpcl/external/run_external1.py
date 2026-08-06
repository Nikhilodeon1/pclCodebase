"""Detector 1 external validation, with a broadened false-positive side.

Detector 1's diagnostic is a within-audit-subset Cohen's kappa between two
labelling criteria scored on the SAME patients. "External" therefore means
running that diagnostic on a database it was not built against: it was developed
on eICU, so MIMIC-IV demo is the external case.

The negative side is the point of this run. Detector 1 previously had exactly ONE
clean control (SOFA window-mode vs single-mode), and a single passing control is
an untested edge, not evidence -- that is what nearly hid detector 5's problem.
Here every database gets several controls drawn from choices a competent group
could each defend:

  * window-mode vs single-point scoring
  * suspicion window [-48,+24] vs [-24,+12] hours
  * suspicion window [-48,+24] vs [-72,+24] hours

All three are Sepsis-3 operationalizations; none is a definition change. A
detector that fires on any of them is producing a false positive on legitimate
implementation variation.

The positive case is ICD administrative coding against SOFA-derived Sepsis-3,
whose ground truth comes from the definitions themselves rather than from the
detector's own output.

    PCL_TEST_MODE=1 python rebootpcl/external/run_external1.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.harness import Case, confusion, fmt_matrix
from rebootpcl.checks.check1_label_shift import diagnose

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")

# (label, mode, pre_h, post_h) — each a defensible Sepsis-3 operationalization.
SOFA_VARIANTS = [
    ("SOFA win[-48,+24]", "window", 48.0, 24.0),
    ("SOFA single-point", "single", 48.0, 24.0),
    ("SOFA win[-24,+12]", "window", 24.0, 12.0),
    ("SOFA win[-72,+24]", "window", 72.0, 24.0),
]


def mimic_labels():
    """(ids, icd, {variant label: sofa array}) over the MIMIC-IV demo cohort."""
    from config import MIMIC_DIR
    from src.data.mimic4 import load_stays
    from src.data.sepsis import mimic_sepsis_hadm_ids
    from src.data.sofa_sepsis import mimic_sofa_sepsis_labels

    stays = load_stays(MIMIC_DIR)
    ids = stays["stay_id"].astype(int).values
    hadm_pos = mimic_sepsis_hadm_ids(MIMIC_DIR)
    icd = np.array([1 if int(h) in hadm_pos else 0
                    for h in stays["hadm_id"].astype(int).values])

    sofa = {}
    for label, mode, pre, post in SOFA_VARIANTS:
        m, _ = mimic_sofa_sepsis_labels(MIMIC_DIR, stays, mode=mode,
                                        pre_h=pre, post_h=post)
        sofa[label] = np.array([int(m.get(int(i), 0)) for i in ids])
    return ids, icd, sofa


def eicu_labels():
    """(ids, icd, {variant label: sofa array}) over the eICU cohort."""
    from config import EICU_DIR
    from src.data.sepsis import eicu_sepsis_stay_ids
    from src.data.sofa_sepsis import eicu_sofa_sepsis_labels

    pats = pd.read_csv(os.path.join(EICU_DIR, "patient.csv.gz"),
                       usecols=["patientunitstayid", "unitdischargeoffset"],
                       encoding_errors="replace")
    pats = pats[pats["unitdischargeoffset"] / 60.0 >= 24].reset_index(drop=True)
    ids = pats["patientunitstayid"].astype(int).values
    pos = eicu_sepsis_stay_ids(EICU_DIR)
    icd = np.array([1 if i in pos else 0 for i in ids])

    sofa = {}
    for label, mode, pre, post in SOFA_VARIANTS:
        m, _ = eicu_sofa_sepsis_labels(EICU_DIR, pats, mode=mode,
                                       pre_h=pre, post_h=post)
        sofa[label] = np.array([int(m.get(int(i), 0)) for i in ids])
    return ids, icd, sofa


def scenarios(db, ids, icd, sofa, seed=0, verbose=False):
    """One positive (ICD vs SOFA) and several legitimate-variation negatives."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    half = len(ids) // 2
    s1, s2 = perm[:half], perm[half:]
    audit = rng.choice(len(ids), size=min(500, len(ids)), replace=False)

    base = "SOFA win[-48,+24]"
    specs = [(f"{db} POSITIVE ICD vs {base}", icd, sofa[base], True)]
    for label in ("SOFA single-point", "SOFA win[-24,+12]", "SOFA win[-72,+24]"):
        specs.append((f"{db} negative {base} vs {label}",
                      sofa[base], sofa[label], False))

    out = []
    for name, a, b, expected in specs:
        flag, k, ratio = diagnose(name, a[s1], b[s2], a[audit], b[audit],
                                  verbose=verbose)
        out.append(Case(name, bool(flag), bool(expected),
                        {"kappa": float(k), "prevalence_ratio": float(ratio),
                         "n_audit": int(len(audit)), "n_cohort": int(len(ids)),
                         "prev_a": float(a.mean()), "prev_b": float(b.mean())}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 78)
    print("DETECTOR 1 — external validation (MIMIC-IV demo) + broadened controls")
    print("=" * 78, flush=True)

    print("\nlabelling MIMIC-IV demo ...", flush=True)
    m_ids, m_icd, m_sofa = mimic_labels()
    print(f"  cohort n={len(m_ids)}  ICD prevalence={m_icd.mean():.3f}")
    for k, v in m_sofa.items():
        print(f"    {k:<22} prevalence={v.mean():.3f}")

    print("\nlabelling eICU ...", flush=True)
    e_ids, e_icd, e_sofa = eicu_labels()
    print(f"  cohort n={len(e_ids)}  ICD prevalence={e_icd.mean():.3f}")
    for k, v in e_sofa.items():
        print(f"    {k:<22} prevalence={v.mean():.3f}")

    cases = (scenarios("MIMIC", m_ids, m_icd, m_sofa, args.seed, verbose=True)
             + scenarios("eICU", e_ids, e_icd, e_sofa, args.seed, verbose=True))

    print("\n" + "-" * 78)
    for c in cases:
        ok = (c.flagged == c.expected)
        print(f"{c.name:<44} kappa={c.stats['kappa']:.3f}  "
              f"flagged={str(c.flagged):<6} expected={str(c.expected):<6} "
              f"{'PASS' if ok else 'FAIL'}")

    ext = [c for c in cases if c.name.startswith("MIMIC")]
    print("\n" + fmt_matrix("check1 EXTERNAL (MIMIC)", confusion(ext)))
    print(fmt_matrix("check1 eICU (original db)",
                     confusion([c for c in cases if c.name.startswith("eICU")])))
    print(fmt_matrix("check1 combined", confusion(cases)))

    neg = [c for c in cases if not c.expected]
    worst = min(neg, key=lambda c: c.stats["kappa"])
    print(f"\nthinnest margin on a legitimate variant: {worst.name} "
          f"kappa={worst.stats['kappa']:.3f} (flag threshold 0.60)")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "external1.json"), "w", encoding="utf-8") as fh:
        json.dump({c.name: {"flagged": c.flagged, "expected": c.expected,
                            **c.stats} for c in cases}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
