"""
CHECK 1 — label-definition shift across sites.

Confound: two "sites" whose outcome labels are produced by different criteria.
Cross-site performance differences then conflate domain shift with a change in
what the label means. This is the ICD-code vs. SOFA-derived Sepsis-3 situation.

Diagnostic, two signals:
  (a) PREVALENCE RATIO between sites. Cheap, but weak on its own: two hospitals
      can legitimately differ in case mix, so prevalence alone false-positives.
  (b) AUDIT-SUBSET AGREEMENT. Score the SAME patients under both criteria and
      measure Cohen's kappa. Because the patients are held fixed, disagreement
      cannot be explained by case mix -- it isolates the definition itself.

We flag on (b), and report (a) alongside to show why (a) alone is insufficient.

Validation uses the project's own two labelers on eICU: explicit-sepsis ICD codes
(src/data/sepsis.py) and SOFA-based Sepsis-3 (src/data/sofa_sepsis.py).
  positive : site 1 labeled by ICD, site 2 by SOFA        -> must flag
  negative : both sites labeled by the SAME criterion     -> must stay silent
The negative control is run for both criteria so a false positive cannot hide.
"""
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

logging.disable(logging.WARNING)

KAPPA_FLAG = 0.60          # below this, the two criteria disagree materially
PREV_RATIO_FLAG = 1.50     # reported for comparison; not used to flag


def cohens_kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe >= 1.0 else float((po - pe) / (1 - pe))


def diagnose(name, site1, site2, audit_1, audit_2, verbose=True):
    """site1/site2: label arrays as each site would ship them (may use different
    criteria). audit_1/audit_2: BOTH criteria evaluated on the SAME held-out
    audit patients."""
    p1, p2 = float(np.mean(site1)), float(np.mean(site2))
    ratio = max(p1, p2) / max(min(p1, p2), 1e-9)
    k = cohens_kappa(audit_1, audit_2)
    agree = float((np.asarray(audit_1) == np.asarray(audit_2)).mean())
    flagged = (k < KAPPA_FLAG)
    if verbose:
        print(f"\n--- {name} ---")
        print(f"  prevalence      site1={p1:.3f}  site2={p2:.3f}  ratio={ratio:.2f}"
              f"   (weak signal, flag>{PREV_RATIO_FLAG})")
        print(f"  audit subset    n={len(audit_1)}  raw agreement={agree:.3f}  "
              f"kappa={k:.3f}   (flag<{KAPPA_FLAG})")
        print(f"  ==> {'FLAGGED: label-definition mismatch' if flagged else 'clean'}")
    return flagged, k, ratio


def main():
    from config import EICU_DIR
    from src.data.sepsis import eicu_sepsis_stay_ids
    from src.data.sofa_sepsis import eicu_sofa_sepsis_labels

    print("=" * 78)
    print("CHECK 1 — label-definition shift across sites")
    print("=" * 78)
    print(f"data: {EICU_DIR}")

    pats = pd.read_csv(os.path.join(EICU_DIR, "patient.csv.gz"),
                       usecols=["patientunitstayid", "unitdischargeoffset"],
                       encoding_errors="replace")
    pats = pats[pats["unitdischargeoffset"] / 60.0 >= 24].reset_index(drop=True)
    ids = pats["patientunitstayid"].astype(int).values

    # Criterion A: explicit-sepsis ICD codes.  Criterion B: SOFA-based Sepsis-3.
    icd_set = eicu_sepsis_stay_ids(EICU_DIR)
    icd = np.array([1 if i in icd_set else 0 for i in ids])
    sofa_map, _ = eicu_sofa_sepsis_labels(EICU_DIR, pats, mode="single")
    sofa = np.array([int(sofa_map.get(int(i), 0)) for i in ids])
    print(f"cohort n={len(ids)}   ICD prevalence={icd.mean():.3f}   "
          f"SOFA prevalence={sofa.mean():.3f}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ids))
    half = len(ids) // 2
    s1, s2 = perm[:half], perm[half:]
    # Audit subset: patients scored under BOTH criteria. Held fixed across
    # scenarios so case mix cannot explain any disagreement.
    audit = rng.choice(len(ids), size=min(500, len(ids)), replace=False)

    results = {}
    # POSITIVE: site 1 ships ICD labels, site 2 ships SOFA labels.
    results["positive (ICD vs SOFA)"] = diagnose(
        "POSITIVE: site1=ICD, site2=SOFA", icd[s1], sofa[s2], icd[audit], sofa[audit])

    # NEGATIVE CONTROLS: both sites use the same criterion. The audit subset is
    # then the same criterion against itself, so kappa must be 1.
    results["negative (both ICD)"] = diagnose(
        "NEGATIVE CONTROL: both sites = ICD", icd[s1], icd[s2], icd[audit], icd[audit])
    results["negative (both SOFA)"] = diagnose(
        "NEGATIVE CONTROL: both sites = SOFA", sofa[s1], sofa[s2], sofa[audit], sofa[audit])

    print("\n" + "-" * 78)
    exp = {"positive (ICD vs SOFA)": True,
           "negative (both ICD)": False, "negative (both SOFA)": False}
    tp = fp = fn_ = tn = 0
    for name, (flag, k, ratio) in results.items():
        e = exp[name]
        ok = (flag == e)
        tp += (flag and e); fp += (flag and not e)
        fn_ += ((not flag) and e); tn += ((not flag) and not e)
        print(f"{name:<26} flagged={str(flag):<6} expected={str(e):<6} "
              f"kappa={k:.3f}  {'PASS' if ok else 'FAIL'}")
    print(f"\ndetections: TP={tp} FP={fp} FN={fn_} TN={tn}")
    print("verdict:", "DETECTOR VALIDATED" if fp == 0 and fn_ == 0 else "NEEDS WORK")

    # Show why prevalence alone would not be a sound flag.
    pos_ratio = results["positive (ICD vs SOFA)"][2]
    neg_ratio = results["negative (both ICD)"][2]
    print(f"\nprevalence ratio, positive={pos_ratio:.2f} vs negative={neg_ratio:.2f}: "
          f"{'separable' if pos_ratio > PREV_RATIO_FLAG >= neg_ratio else 'NOT separable'}"
          " — kappa on a fixed audit subset is the reliable signal.")


if __name__ == "__main__":
    main()
