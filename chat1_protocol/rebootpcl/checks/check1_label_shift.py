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


def build_scenarios(ids, icd, sofa_single, sofa_window, s1, s2, audit):
    """Scenario name -> (site1_labels, site2_labels, audit_a, audit_b, expected).

    The audit pair must always be two INDEPENDENTLY COMPUTED label arrays over
    the same patients. Passing one array twice makes kappa 1.0 by construction
    and proves nothing about the detector.

    Negative control: SOFA "window" and "single" are both valid Sepsis-3
    operationalizations, differing only in how the suspicion window is scored.
    A detector that flags this pair is flagging legitimate implementation
    variation, which is a false positive.
    """
    return {
        "positive (ICD vs SOFA)": (
            icd[s1], sofa_single[s2], icd[audit], sofa_single[audit], True),
        "negative (SOFA window vs single)": (
            sofa_window[s1], sofa_single[s2],
            sofa_window[audit], sofa_single[audit], False),
    }


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

    # Second valid Sepsis-3 operationalization, used as the negative control.
    sofa_win_map, _ = eicu_sofa_sepsis_labels(EICU_DIR, pats, mode="window")
    sofa_window = np.array([int(sofa_win_map.get(int(i), 0)) for i in ids])
    print(f"SOFA(window) prevalence={sofa_window.mean():.3f}")

    results, exp = {}, {}
    for name, (a, b, aa, ab, e) in build_scenarios(
            ids, icd, sofa, sofa_window, s1, s2, audit).items():
        results[name] = diagnose(name, a, b, aa, ab)
        exp[name] = e

    print("\n" + "-" * 78)
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

    # Show why prevalence alone would not be a sound flag. The split-noise
    # ratio is measured directly on ONE criterion across the two halves: that
    # is a genuine measurement (the halves are different patients), unlike a
    # kappa computed between an array and itself.
    # Measured for BOTH criteria: the rarer label carries more split noise, so
    # reporting only the commoner one would understate how close a legitimate
    # split comes to the prevalence threshold.
    pos_ratio = results["positive (ICD vs SOFA)"][2]
    print("\nsame-criterion split noise (one criterion, disjoint halves):")
    neg_ratio = 0.0
    for cname, arr in (("ICD", icd), ("SOFA", sofa)):
        pa, pb = float(arr[s1].mean()), float(arr[s2].mean())
        r = max(pa, pb) / max(min(pa, pb), 1e-9)
        neg_ratio = max(neg_ratio, r)
        print(f"  {cname:<5} prevalence {pa:.3f} vs {pb:.3f}  ratio={r:.2f}"
              f"{'   <-- within reach of the ' + str(PREV_RATIO_FLAG) + ' threshold' if r > 1.25 else ''}")
    print(f"\nprevalence ratio, positive={pos_ratio:.2f} vs negative={neg_ratio:.2f}: "
          f"{'separable' if pos_ratio > PREV_RATIO_FLAG >= neg_ratio else 'NOT separable'}"
          " — kappa on a fixed audit subset is the reliable signal.")


if __name__ == "__main__":
    main()
