"""
CHECK 4 — circular / self-referential derived constraints.

Confound: a constraint is "verified" against a variable that was itself produced
by inverting that same equation. The residual is then partly an artifact of the
imputation, not evidence the model learned physiology. This is exactly the
Severinghaus situation on PhysioNet 2019, which records SaO2 but not PaO2, so
PaO2 is reconstructed by inverting the oxygen-dissociation relation the
constraint goes on to enforce.

Diagnostic: data-lineage tracing. Declare, per dataset, how each constrained
variable is obtained -- measured directly, or derived from another variable via a
named equation. For each (dataset, constraint) pair, flag it when any input to
the constraint carries a derivation whose equation is the constraint's own.

Ground truth is the project's real variable mapping: Severinghaus-on-PhysioNet
should flag; MAP and Henderson-Hasselbalch should never flag; Severinghaus on
MIMIC-IV and eICU should not flag, since both record PaO2 directly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# ── constraint definitions: which variables each equation consumes ───────────
CONSTRAINTS = {
    "MAP":          {"eq": "map_identity",  "inputs": ["SBP", "DBP", "MAP"]},
    "HendersonHass": {"eq": "henderson_hasselbalch", "inputs": ["pH", "HCO3", "pCO2"]},
    "Severinghaus": {"eq": "severinghaus", "inputs": ["SpO2", "PaO2"]},
}

# ── data lineage: (dataset, variable) -> how it was obtained ────────────────
# "measured"           : charted directly by the source system
# ("derived", equation): computed from other variables via that equation
LINEAGE = {
    "PhysioNet": {
        "SBP": "measured", "DBP": "measured", "MAP": "measured",
        "pH": "measured", "HCO3": "measured", "pCO2": "measured",
        "SpO2": "measured",
        # physionet2019.py reconstructs PaO2 from SaO2 by inverting Severinghaus
        "PaO2": ("derived", "severinghaus", "SaO2"),
    },
    "MIMIC-IV": {
        "SBP": "measured", "DBP": "measured", "MAP": "measured",
        "pH": "measured", "HCO3": "measured", "pCO2": "measured",
        "SpO2": "measured", "PaO2": "measured",
    },
    "eICU": {
        "SBP": "measured", "DBP": "measured", "MAP": "measured",
        "pH": "measured", "HCO3": "measured", "pCO2": "measured",
        "SpO2": "measured", "PaO2": "measured",
    },
}

# Expected verdicts, from the project's own audit of its loaders.
EXPECTED = {
    ("PhysioNet", "Severinghaus"): True,
    ("PhysioNet", "MAP"): False,
    ("PhysioNet", "HendersonHass"): False,
    ("MIMIC-IV", "Severinghaus"): False,
    ("MIMIC-IV", "MAP"): False,
    ("MIMIC-IV", "HendersonHass"): False,
    ("eICU", "Severinghaus"): False,
    ("eICU", "MAP"): False,
    ("eICU", "HendersonHass"): False,
}


def is_circular(dataset, constraint, lineage=LINEAGE, constraints=CONSTRAINTS):
    """True when any input to the constraint was derived from the same equation."""
    spec = constraints[constraint]
    for var in spec["inputs"]:
        prov = lineage.get(dataset, {}).get(var, "measured")
        if isinstance(prov, tuple) and prov[0] == "derived" and prov[1] == spec["eq"]:
            return True, f"{var} derived from {prov[2]} via {prov[1]}"
    return False, ""


def verify_lineage_against_code():
    """Cross-check the declared PhysioNet PaO2 derivation against the loader, so
    the lineage table cannot drift silently away from what the code does."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "src", "data", "physionet2019.py")
    if not os.path.exists(p):
        return None
    src = open(p, encoding="utf-8").read()
    # The inversion is recognisable by the Severinghaus constant and a cube root.
    inverts = "23400" in src and "cbrt" in src and "SaO2" in src
    return inverts


def run(lineage=None, expected=None, verbose=False):
    """One Case per (dataset, constraint) cell of the lineage table."""
    from rebootpcl.harness import Case
    lineage = LINEAGE if lineage is None else lineage
    expected = EXPECTED if expected is None else expected
    out = []
    for (ds, c), exp in expected.items():
        got, why = is_circular(ds, c, lineage=lineage)
        if verbose:
            print(f"{ds:<12}{c:<16}{str(got):>9}{str(exp):>10}   {why}")
        out.append(Case(f"{ds}/{c}", bool(got), bool(exp), {"reason": why}))
    return out


def main():
    print("=" * 74)
    print("CHECK 4 — circular / self-referential derived constraints")
    print("=" * 74)

    v = verify_lineage_against_code()
    print(f"\nlineage cross-check against src/data/physionet2019.py: "
          f"{'PaO2 inversion CONFIRMED in code' if v else 'not found' if v is False else 'loader not available'}")

    print(f"\n{'dataset':<12}{'constraint':<16}{'flagged':>9}{'expected':>10}{'result':>9}   reason")
    tp = fp = fn_ = tn = 0
    for (ds, c), exp in EXPECTED.items():
        got, why = is_circular(ds, c)
        ok = (got == exp)
        tp += (got and exp); fp += (got and not exp)
        fn_ += ((not got) and exp); tn += ((not got) and not exp)
        print(f"{ds:<12}{c:<16}{str(got):>9}{str(exp):>10}{'PASS' if ok else 'FAIL':>9}   {why}")

    print("\n" + "-" * 74)
    print(f"detections: TP={tp} FP={fp} FN={fn_} TN={tn}")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn_) if (tp + fn_) else float("nan")
    print(f"precision={prec:.2f}  recall={rec:.2f}")
    print("verdict:", "DETECTOR VALIDATED" if fp == 0 and fn_ == 0 else "NEEDS WORK")


if __name__ == "__main__":
    main()
