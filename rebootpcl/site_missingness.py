"""
Why is mean constraint-violation higher on the Site-A calibration split than on
every target site?

Hypothesis: the violation score averages over whichever constraints happen to be
COMPUTABLE for a sample, and the three constraints live on very different scales
(MAP is an L1 residual, Henderson-Hasselbalch and Severinghaus are squared
residuals of differently-scaled quantities). Constraint computability depends on
arterial-blood-gas draw frequency, which varies by site. So a site that draws more
blood gases fires the acid-base constraints more often and its mean violation
moves toward those terms' magnitude -- independent of whether its physiology is
any less consistent.

If true, "mean violation" partly measures lab-ordering practice, not physiological
inconsistency, and any conformal result driven by cross-site differences in it is
an artifact.

Reads raw PhysioNet PSVs directly (Site A vs Site B), no model involved.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "physionet2019")
# Column names as they appear in the challenge PSVs.
MAP_COLS = ["SBP", "DBP", "MAP"]
HH_COLS = ["pH", "HCO3", "PaCO2"]
SPO2_COLS = ["O2Sat", "PaO2"]
SAO2_ALT = "SaO2"          # PaO2 is derived from SaO2 when PaO2 is absent


def scan(site_dir, limit=None):
    files = sorted(f for f in os.listdir(site_dir) if f.endswith(".psv"))
    if limit:
        files = files[:limit]
    tot_hours = 0
    comp = {"MAP": 0, "HH": 0, "SpO2": 0}
    present = {}
    n_stay = 0
    for i, fn in enumerate(files):
        df = pd.read_csv(os.path.join(site_dir, fn), sep="|")
        n_stay += 1
        h = len(df)
        tot_hours += h
        for c in set(MAP_COLS + HH_COLS + SPO2_COLS + [SAO2_ALT]):
            if c in df.columns:
                present[c] = present.get(c, 0) + int(df[c].notna().sum())
        ok = lambda cols: np.ones(h, bool) if not cols else np.logical_and.reduce(
            [df[c].notna().values if c in df.columns else np.zeros(h, bool) for c in cols])
        comp["MAP"] += int(ok(MAP_COLS).sum())
        comp["HH"] += int(ok(HH_COLS).sum())
        # PaO2 is reconstructed from SaO2 when absent, so the Severinghaus
        # constraint is computable if EITHER is charted alongside O2Sat.
        o2 = df["O2Sat"].notna().values if "O2Sat" in df.columns else np.zeros(h, bool)
        pao2 = df["PaO2"].notna().values if "PaO2" in df.columns else np.zeros(h, bool)
        sao2 = df[SAO2_ALT].notna().values if SAO2_ALT in df.columns else np.zeros(h, bool)
        comp["SpO2"] += int((o2 & (pao2 | sao2)).sum())
    return n_stay, tot_hours, comp, present


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rows = []
    for site in ["training_setA", "training_setB"]:
        d = os.path.join(ROOT, site)
        if not os.path.isdir(d):
            sys.exit(f"missing {d}")
        n, hours, comp, present = scan(d, limit)
        rows.append((site, n, hours, comp, present))
        print(f"{site}: {n} stays, {hours} stay-hours")

    print("\n" + "=" * 66)
    print("CONSTRAINT COMPUTABILITY BY SITE (fraction of stay-hours)")
    print("=" * 66)
    print(f"{'constraint':<12}{'Site A':>12}{'Site B':>12}{'A/B ratio':>12}")
    for k in ["MAP", "HH", "SpO2"]:
        a = rows[0][3][k] / max(rows[0][2], 1)
        b = rows[1][3][k] / max(rows[1][2], 1)
        r = (a / b) if b > 0 else float("inf")
        print(f"{k:<12}{a:>12.4f}{b:>12.4f}{r:>12.2f}")

    print("\nRaw variable availability (fraction of stay-hours charted)")
    keys = sorted(set(rows[0][4]) | set(rows[1][4]))
    print(f"{'variable':<12}{'Site A':>12}{'Site B':>12}")
    for c in keys:
        a = rows[0][4].get(c, 0) / max(rows[0][2], 1)
        b = rows[1][4].get(c, 0) / max(rows[1][2], 1)
        print(f"{c:<12}{a:>12.4f}{b:>12.4f}")


if __name__ == "__main__":
    main()
