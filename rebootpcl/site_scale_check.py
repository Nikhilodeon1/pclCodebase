"""
Mechanistic check: does the violation score's per-site mean move because the
CONSTRAINT MIX changes, rather than because physiology is less consistent?

The score averages whichever constraints are computable. If the three constraints
have very different residual magnitudes, then a site that draws more blood gases
(firing HH and Severinghaus more often) gets a different average purely from the
mix. Measured on RAW PhysioNet data -- no model, so no model behaviour can be
blamed for the result.
"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "physionet2019")
LIM = int(sys.argv[1]) if len(sys.argv) > 1 else 4000

def site(dirname):
    d = os.path.join(ROOT, dirname)
    files = sorted(f for f in os.listdir(d) if f.endswith(".psv"))[:LIM]
    res = {"MAP": [], "HH": [], "SpO2": []}
    for fn in files:
        df = pd.read_csv(os.path.join(d, fn), sep="|")
        c = df.columns
        # MAP identity residual, normalized to [0,1] like the loss does
        if all(k in c for k in ("SBP", "DBP", "MAP")):
            m = df[["SBP", "DBP", "MAP"]].dropna()
            if len(m):
                pred = m.DBP + (m.SBP - m.DBP) / 3.0
                res["MAP"].append(np.abs((m.MAP - pred) / 180.0).values)
        # Henderson-Hasselbalch (squared, normalized pH scale)
        if all(k in c for k in ("pH", "HCO3", "PaCO2")):
            m = df[["pH", "HCO3", "PaCO2"]].dropna()
            m = m[(m.HCO3 > 0) & (m.PaCO2 > 0)]
            if len(m):
                pred = 6.1 + np.log10(m.HCO3 / (0.0307 * m.PaCO2))
                res["HH"].append((((m.pH - pred) / 1.4) ** 2).values)
        # Severinghaus (squared, normalized SpO2 scale)
        if "O2Sat" in c and "SaO2" in c:
            m = df[["O2Sat", "SaO2"]].dropna()
            if len(m):
                res["SpO2"].append((((m.O2Sat - m.SaO2) / 50.0) ** 2).values)
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in res.items()}

A, B = site("training_setA"), site("training_setB")
print(f"(first {LIM} stays per site)\n")
print(f"{'constraint':<10}{'A n':>10}{'A mean':>12}{'B n':>10}{'B mean':>12}{'scale vs MAP':>14}")
mA = A["MAP"].mean() if len(A["MAP"]) else np.nan
for k in ["MAP", "HH", "SpO2"]:
    a, b = A[k], B[k]
    am = a.mean() if len(a) else np.nan
    bm = b.mean() if len(b) else np.nan
    print(f"{k:<10}{len(a):>10}{am:>12.5f}{len(b):>10}{bm:>12.5f}{am/mA:>14.2f}")

# Simulate the score's averaging rule using each site's own constraint mix.
print("\nMean-of-active-constraints under each site's OWN mix:")
for nm, S in (("Site A", A), ("Site B", B)):
    vals = [S[k].mean() for k in ["MAP", "HH", "SpO2"] if len(S[k])]
    print(f"  {nm}: mean over {len(vals)} active constraint types = {np.mean(vals):.5f}")
