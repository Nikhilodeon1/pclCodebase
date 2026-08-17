"""
CHECK 5 — missingness/scale-driven artifacts in aggregate cross-site scores.

Confound: a score that aggregates several component terms, each computable only
when its inputs happen to be recorded, will move across sites when RECORDING
PRACTICE changes -- even if nothing physiological differs. If the components also
occupy different magnitude ranges, a site that records the large-magnitude
component more often scores higher for purely administrative reasons.

Diagnostic, three parts:
  (a) component availability ratio between the two sites;
  (b) component magnitude spread (how unequal the terms are);
  (c) a composition-only control -- recompute each stay's aggregate using POOLED
      component values shared by both sites, so component magnitude is held fixed
      and only the active-set composition can vary. Whatever gap survives is
      produced by recording practice alone.

Flag when availability is skewed AND the composition-only gap is large relative
to the observed gap. That relative quantity is a RATIO, not a share: it is
unbounded above 1 and must never be reported as a percentage (see
`composition_gap_ratio` below). Aggregation is per stay, matching how the
audited score works; aggregating at the site level averages the effect away.

Validation: PhysioNet Site A vs Site B is the positive case (known artifact,
50-fold difference in bicarbonate charting). A random split of Site A against
itself is the negative control -- identical recording practice, so the check must
stay silent.

Runs on raw PSV files. No model, no GPU.
"""
import os
import sys

import numpy as np
import pandas as pd

def _physionet_root():
    """PhysioNet location, honouring PHYSIONET_DIR.

    This used to be a hardcoded path relative to __file__, which silently
    ignored the PHYSIONET_DIR override every other loader respects. On the pod
    that meant detector 5 looked inside the repo for data living on the network
    volume, and the E4 regression guard died mid-run AFTER the multi-hour eICU
    and MIMIC loads had already completed.
    """
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        from config import PHYSIONET_DIR
        return PHYSIONET_DIR
    except Exception:
        return os.path.join(here, "data", "physionet2019")


ROOT = _physionet_root()

AVAIL_RATIO_FLAG = 2.0     # component availability differing by more than this
# Threshold on composition_gap_ratio. Value unchanged from the one frozen in
# PREREGISTRATION.md, where it is named COMP_EXPLAINS_FLAG; renamed here because
# "explains" wrongly implies a percentage. See the warning on the ratio below.
COMP_GAP_RATIO_FLAG = 0.30
COMP_EXPLAINS_FLAG = COMP_GAP_RATIO_FLAG   # pre-registration alias, do not reuse


# ── decision rules ───────────────────────────────────────────────────────────
# Two variants, specified in detectors/PREREGISTRATION.md BEFORE any external
# result existed. Variant A is the detector as originally written. Variant B
# drops the composition gate, which would recover the PhysioNet A/B true
# positive -- which is exactly why it may not be adopted on the strength of
# that case. It is tested prospectively on external data instead.

def variant_a(stats):
    """Conjunction: availability skew AND composition explaining much of the gap."""
    return bool(stats["max_avail_ratio"] > AVAIL_RATIO_FLAG
                and stats["composition_gap_ratio"] > COMP_GAP_RATIO_FLAG)


def variant_b(stats):
    """Availability-only. Reports composition_gap_ratio but does not gate on it."""
    return bool(stats["max_avail_ratio"] > AVAIL_RATIO_FLAG)


VARIANTS = {"A_conjunction": variant_a, "B_availability_only": variant_b}


def components(df):
    """Per-timestep residuals for each constraint, normalized to comparable units.
    Returns {name: array} using only timesteps where that constraint is computable."""
    out = {}
    c = df.columns
    if all(k in c for k in ("SBP", "DBP", "MAP")):
        m = df[["SBP", "DBP", "MAP"]].dropna()
        if len(m):
            out["MAP"] = np.abs((m.MAP - (m.DBP + (m.SBP - m.DBP) / 3.0)) / 180.0).values
    if all(k in c for k in ("pH", "HCO3", "PaCO2")):
        m = df[["pH", "HCO3", "PaCO2"]].dropna()
        m = m[(m.HCO3 > 0) & (m.PaCO2 > 0)]
        if len(m):
            pred = 6.1 + np.log10(m.HCO3 / (0.0307 * m.PaCO2))
            out["HH"] = (((m.pH - pred) / 1.4) ** 2).values
    if "O2Sat" in c and "SaO2" in c:
        m = df[["O2Sat", "SaO2"]].dropna()
        if len(m):
            out["SpO2"] = (((m.O2Sat - m.SaO2) / 50.0) ** 2).values
    return out


def scan(files):
    """Per-STAY component residuals.

    The score being audited aggregates per sample: each stay averages over the
    constraints computable FOR THAT STAY. Aggregating at the site level instead
    averages away the very composition effect we are trying to detect, so we keep
    stay-level structure here.

    Returns (list of {component: mean residual for that stay}, total timesteps).
    """
    stays, hours = [], 0
    for path in files:
        df = pd.read_csv(path, sep="|")
        hours += len(df)
        comp = components(df)
        if comp:
            stays.append({k: float(v.mean()) for k, v in comp.items() if len(v)})
    return stays, hours


def run_check(name_a, files_a, name_b, files_b, verbose=True, rule=variant_a):
    """Path-based entry point: read PSVs, then run the shared diagnostic."""
    A, _ = scan(files_a)
    B, _ = scan(files_b)
    return run_check_on_stays(name_a, A, name_b, B, verbose=verbose, rule=rule,
                              keys=["MAP", "HH", "SpO2"])


def run_check_on_stays(name_a, A, name_b, B, verbose=True, rule=variant_a,
                       keys=("MAP", "HH", "SpO2")):
    """The diagnostic itself, over per-stay {component: mean residual} dicts.

    Split out from run_check so external datasets run the SAME code rather than
    a reimplementation of it -- a reimplementation would validate the rewrite,
    not the detector.
    """
    keys = list(keys)

    def avail(stays, k):
        return sum(1 for s in stays if k in s) / max(len(stays), 1)

    def comp_mean(stays, k):
        v = [s[k] for s in stays if k in s]
        return float(np.mean(v)) if v else np.nan

    # Pooled per-component level, shared by both sites. Used to hold component
    # MAGNITUDE fixed so only the active-set composition can vary.
    pooled = {}
    for k in keys:
        v = [s[k] for s in A + B if k in s]
        pooled[k] = float(np.mean(v)) if v else np.nan

    def aggregate(stays, use_pooled):
        """Mean over stays of (mean over that stay's ACTIVE components)."""
        out = []
        for st in stays:
            vals = [pooled[k] if use_pooled else st[k] for k in keys if k in st]
            if vals:
                out.append(float(np.mean(vals)))
        return float(np.mean(out)) if out else np.nan

    naive = (aggregate(A, False), aggregate(B, False))
    # Composition-only: identical component values for both sites, so any
    # remaining gap is produced purely by WHICH components are available.
    comp_only = (aggregate(A, True), aggregate(B, True))

    rel = lambda t: abs(t[0] - t[1]) / max(abs(t[0]), abs(t[1]), 1e-12)
    naive_gap, comp_gap = rel(naive), rel(comp_only)
    # NOT a share and NOT a percentage. This is the ratio of two gaps that move
    # independently, so it is UNBOUNDED ABOVE 1: when the composition and
    # magnitude effects partially cancel in the naive aggregate the denominator
    # goes small and the ratio exceeds 1 (measured at 1.414 on PhysioNet A vs
    # itself). Never report it as "composition explains X% of the gap".
    composition_gap_ratio = (comp_gap / naive_gap) if naive_gap > 1e-9 else 0.0

    max_ratio = 0.0
    for k in keys:
        a, b = avail(A, k), avail(B, k)
        if a > 0 and b > 0:
            max_ratio = max(max_ratio, max(a / b, b / a))
        elif a != b:
            max_ratio = float("inf")

    stats = {"max_avail_ratio": float(max_ratio),
             "composition_gap_ratio": float(composition_gap_ratio),
             "naive_gap": float(naive_gap), "comp_gap": float(comp_gap),
             "n_stays_a": len(A), "n_stays_b": len(B)}
    flagged = rule(stats)

    if verbose:
        print(f"")
        print(f"--- {name_a}  vs  {name_b} ---")
        print(f"{'component':<10}{'avail A':>10}{'avail B':>10}{'ratio':>9}"
              f"{'mean A':>11}{'mean B':>11}{'scale':>8}")
        base = pooled[keys[0]]      # scale column is relative to the first term
        for k in keys:
            a, b = avail(A, k), avail(B, k)
            r = max(a, b) / min(a, b) if min(a, b) > 0 else float("inf")
            ma, mb = comp_mean(A, k), comp_mean(B, k)
            print(f"{k:<10}{a:>10.4f}{b:>10.4f}{r:>9.1f}{ma:>11.5f}{mb:>11.5f}"
                  f"{pooled[k]/base:>8.2f}")
        print(f"  aggregate, as scored      : {naive[0]:.5f} vs {naive[1]:.5f}"
              f"   (relative gap {naive_gap:.3f})")
        print(f"  aggregate, composition-only: {comp_only[0]:.5f} vs {comp_only[1]:.5f}"
              f"   (relative gap {comp_gap:.3f})")
        print(f"  max availability ratio {max_ratio:.1f} (flag > {AVAIL_RATIO_FLAG})")
        print(f"  composition/naive gap RATIO (unbounded, not a %): "
              f"{composition_gap_ratio:.2f} (flag > {COMP_GAP_RATIO_FLAG})")
        print(f"  ==> {'FLAGGED: composition artifact' if flagged else 'clean'}")
    return flagged, stats


def sample_files(seed, n, legacy=False):
    """Seeded random stay samples from each site.

    `legacy=True` reproduces the original selection: `sorted(listdir)[:n]`, an
    alphabetical prefix. That prefix scored the same stays on every run, so the
    result carried no sampling variance -- and PhysioNet filenames are patient
    IDs, so a prefix is not a random sample of the site.
    """
    a_dir = os.path.join(ROOT, "training_setA")
    b_dir = os.path.join(ROOT, "training_setB")
    if not os.path.isdir(a_dir):
        sys.exit(f"missing {a_dir}")
    all_a = [f for f in sorted(os.listdir(a_dir)) if f.endswith(".psv")]
    all_b = [f for f in sorted(os.listdir(b_dir)) if f.endswith(".psv")]
    if legacy:
        ia, ib = range(2 * n), range(n)
    else:
        rng = np.random.default_rng(seed)
        ia = rng.permutation(len(all_a))[:2 * n]
        ib = rng.permutation(len(all_b))[:n]
    return ([os.path.join(a_dir, all_a[i]) for i in ia],
            [os.path.join(b_dir, all_b[i]) for i in ib])


def run(seed=0, n=1200, pair=None, verbose=False, legacy=False):
    """One Case for the cross-site positive and one for the same-site control.

    `pair` overrides the default PhysioNet A/B positive with
    (name_a, files_a, name_b, files_b, expected_flag), used by the external
    validation to point the same diagnostic at a different dataset pair.
    """
    from detectors.harness import Case
    fa, fb = sample_files(seed, n, legacy=legacy)

    if pair is None:
        pos_args = ("PhysioNet Site A", fa[:n], "PhysioNet Site B", fb, True)
    else:
        pos_args = pair
    na, la, nb, lb, exp = pos_args
    pos, pos_stats = run_check(na, la, nb, lb, verbose=verbose)

    # NEGATIVE CONTROL: Site A split against itself. Same recording practice,
    # so a well-behaved detector must stay silent.
    half = len(fa) // 2
    neg, neg_stats = run_check(f"{na} (half 1)", fa[:half],
                               f"{na} (half 2)", fa[half:], verbose=verbose)

    return [Case(f"positive ({na} vs {nb})", bool(pos), bool(exp),
                 dict(pos_stats, seed=seed, n=n)),
            Case(f"negative ({na} vs itself)", bool(neg), False,
                 dict(neg_stats, seed=seed, n=n))]


def main():
    from detectors.harness import confusion, fmt_matrix
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print("=" * 78)
    print("CHECK 5 — missingness/scale artifacts in aggregate cross-site scores")
    print("=" * 78)
    print(f"(sampling {n} stays per arm, seed={seed})")

    cases = run(seed=seed, n=n, verbose=True)

    print("\n" + "-" * 78)
    for c in cases:
        print(f"{c.name:<40} flagged={str(c.flagged):<6} "
              f"expected={str(c.expected)}")
    counts = confusion(cases)
    print("\n" + fmt_matrix("check5", counts))
    print("verdict:", "DETECTOR VALIDATED"
          if counts["FP"] == 0 and counts["FN"] == 0 else "NEEDS WORK")


if __name__ == "__main__":
    main()
