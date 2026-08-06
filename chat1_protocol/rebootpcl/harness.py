"""Shared result vocabulary for the confound detectors.

Every detector reports a list of Case records: one per scenario it evaluates,
each carrying what the detector decided and what the ground truth says. Keeping
this uniform is what lets the multi-seed, specificity, external-pipeline and
baseline-comparison experiments be loops over one interface instead of four
bespoke scripts.
"""
from collections import namedtuple

Case = namedtuple("Case", "name flagged expected stats")


def confusion(cases):
    """Confusion counts over a list of Case records."""
    c = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for x in cases:
        if x.flagged and x.expected:
            c["TP"] += 1
        elif x.flagged and not x.expected:
            c["FP"] += 1
        elif not x.flagged and x.expected:
            c["FN"] += 1
        else:
            c["TN"] += 1
    return c


def rates(counts):
    """Precision, recall and false-positive rate.

    Undefined ratios are nan, not zero: a detector that never fires has
    undefined precision, not zero precision, and reporting 0.0 would
    misrepresent it as maximally wrong.
    """
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    nan = float("nan")
    return {
        "precision": tp / (tp + fp) if (tp + fp) else nan,
        "recall": tp / (tp + fn) if (tp + fn) else nan,
        "fpr": fp / (fp + tn) if (fp + tn) else nan,
    }


def seed_sweep(run_fn, seeds):
    """Run a detector across seeds; report per-case flag rate.

    run_fn(seed) must return the same case names, with the same ground truth,
    for every seed. A case that appears under one seed and not another, or whose
    expected value moves, makes the flag rate a comparison between two different
    questions, so we raise rather than average them together.
    """
    acc, expected, names0 = {}, {}, None
    for s in seeds:
        cases = run_fn(s)
        names = {c.name for c in cases}
        if names0 is None:
            names0 = names
        elif names != names0:
            raise ValueError(f"case set changed at seed {s}: {names ^ names0}")
        for c in cases:
            if c.name in expected and expected[c.name] != bool(c.expected):
                raise ValueError(
                    f"ground truth for {c.name!r} changed at seed {s}")
            acc.setdefault(c.name, []).append(bool(c.flagged))
            expected[c.name] = bool(c.expected)
    return {n: {"flags": v,
                "n": len(v),
                "flag_rate": sum(v) / len(v),
                "expected": expected[n]}
            for n, v in acc.items()}


# Detectors that operate on real (non-injected) data or small-n splits must
# report an uncertainty measure alongside their verdict BY DEFAULT, not only
# when a number looks close. Detector 1's MIMIC near-miss (kappa 0.651, 95% CI
# [0.528, 0.776], P(flag)=0.216) was only found because someone went and
# computed it; nothing in the verdict itself signalled that the control was one
# resample in five away from firing.
#
# Detectors 3 and 4 are deliberately EXCLUDED. They are deterministic static
# analysis over fixed source files: the same input yields the same verdict every
# time, so any interval on them would be manufactured precision.
STOCHASTIC_CHECKS = {1, 2, 5}
DETERMINISTIC_CHECKS = {3, 4}


def requires_uncertainty(check_number):
    """Whether this detector must ship a variance or CI with its verdict."""
    if check_number in DETERMINISTIC_CHECKS:
        return False
    if check_number in STOCHASTIC_CHECKS:
        return True
    raise ValueError(f"unknown check {check_number!r}: classify it as "
                     "stochastic or deterministic before reporting it")


def fmt_uncertainty(check_number, stats):
    """Render the uncertainty column for a results-table row.

    Accepts whichever measure the detector produced -- a bootstrap CI, a
    flag rate across seeds, or a spread -- and refuses to render nothing for a
    detector that owes one, so a missing measure shows up as a visible gap in
    the table rather than as silent absence.
    """
    if not requires_uncertainty(check_number):
        return "deterministic (no CI: static analysis)"
    if stats is None:
        return "MISSING — stochastic detector reported no uncertainty"
    if "ci_lo" in stats and "ci_hi" in stats:
        s = f"95% CI [{stats['ci_lo']:.3f}, {stats['ci_hi']:.3f}]"
        if stats.get("p_would_flag") is not None:
            s += f", P(flag)={stats['p_would_flag']:.3f}"
        return s
    if "flag_rate" in stats and "n" in stats:
        return f"flagged {int(round(stats['flag_rate'] * stats['n']))}/{stats['n']} seeds"
    if "mean" in stats and "sd" in stats:
        return f"{stats['mean']:.3f} +/- {stats['sd']:.3f}"
    return "MISSING — stochastic detector reported no uncertainty"


def fmt_matrix(label, counts):
    r = rates(counts)
    return (f"{label:<28}TP={counts['TP']:<3}FP={counts['FP']:<3}"
            f"FN={counts['FN']:<3}TN={counts['TN']:<3}"
            f"prec={r['precision']:.2f} rec={r['recall']:.2f} "
            f"fpr={r['fpr']:.2f}")
