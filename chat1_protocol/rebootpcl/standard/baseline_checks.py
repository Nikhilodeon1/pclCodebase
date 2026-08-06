"""What standard evaluation practice catches, scored the same way as the detectors.

Three checks a competent group would already run. Each takes the SAME injected
scenarios detector 2 and detector 5 were validated on and returns Case records,
so the comparison is like-for-like rather than a narrative.

  kfold_cv_instability   flags when k-fold AUROC variance on the source is high
  train_test_gap         flags when in-domain AUROC exceeds cross-site AUROC by
                         more than a threshold (the usual "it dropped on the
                         other hospital" alarm)
  external_floor         flags when cross-site AUROC falls below an absolute floor

The hypothesis under test -- NOT assumed here -- is that under pretraining
leakage cross-site performance IMPROVES, so a degradation-triggered check stays
silent exactly when the confound is most damaging, and its miss rate rises with
leakage. That is a directional prediction and it is scored, not asserted.
Thresholds are set once, below, and are not tuned per scenario.
"""
import numpy as np

# Fixed thresholds. Chosen to be conventional rather than fitted: a 5-point
# AUROC drop is a common informal alarm level, 0.70 a common "acceptable
# discrimination" floor, and 0.05 a visibly unstable fold spread.
KFOLD_SD_FLAG = 0.05
GAP_FLAG = 0.05
FLOOR_FLAG = 0.70


# Every check returns (flagged, decidable). `decidable` is False when the
# metric it needs is undefined -- an AUROC on a split that turned out to hold
# one class only. Returning a bare bool would silently score nan as "did not
# flag", which counts a non-measurement as a correct silence. That is the same
# error as scoring detector 3's INDETERMINATE as a true negative, and it
# produced a spuriously perfect matrix for train_test_gap on the first run.
def _ok(*vals):
    return all(v is not None and not np.isnan(v) for v in vals)


def kfold_cv_instability(metrics):
    """Standard k-fold cross-validation on the SOURCE site only.

    Included because it is the most common thing people actually do, and it
    never looks at the target at all.
    """
    v = metrics.get("kfold_sd")
    if not _ok(v):
        return False, False
    return bool(v > KFOLD_SD_FLAG), True


def train_test_gap(metrics):
    """In-domain minus cross-site AUROC. Fires only on DEGRADATION."""
    a, b = metrics.get("indomain_auroc"), metrics.get("target_auroc")
    if not _ok(a, b):
        return False, False
    return bool((a - b) > GAP_FLAG), True


def external_floor(metrics):
    """Absolute floor on cross-site AUROC."""
    v = metrics.get("target_auroc")
    if not _ok(v):
        return False, False
    return bool(v < FLOOR_FLAG), True


BASELINES = {
    "kfold_cv_instability": kfold_cv_instability,
    "train_test_gap": train_test_gap,
    "external_floor": external_floor,
}


def auroc(y_true, scores):
    """Rank-based AUROC; nan when only one class is present."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    pos, neg = (y > 0.5), (y <= 0.5)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # average ranks within ties so tied scores cannot inflate the statistic
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
