import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from detectors.baselines.baseline_checks import (BASELINES, auroc,
                                                external_floor,
                                                kfold_cv_instability,
                                                train_test_gap)


class TestAuroc(unittest.TestCase):
    def test_perfect_separation(self):
        self.assertAlmostEqual(auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_inverted_separation(self):
        self.assertAlmostEqual(auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)

    def test_all_ties_is_one_half(self):
        self.assertAlmostEqual(auroc([0, 1, 0, 1], [0.5] * 4), 0.5)

    def test_single_class_is_nan(self):
        self.assertTrue(np.isnan(auroc([1, 1, 1], [0.1, 0.2, 0.3])))

    def test_matches_a_known_value(self):
        # 1 positive ranked above 2 of 3 negatives -> 2/3
        self.assertAlmostEqual(auroc([0, 0, 1, 0], [0.1, 0.2, 0.5, 0.9]),
                               2.0 / 3.0)


class TestBaselines(unittest.TestCase):
    def test_registry_has_three_checks(self):
        self.assertEqual(sorted(BASELINES), ["external_floor",
                                             "kfold_cv_instability",
                                             "train_test_gap"])

    def test_gap_check_fires_only_on_degradation(self):
        worse = {"indomain_auroc": 0.85, "target_auroc": 0.70}
        better = {"indomain_auroc": 0.85, "target_auroc": 0.95}
        self.assertEqual(train_test_gap(worse), (True, True))
        self.assertEqual(train_test_gap(better), (False, True),
                         "a degradation-triggered check must stay silent when "
                         "cross-site performance IMPROVES -- this is the "
                         "directional property the experiment tests")

    def test_floor_check_ignores_indomain(self):
        self.assertEqual(external_floor({"target_auroc": 0.60}), (True, True))
        self.assertEqual(external_floor({"target_auroc": 0.80}), (False, True))

    def test_kfold_never_looks_at_the_target(self):
        m = {"kfold_sd": 0.01, "target_auroc": 0.10, "indomain_auroc": 0.99}
        self.assertEqual(kfold_cv_instability(m), (False, True),
                         "k-fold on the source cannot see a target-side "
                         "problem, by construction")

    def test_kfold_fires_on_unstable_folds(self):
        self.assertEqual(kfold_cv_instability({"kfold_sd": 0.12}), (True, True))


class TestUndecidable(unittest.TestCase):
    """An undefined AUROC must report UNDECIDABLE, never a silent False.

    On the first run seed 1's in-domain AUROC was nan, `nan > 0.05` evaluated
    False, and the abstention was counted as a correct silence -- which handed
    train_test_gap a spuriously perfect confusion matrix."""

    def test_nan_indomain_is_undecidable_not_silent(self):
        flagged, ok = train_test_gap({"indomain_auroc": float("nan"),
                                      "target_auroc": 0.62})
        self.assertFalse(ok)
        self.assertFalse(flagged)

    def test_nan_target_is_undecidable(self):
        self.assertEqual(external_floor({"target_auroc": float("nan")})[1],
                         False)

    def test_nan_kfold_is_undecidable(self):
        self.assertEqual(kfold_cv_instability({"kfold_sd": float("nan")})[1],
                         False)

    def test_missing_key_is_undecidable_not_a_crash(self):
        self.assertEqual(train_test_gap({})[1], False)


if __name__ == "__main__":
    unittest.main()
