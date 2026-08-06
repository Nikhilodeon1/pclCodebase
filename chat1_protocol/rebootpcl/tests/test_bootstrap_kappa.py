import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.external.bootstrap_kappa import bootstrap_kappa_ci


class TestBootstrapCI(unittest.TestCase):
    def test_identical_labels_give_an_interval_at_one(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, 200)
        point, lo, hi, _ = bootstrap_kappa_ci(a, a, iters=400, seed=1)
        self.assertAlmostEqual(point, 1.0, places=9)
        self.assertGreater(lo, 0.99)

    def test_independent_labels_give_an_interval_around_zero(self):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 2, 400)
        b = rng.integers(0, 2, 400)
        point, lo, hi, _ = bootstrap_kappa_ci(a, b, iters=400, seed=2)
        self.assertLess(abs(point), 0.25)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(2)
        a = rng.integers(0, 2, 150)
        b = a.copy()
        flip = rng.choice(150, 30, replace=False)
        b[flip] = 1 - b[flip]
        point, lo, hi, _ = bootstrap_kappa_ci(a, b, iters=500, seed=3)
        self.assertLessEqual(lo, point)
        self.assertLessEqual(point, hi)

    def test_smaller_n_gives_a_wider_interval(self):
        """The whole reason for this analysis: n=117 must not be reported with
        the confidence of n=500."""
        rng = np.random.default_rng(3)
        big = rng.integers(0, 2, 800)
        big_b = big.copy()
        flip = rng.choice(800, 160, replace=False)
        big_b[flip] = 1 - big_b[flip]
        _, lo_b, hi_b, _ = bootstrap_kappa_ci(big, big_b, iters=500, seed=4)
        _, lo_s, hi_s, _ = bootstrap_kappa_ci(big[:100], big_b[:100],
                                              iters=500, seed=4)
        self.assertGreater(hi_s - lo_s, hi_b - lo_b)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            bootstrap_kappa_ci(np.zeros(10, int), np.zeros(9, int), iters=10)

    def test_degenerate_resamples_are_dropped_not_coerced(self):
        """An all-zero arm makes kappa undefined. Coercing those draws to 0 or 1
        would bias the interval toward or away from the threshold."""
        a = np.zeros(40, int)
        a[0] = 1
        b = a.copy()
        point, lo, hi, draws = bootstrap_kappa_ci(a, b, iters=300, seed=5)
        self.assertTrue(np.all(np.isfinite(draws)))
        self.assertGreater(len(draws), 0)


if __name__ == "__main__":
    unittest.main()
