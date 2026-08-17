"""Check 2's detection threshold must match its sample size.

The original flag used a fixed t <= -2.0, which corresponds to no alpha at the
sample sizes actually run (df=2 needs 2.920 one-sided at 5%), and flagged on
sign agreement alone whenever fewer than 3 seeds were used.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from detectors.checks.check2_pretrain_leakage import flag_from_relative


class TestFlagThreshold(unittest.TestCase):
    def test_uses_df_appropriate_critical_value_not_fixed_two(self):
        # t around -2.5 at n=3 (df=2) is NOT significant one-sided at 5%
        # (critical 2.920). A fixed -2.0 threshold would wrongly flag it.
        rel = np.array([-0.10, -0.09, -0.02])
        flagged, t, crit = flag_from_relative(rel)
        self.assertGreater(crit, 2.9, "df=2 critical value must exceed 2.9")
        if t > -crit:
            self.assertFalse(flagged)

    def test_clear_effect_at_five_seeds_flags(self):
        rel = np.array([-0.20, -0.18, -0.22, -0.19, -0.21])
        flagged, t, crit = flag_from_relative(rel)
        self.assertTrue(flagged)
        self.assertLess(t, -crit)

    def test_requires_sign_agreement(self):
        rel = np.array([-0.30, -0.28, +0.25, -0.31, -0.29])
        flagged, _, _ = flag_from_relative(rel)
        self.assertFalse(flagged, "must not flag when seeds disagree in sign")

    def test_two_seeds_never_flag(self):
        rel = np.array([-0.40, -0.42])
        flagged, _, _ = flag_from_relative(rel)
        self.assertFalse(flagged, "n=2 is not enough evidence to flag")

    def test_positive_effect_never_flags(self):
        """Leakage must LOWER probe loss. A rise is not leakage."""
        rel = np.array([0.20, 0.18, 0.22, 0.19, 0.21])
        flagged, _, _ = flag_from_relative(rel)
        self.assertFalse(flagged)

    def test_zero_variance_does_not_divide_by_zero(self):
        rel = np.array([-0.10, -0.10, -0.10])
        flagged, t, _ = flag_from_relative(rel)
        self.assertTrue(np.isfinite(t))


class TestSplitSeedIsNotPinned(unittest.TestCase):
    def test_main_does_not_call_build_with_a_literal_seed(self):
        """The split was drawn once with seed=0 outside the seed loop, so every
        'seed' shared one split and split variance was excluded from the
        reported spread. Guard against that regressing."""
        import detectors.checks.check2_pretrain_leakage as m
        src = open(m.__file__, encoding="utf-8").read()
        self.assertNotIn("build(args.stays, seed=0)", src,
                         "build() must be called inside the seed loop with the "
                         "loop's seed, not pinned to 0")


if __name__ == "__main__":
    unittest.main()
