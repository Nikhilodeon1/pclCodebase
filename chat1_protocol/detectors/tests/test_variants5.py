"""The two detector-5 decision rules fixed in detectors/PREREGISTRATION.md.

Thresholds are frozen there and must not drift: AVAIL_RATIO_FLAG = 2.0, and the
composition gate at 0.30 (named COMP_EXPLAINS_FLAG in the pre-registration,
COMP_GAP_RATIO_FLAG in code -- same value, renamed because "explains" wrongly
implies a percentage).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from detectors.checks.check5_missingness_scale import (
    AVAIL_RATIO_FLAG, COMP_EXPLAINS_FLAG, COMP_GAP_RATIO_FLAG, VARIANTS,
    variant_a, variant_b)


def stats(ratio, gap_ratio):
    return {"max_avail_ratio": ratio, "composition_gap_ratio": gap_ratio}


class TestFrozenThresholds(unittest.TestCase):
    def test_thresholds_match_the_preregistration(self):
        self.assertEqual(AVAIL_RATIO_FLAG, 2.0)
        self.assertEqual(COMP_GAP_RATIO_FLAG, 0.30)

    def test_rename_kept_the_preregistered_value(self):
        self.assertEqual(COMP_EXPLAINS_FLAG, COMP_GAP_RATIO_FLAG)


class TestGapRatioIsNotAShare(unittest.TestCase):
    def test_a_ratio_above_one_is_accepted_not_clipped(self):
        """Measured at 1.414 on PhysioNet A vs itself. If this ever raises or
        clips, someone has re-imposed a [0,1] assumption the quantity does not
        satisfy."""
        self.assertTrue(variant_a(stats(5.0, 1.414)))
        self.assertTrue(variant_b(stats(5.0, 1.414)))


class TestVariantA(unittest.TestCase):
    def test_needs_both_signals(self):
        self.assertTrue(variant_a(stats(5.0, 0.50)))
        self.assertFalse(variant_a(stats(5.0, 0.20)), "composition gate")
        self.assertFalse(variant_a(stats(1.5, 0.50)), "availability gate")
        self.assertFalse(variant_a(stats(1.5, 0.20)))

    def test_is_silent_on_the_physionet_positive_case(self):
        """The measured PhysioNet A/B values: availability far past its gate,
        composition just under its own. This is the documented false negative."""
        self.assertFalse(variant_a(stats(36.4, 0.274)))


class TestVariantB(unittest.TestCase):
    def test_ignores_the_composition_share(self):
        self.assertTrue(variant_b(stats(5.0, 0.50)))
        self.assertTrue(variant_b(stats(5.0, 0.02)))
        self.assertFalse(variant_b(stats(1.5, 0.99)))

    def test_would_flag_the_physionet_positive_case(self):
        self.assertTrue(variant_b(stats(36.4, 0.274)))

    def test_still_respects_the_availability_gate(self):
        """B drops one constraint; it must not become unconditional."""
        self.assertFalse(variant_b(stats(2.0, 0.90)),
                         "gate is strictly greater-than")


class TestRegistry(unittest.TestCase):
    def test_exactly_the_two_preregistered_variants(self):
        self.assertEqual(sorted(VARIANTS), ["A_conjunction",
                                            "B_availability_only"])

    def test_registry_entries_are_the_functions(self):
        self.assertIs(VARIANTS["A_conjunction"], variant_a)
        self.assertIs(VARIANTS["B_availability_only"], variant_b)


if __name__ == "__main__":
    unittest.main()
