"""Every real-data detector ships an uncertainty measure; static ones do not.

The MIMIC near-miss (kappa 0.651, CI [0.528, 0.776], P(flag)=0.216) was found
only because it was deliberately computed. Nothing in the verdict signalled that
a control sat one resample in five from firing. These tests make the omission
visible instead of silent.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from detectors.harness import (DETERMINISTIC_CHECKS, STOCHASTIC_CHECKS,
                               fmt_uncertainty, requires_uncertainty)


class TestPolicy(unittest.TestCase):
    def test_real_data_detectors_require_uncertainty(self):
        self.assertEqual(sorted(STOCHASTIC_CHECKS), [1, 2, 5])
        for n in (1, 2, 5):
            self.assertTrue(requires_uncertainty(n))

    def test_static_analysis_detectors_do_not(self):
        self.assertEqual(sorted(DETERMINISTIC_CHECKS), [3, 4])
        for n in (3, 4):
            self.assertFalse(requires_uncertainty(n),
                             "a CI on deterministic static analysis would be "
                             "manufactured precision")

    def test_every_detector_is_classified(self):
        self.assertEqual(sorted(STOCHASTIC_CHECKS | DETERMINISTIC_CHECKS),
                         [1, 2, 3, 4, 5])

    def test_an_unclassified_detector_raises_rather_than_defaulting(self):
        with self.assertRaises(ValueError):
            requires_uncertainty(9)


class TestRendering(unittest.TestCase):
    def test_missing_measure_on_a_stochastic_detector_is_loud(self):
        self.assertIn("MISSING", fmt_uncertainty(1, None))
        self.assertIn("MISSING", fmt_uncertainty(5, {"kappa": 0.65}))

    def test_static_detectors_render_a_reason_not_a_gap(self):
        self.assertIn("deterministic", fmt_uncertainty(3, None))
        self.assertIn("deterministic", fmt_uncertainty(4, None))

    def test_confidence_interval_rendering(self):
        s = fmt_uncertainty(1, {"ci_lo": 0.528, "ci_hi": 0.776,
                                "p_would_flag": 0.216})
        self.assertIn("0.528", s)
        self.assertIn("0.776", s)
        self.assertIn("0.216", s)

    def test_flag_rate_rendering(self):
        self.assertIn("2/5", fmt_uncertainty(5, {"flag_rate": 0.4, "n": 5}))

    def test_mean_sd_rendering(self):
        self.assertIn("+/-", fmt_uncertainty(2, {"mean": 0.274, "sd": 0.036}))


if __name__ == "__main__":
    unittest.main()
