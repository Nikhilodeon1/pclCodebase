"""Check 1's negative control must be a measurement, not an identity.

The original control passed the same audit array as both arguments to Cohen's
kappa, so the result was 1.0 by construction. These tests pin the property that
every scenario compares two independently computed label arrays.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from detectors.checks.check1_label_shift import build_scenarios


class TestNegativeControlIsNotTautological(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.n = 400
        self.ids = np.arange(self.n)
        self.icd = rng.integers(0, 2, self.n)
        self.sofa_single = rng.integers(0, 2, self.n)
        # window mode: mostly agrees with single, differs on 10% of stays
        self.sofa_window = self.sofa_single.copy()
        flip = rng.choice(self.n, size=self.n // 10, replace=False)
        self.sofa_window[flip] = 1 - self.sofa_window[flip]
        self.s1, self.s2 = np.arange(0, 200), np.arange(200, 400)
        self.audit = np.arange(0, 200)

    def scenarios(self):
        return build_scenarios(self.ids, self.icd, self.sofa_single,
                               self.sofa_window, self.s1, self.s2, self.audit)

    def test_no_scenario_compares_an_array_to_itself(self):
        for name, (_, _, a, b, _) in self.scenarios().items():
            self.assertFalse(
                np.array_equal(np.asarray(a), np.asarray(b)),
                f"scenario {name!r} compares an audit array to itself; its "
                "kappa is 1.0 by construction, not by measurement")

    def test_audit_arrays_are_the_same_patients(self):
        """Kappa is only interpretable when both criteria score the SAME
        patients -- that is what holds case mix fixed."""
        for name, (_, _, a, b, _) in self.scenarios().items():
            self.assertEqual(len(a), len(b),
                             f"scenario {name!r} scores different numbers of "
                             "patients under the two criteria")

    def test_a_negative_control_exists(self):
        negs = {n: v for n, v in self.scenarios().items() if v[4] is False}
        self.assertGreaterEqual(len(negs), 1)

    def test_positive_scenario_is_icd_vs_sofa(self):
        pos = [n for n, v in self.scenarios().items() if v[4] is True]
        self.assertEqual(len(pos), 1)
        self.assertIn("ICD", pos[0])


if __name__ == "__main__":
    unittest.main()
