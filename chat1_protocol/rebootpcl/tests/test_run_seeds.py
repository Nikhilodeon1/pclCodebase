import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.run_seeds import SEEDED_CHECKS, summarize


class TestSeedSweep(unittest.TestCase):
    def test_only_stochastic_checks_are_swept(self):
        """Checks 3 and 4 are deterministic static analysis over fixed files.
        Sweeping seeds over them yields a flag rate of exactly 0 or 1 with zero
        variance by construction, which is not evidence of stability."""
        self.assertEqual(sorted(SEEDED_CHECKS), [1, 2, 5])

    def test_summarize_reports_instability(self):
        sweep = {"pos": {"flags": [True, True, False, True, True],
                         "n": 5, "flag_rate": 0.8, "expected": True},
                 "neg": {"flags": [False] * 5,
                         "n": 5, "flag_rate": 0.0, "expected": False}}
        s = summarize(sweep)
        self.assertAlmostEqual(s["pos"]["flag_rate"], 0.8)
        self.assertTrue(s["pos"]["unstable"],
                        "a case that flags on 4 of 5 seeds is unstable")
        self.assertFalse(s["neg"]["unstable"])

    def test_stable_positive_is_not_flagged_unstable(self):
        sweep = {"pos": {"flags": [True] * 5, "n": 5,
                         "flag_rate": 1.0, "expected": True}}
        self.assertFalse(summarize(sweep)["pos"]["unstable"])

    def test_agreement_rate_accounts_for_ground_truth_direction(self):
        """A negative case that never flags agrees with truth 100% of the time,
        even though its flag rate is 0."""
        sweep = {"neg": {"flags": [False] * 5, "n": 5,
                         "flag_rate": 0.0, "expected": False},
                 "pos": {"flags": [True] * 5, "n": 5,
                         "flag_rate": 1.0, "expected": True}}
        s = summarize(sweep)
        self.assertEqual(s["neg"]["agrees_with_truth_rate"], 1.0)
        self.assertEqual(s["pos"]["agrees_with_truth_rate"], 1.0)

    def test_a_case_that_always_gets_it_wrong_scores_zero_agreement(self):
        sweep = {"pos": {"flags": [False] * 5, "n": 5,
                         "flag_rate": 0.0, "expected": True}}
        self.assertEqual(summarize(sweep)["pos"]["agrees_with_truth_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
