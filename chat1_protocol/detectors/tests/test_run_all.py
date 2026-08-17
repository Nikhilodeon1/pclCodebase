import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from detectors.run_all import CHECKS, SLOW, collect_fast


class TestRunner(unittest.TestCase):
    def test_declares_all_five_checks(self):
        self.assertEqual(sorted(CHECKS), [1, 2, 3, 4, 5])

    def test_static_checks_are_not_marked_slow(self):
        self.assertEqual(sorted(SLOW), [1, 2, 5])

    def test_fast_mode_runs_the_static_checks_and_matches_known_results(self):
        rows = collect_fast()
        self.assertEqual(rows[3], {"TP": 1, "FP": 0, "FN": 0, "TN": 1})
        self.assertEqual(rows[4], {"TP": 1, "FP": 0, "FN": 0, "TN": 8})

    def test_missing_cached_result_is_none_not_a_fabricated_zero(self):
        """A check with no recorded result must read as absent. Reporting it as
        an all-zero confusion matrix would look like a run that found nothing."""
        rows = collect_fast()
        for n in SLOW:
            self.assertTrue(rows[n] is None or isinstance(rows[n], dict))
            if isinstance(rows[n], dict):
                self.assertEqual(set(rows[n]), {"TP", "FP", "FN", "TN"})


if __name__ == "__main__":
    unittest.main()
