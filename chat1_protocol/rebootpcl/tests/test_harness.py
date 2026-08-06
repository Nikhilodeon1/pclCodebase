import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.harness import Case, confusion, rates, seed_sweep, fmt_matrix


class TestConfusion(unittest.TestCase):
    def test_counts_all_four_cells(self):
        cases = [Case("a", True, True, {}),
                 Case("b", True, False, {}),
                 Case("c", False, True, {}),
                 Case("d", False, False, {})]
        self.assertEqual(confusion(cases),
                         {"TP": 1, "FP": 1, "FN": 1, "TN": 1})

    def test_empty_is_all_zero(self):
        self.assertEqual(confusion([]), {"TP": 0, "FP": 0, "FN": 0, "TN": 0})


class TestRates(unittest.TestCase):
    def test_perfect_detector(self):
        r = rates({"TP": 1, "FP": 0, "FN": 0, "TN": 8})
        self.assertEqual(r["precision"], 1.0)
        self.assertEqual(r["recall"], 1.0)
        self.assertEqual(r["fpr"], 0.0)

    def test_undefined_precision_is_nan_not_zero(self):
        r = rates({"TP": 0, "FP": 0, "FN": 0, "TN": 5})
        self.assertTrue(math.isnan(r["precision"]),
                        "precision with no positive predictions is undefined")
        self.assertEqual(r["fpr"], 0.0)

    def test_undefined_fpr_is_nan(self):
        r = rates({"TP": 1, "FP": 0, "FN": 0, "TN": 0})
        self.assertTrue(math.isnan(r["fpr"]),
                        "fpr with no negative cases is undefined")


class TestSeedSweep(unittest.TestCase):
    def test_reports_flag_rate_per_case_across_seeds(self):
        def run_fn(seed):
            return [Case("x", seed % 2 == 0, True, {}),
                    Case("y", True, True, {})]

        out = seed_sweep(run_fn, [0, 1, 2, 3])
        self.assertEqual(out["x"]["flag_rate"], 0.5)
        self.assertEqual(out["y"]["flag_rate"], 1.0)
        self.assertEqual(out["x"]["n"], 4)
        self.assertTrue(out["x"]["expected"])

    def test_raises_when_a_case_vanishes_between_seeds(self):
        def run_fn(seed):
            return [Case("x", True, True, {})] if seed == 0 else []

        with self.assertRaises(ValueError):
            seed_sweep(run_fn, [0, 1])

    def test_raises_when_ground_truth_changes_between_seeds(self):
        """If a case's expected value moves, the flag rate is comparing two
        different questions and is meaningless."""
        def run_fn(seed):
            return [Case("x", True, seed == 0, {})]

        with self.assertRaises(ValueError):
            seed_sweep(run_fn, [0, 1])


class TestFmtMatrix(unittest.TestCase):
    def test_row_contains_label_and_all_counts(self):
        row = fmt_matrix("check1", {"TP": 1, "FP": 0, "FN": 0, "TN": 2})
        for token in ("check1", "TP=1", "FP=0", "TN=2"):
            self.assertIn(token, row)


if __name__ == "__main__":
    unittest.main()
