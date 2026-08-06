import importlib
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.harness import Case, confusion

MODULES = ["check1_label_shift", "check2_pretrain_leakage",
           "check3_selection_audit", "check4_circularity",
           "check5_missingness_scale"]
FAST = ["check3_selection_audit", "check4_circularity"]


class TestEveryCheckExposesRun(unittest.TestCase):
    def test_run_exists_and_is_callable(self):
        for name in MODULES:
            m = importlib.import_module(f"rebootpcl.checks.{name}")
            self.assertTrue(hasattr(m, "run"), f"{name} has no run()")
            self.assertTrue(callable(m.run))

    def test_fast_checks_return_well_formed_cases(self):
        for name in FAST:
            m = importlib.import_module(f"rebootpcl.checks.{name}")
            cases = m.run(verbose=False)
            self.assertGreater(len(cases), 0, f"{name}.run() returned nothing")
            for c in cases:
                self.assertIsInstance(c, Case)
                self.assertIsInstance(c.flagged, bool)
                self.assertIsInstance(c.expected, bool)
                self.assertIsInstance(c.stats, dict)

    def test_fast_check_verdicts_are_unchanged_by_the_refactor(self):
        c3 = confusion(importlib.import_module(
            "rebootpcl.checks.check3_selection_audit").run(verbose=False))
        self.assertEqual(c3, {"TP": 1, "FP": 0, "FN": 0, "TN": 1})
        c4 = confusion(importlib.import_module(
            "rebootpcl.checks.check4_circularity").run(verbose=False))
        self.assertEqual(c4, {"TP": 1, "FP": 0, "FN": 0, "TN": 8})


class TestCheck3FileVerdictUsesAllSweeps(unittest.TestCase):
    def test_a_clean_first_sweep_does_not_mask_a_contaminated_second(self):
        """main() read res[0]['verdict'], so a file whose first sweep is clean
        and whose second is contaminated would be reported clean."""
        from rebootpcl.checks.check3_selection_audit import verdict_for_file
        res = [{"function": "a", "line": 1, "verdict": "OK", "findings": []},
               {"function": "b", "line": 9, "verdict": "CONTAMINATED",
                "findings": []}]
        self.assertEqual(verdict_for_file(res), "CONTAMINATED")

    def test_all_clean_is_ok(self):
        from rebootpcl.checks.check3_selection_audit import verdict_for_file
        res = [{"function": "a", "line": 1, "verdict": "OK", "findings": []},
               {"function": "b", "line": 9, "verdict": "OK", "findings": []}]
        self.assertEqual(verdict_for_file(res), "OK")

    def test_no_sweep_found_is_indeterminate_not_clean(self):
        from rebootpcl.checks.check3_selection_audit import verdict_for_file
        self.assertEqual(verdict_for_file([]), "INDETERMINATE")


if __name__ == "__main__":
    unittest.main()
