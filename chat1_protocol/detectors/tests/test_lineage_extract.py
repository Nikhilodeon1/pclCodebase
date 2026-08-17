"""Extraction must reproduce the hand-written table INCLUDING the guarded case.

physionet2019.py derives PaO2 from SaO2 unconditionally (reachable) and HCO3
from base excess behind `if "HCO3" not in df.columns` (never reachable, because
every PhysioNet PSV carries an HCO3 column). An extractor without guard
reasoning gets the second one wrong and hands detector 4 a false positive.
"""
import ast
import os
import sys
import textwrap
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from detectors.lineage_extract import (classify_guard, extract_lineage,
                                       match_equation, resolve,
                                       to_detector4_lineage)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
LOADER = os.path.join(ROOT, "src", "data", "physionet2019.py")
VARS = ["SBP", "DBP", "MAP", "pH", "HCO3", "pCO2", "PaO2", "SpO2", "SaO2"]


class TestEquationMatching(unittest.TestCase):
    def test_severinghaus_by_constant_and_cube_root(self):
        node = ast.parse("np.cbrt(23400.0 * s / (1.0 - s + 1e-8))")
        self.assertEqual(match_equation(node), "severinghaus")

    def test_henderson_hasselbalch_by_its_constants(self):
        node = ast.parse("6.1 + np.log10(hco3 / (0.0307 * pco2))")
        self.assertEqual(match_equation(node), "henderson_hasselbalch")

    def test_base_excess_conversion(self):
        node = ast.parse("24.0 + 0.5 * be")
        self.assertEqual(match_equation(node), "base_excess_to_hco3")

    def test_unrelated_arithmetic_matches_nothing(self):
        self.assertIsNone(match_equation(ast.parse("x * 2.0 + 7.0")))


class TestGuardClassification(unittest.TestCase):
    def test_absent_column_test(self):
        t = ast.parse('"HCO3" not in df.columns').body[0].value
        self.assertIn(("column_absent", "HCO3"), classify_guard(t))

    def test_present_column_test(self):
        t = ast.parse('"SaO2" in df.columns').body[0].value
        self.assertIn(("column_present", "SaO2"), classify_guard(t))

    def test_membership_in_a_plain_list_is_not_a_column_guard(self):
        t = ast.parse('"HCO3" in some_list').body[0].value
        self.assertEqual(classify_guard(t), [("unknown", None)])

    def test_unrecognised_predicate_is_unknown_not_absent(self):
        t = ast.parse("x > 3").body[0].value
        self.assertEqual(classify_guard(t), [("unknown", None)])


@unittest.skipUnless(os.path.exists(LOADER), "loader not present")
class TestAgainstTheRealLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = extract_lineage(LOADER, VARS)

    def test_finds_the_pao2_severinghaus_inversion(self):
        self.assertIn("PaO2", self.raw)
        self.assertEqual(self.raw["PaO2"]["equation"], "severinghaus")

    def test_finds_the_hco3_base_excess_derivation_too(self):
        """The hand-written table missed this one entirely."""
        self.assertIn("HCO3", self.raw)
        self.assertEqual(self.raw["HCO3"]["equation"], "base_excess_to_hco3")

    def test_hco3_derivation_is_guarded_on_the_column_being_absent(self):
        self.assertIn(("column_absent", "HCO3"), self.raw["HCO3"]["guards"])

    def test_physionet_columns_make_the_hco3_branch_unreachable(self):
        """Every PhysioNet PSV has an HCO3 column, so the branch never runs and
        HCO3 is measured in practice -- what the hand-written table said."""
        res = resolve(self.raw, ["HCO3", "BaseExcess", "SaO2", "pH", "PaCO2"])
        self.assertFalse(res["HCO3"]["reachable"])
        self.assertIn("IS present", res["HCO3"]["reason"])

    def test_pao2_derivation_stays_reachable(self):
        res = resolve(self.raw, ["HCO3", "BaseExcess", "SaO2"])
        self.assertTrue(res["PaO2"]["reachable"])

    def test_extracted_table_matches_the_hand_written_one(self):
        res = resolve(self.raw, ["HCO3", "BaseExcess", "SaO2", "pH", "PaCO2"])
        table = to_detector4_lineage(res, VARS)
        self.assertEqual(table["HCO3"], "measured")
        self.assertEqual(table["MAP"], "measured")
        self.assertEqual(table["pH"], "measured")
        self.assertEqual(table["PaO2"][0], "derived")
        self.assertEqual(table["PaO2"][1], "severinghaus")

    def test_a_dataset_without_hco3_would_flip_the_verdict(self):
        """Guard reasoning is data-dependent by design: the same code yields a
        different lineage on a dataset lacking the column."""
        res = resolve(self.raw, ["BaseExcess", "SaO2"])
        self.assertTrue(res["HCO3"]["reachable"])
        table = to_detector4_lineage(res, VARS)
        self.assertEqual(table["HCO3"][0], "derived")


class TestSyntheticCircularity(unittest.TestCase):
    def test_detects_an_unguarded_derivation_in_arbitrary_source(self):
        src = textwrap.dedent('''
            def load(df):
                ph_idx = VAR_TO_IDX["pH"]
                ts[:, ph_idx] = 6.1 + np.log10(hco3 / (0.0307 * pco2))
                return ts
        ''')
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_tmp_lineage_fixture.py")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(src)
        try:
            got = extract_lineage(p, VARS)
            self.assertEqual(got["pH"]["equation"], "henderson_hasselbalch")
            self.assertTrue(resolve(got, ["pH"])["pH"]["reachable"])
        finally:
            os.remove(p)


if __name__ == "__main__":
    unittest.main()
