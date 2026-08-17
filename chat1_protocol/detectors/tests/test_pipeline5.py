"""Unit tests for the detector-5 external testbed, on synthetic arrays.

No dataset needed: these pin the ablation and component mechanics, which is
where a silent error would invalidate the external result.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.data.variables import VAR_TO_IDX
from detectors.external.pipeline5 import (ablate, split_arrays, stay_components,
                                          to_stays, TERMS)


def make_stay(n=48, seed=0):
    """A stay with all three terms observable and physiologically consistent."""
    rng = np.random.default_rng(seed)
    ts = np.full((n, len(VAR_TO_IDX)), np.nan)
    sbp = rng.uniform(100, 140, n)
    dbp = rng.uniform(55, 85, n)
    ts[:, VAR_TO_IDX["SBP"]] = sbp
    ts[:, VAR_TO_IDX["DBP"]] = dbp
    ts[:, VAR_TO_IDX["MAP"]] = dbp + (sbp - dbp) / 3.0
    ts[:, VAR_TO_IDX["HCO3"]] = rng.uniform(20, 28, n)
    ts[:, VAR_TO_IDX["pCO2"]] = rng.uniform(35, 45, n)
    ts[:, VAR_TO_IDX["pH"]] = 7.4
    ts[:, VAR_TO_IDX["PaO2"]] = rng.uniform(70, 110, n)
    ts[:, VAR_TO_IDX["SpO2"]] = rng.uniform(94, 99, n)
    return ts


class TestComponents(unittest.TestCase):
    def test_all_three_terms_computable_on_a_complete_stay(self):
        c = stay_components(make_stay())
        self.assertEqual(sorted(c), sorted(TERMS))

    def test_exact_map_identity_gives_zero_residual(self):
        c = stay_components(make_stay())
        self.assertAlmostEqual(c["MAP"], 0.0, places=12)

    def test_a_term_with_no_observations_is_absent_not_zero(self):
        ts = make_stay()
        ts[:, VAR_TO_IDX["HCO3"]] = np.nan
        c = stay_components(ts)
        self.assertNotIn("HH", c)
        self.assertIn("MAP", c)

    def test_stay_with_nothing_observable_is_dropped(self):
        self.assertEqual(to_stays([np.full((48, len(VAR_TO_IDX)), np.nan)]), [])


class TestAblation(unittest.TestCase):
    def test_removes_roughly_the_requested_fraction(self):
        arrays = [make_stay(seed=i) for i in range(40)]
        out = ablate(arrays, "HCO3", p=0.75, seed=0)
        idx = VAR_TO_IDX["HCO3"]
        before = sum(int((~np.isnan(a[:, idx])).sum()) for a in arrays)
        after = sum(int((~np.isnan(a[:, idx])).sum()) for a in out)
        self.assertAlmostEqual(after / before, 0.25, delta=0.02)

    def test_does_not_modify_the_input_arrays(self):
        arrays = [make_stay(seed=1)]
        idx = VAR_TO_IDX["HCO3"]
        before = int((~np.isnan(arrays[0][:, idx])).sum())
        ablate(arrays, "HCO3", p=1.0, seed=0)
        self.assertEqual(int((~np.isnan(arrays[0][:, idx])).sum()), before,
                         "ablation must not mutate its input")

    def test_leaves_other_variables_untouched(self):
        arrays = [make_stay(seed=2)]
        out = ablate(arrays, "HCO3", p=1.0, seed=0)
        for v in ("SBP", "DBP", "MAP", "pH", "pCO2", "SpO2", "PaO2"):
            np.testing.assert_array_equal(
                arrays[0][:, VAR_TO_IDX[v]], out[0][:, VAR_TO_IDX[v]])

    def test_surviving_values_are_the_original_ones(self):
        """Physiology must be held fixed: ablation removes observations, it
        never perturbs the values that remain."""
        arrays = [make_stay(seed=3)]
        out = ablate(arrays, "HCO3", p=0.5, seed=0)
        idx = VAR_TO_IDX["HCO3"]
        kept = ~np.isnan(out[0][:, idx])
        np.testing.assert_array_equal(out[0][kept, idx], arrays[0][kept, idx])

    def test_full_ablation_removes_the_term_entirely(self):
        arrays = [make_stay(seed=4)]
        out = ablate(arrays, "HCO3", p=1.0, seed=0)
        self.assertNotIn("HH", stay_components(out[0]))


class TestSplit(unittest.TestCase):
    def test_halves_are_disjoint_and_cover_everything(self):
        arrays = [make_stay(seed=i) for i in range(21)]
        a, b = split_arrays(arrays, seed=0)
        self.assertEqual(len(a) + len(b), len(arrays))
        ids_a = {id(x) for x in a}
        ids_b = {id(x) for x in b}
        self.assertEqual(ids_a & ids_b, set())

    def test_seed_changes_the_split(self):
        arrays = [make_stay(seed=i) for i in range(20)]
        a0, _ = split_arrays(arrays, seed=0)
        a1, _ = split_arrays(arrays, seed=1)
        self.assertNotEqual([id(x) for x in a0], [id(x) for x in a1])


if __name__ == "__main__":
    unittest.main()
