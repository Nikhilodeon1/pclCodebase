"""Guard the fixed-shared-bounds assumption detector 2's external design rests on.

Detector 2 external validation uses sample["values"] rather than raw_ts, on the
grounds that normalization here is a FIXED affine map by clinical plausibility
bounds -- identical for every dataset -- rather than a data-fitted transform.
MinMaxNormalizer.fit() is a no-op and the bounds come from PLAUS.

If someone later makes normalization data-driven, or edits one loader's scaling
and not the others, that reasoning silently breaks and the cross-site
distribution difference detector 2 probes for would be partly normalized away.
These tests fail loudly if that happens. This is what protects the assumption --
not reimplementing the external path on raw_ts.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.data.preprocessing import MinMaxNormalizer, preprocess_timeseries
from src.data.variables import CANONICAL_VARIABLES, PLAUS

LOADERS = ["physionet2019.py", "mimic4.py", "eicu.py"]
SRC_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src", "data")


def _raw(seed=0, hours=48):
    rng = np.random.default_rng(seed)
    return rng.uniform(30, 120, (hours, len(CANONICAL_VARIABLES)))


class TestFitIsANoOp(unittest.TestCase):
    def test_fitting_on_different_data_does_not_change_the_transform(self):
        x = _raw(0)
        a, b = MinMaxNormalizer(), MinMaxNormalizer()
        a.fit([_raw(1)])
        b.fit([_raw(2) * 10.0])
        np.testing.assert_allclose(a.transform(x), b.transform(x))

    def test_unfitted_matches_fitted(self):
        x = _raw(3)
        fitted = MinMaxNormalizer()
        fitted.fit([_raw(4)])
        np.testing.assert_allclose(MinMaxNormalizer().transform(x),
                                   fitted.transform(x))


class TestBoundsAreThePlausibilityConstants(unittest.TestCase):
    def test_transform_matches_the_plaus_formula_exactly(self):
        x = _raw(5)
        got = MinMaxNormalizer().transform(x)
        for col, var in enumerate(CANONICAL_VARIABLES):
            lo, hi = PLAUS[var]
            np.testing.assert_allclose(got[:, col],
                                       (x[:, col] - lo) / (hi - lo + 1e-8))

    def test_every_canonical_variable_has_bounds(self):
        missing = [v for v in CANONICAL_VARIABLES if v not in PLAUS]
        self.assertEqual(missing, [], f"variables without PLAUS bounds: {missing}")

    def test_preprocess_fallback_uses_the_same_bounds(self):
        """preprocess_timeseries(normalizer=None) has its own inline scaling.
        It must agree with MinMaxNormalizer or the two paths diverge."""
        raw = _raw(6)
        with_norm = preprocess_timeseries(raw, MinMaxNormalizer())["values"]
        without = preprocess_timeseries(raw, None)["values"]
        np.testing.assert_allclose(with_norm, without)


class TestAllLoadersShareTheSameScaling(unittest.TestCase):
    def test_every_loader_uses_MinMaxNormalizer(self):
        for name in LOADERS:
            path = os.path.join(SRC_DATA, name)
            src = open(path, encoding="utf-8").read()
            self.assertIn("MinMaxNormalizer", src,
                          f"{name} does not use MinMaxNormalizer; detector 2's "
                          "shared-scaling assumption no longer holds")

    def test_no_loader_defines_its_own_bounds(self):
        """A loader computing its own min/max would reintroduce per-dataset
        scaling, which is exactly what must not happen."""
        for name in LOADERS:
            src = open(os.path.join(SRC_DATA, name), encoding="utf-8").read()
            for banned in ("np.nanmin(", "np.nanmax(", ".min(axis=0)",
                           ".max(axis=0)"):
                self.assertNotIn(
                    banned, src,
                    f"{name} appears to derive scaling bounds from data "
                    f"({banned}); detector 2 assumes fixed shared bounds")


if __name__ == "__main__":
    unittest.main()
