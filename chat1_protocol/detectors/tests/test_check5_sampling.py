"""Check 5's stay sample must be a random sample, not an alphabetical prefix.

PhysioNet filenames are patient IDs, so `sorted(listdir)[:n]` is a systematic
slice of the site rather than a sample of it, and it produces the same stays on
every run -- no sampling variance to report.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from detectors.checks.check5_missingness_scale import ROOT, sample_files

HAVE_DATA = os.path.isdir(os.path.join(ROOT, "training_setA"))


@unittest.skipUnless(HAVE_DATA, "PhysioNet data not present")
class TestSampling(unittest.TestCase):
    N = 50

    def test_returns_the_requested_sizes(self):
        fa, fb = sample_files(seed=0, n=self.N)
        self.assertEqual(len(fa), 2 * self.N)
        self.assertEqual(len(fb), self.N)

    def test_different_seeds_give_different_samples(self):
        a0, _ = sample_files(seed=0, n=self.N)
        a1, _ = sample_files(seed=1, n=self.N)
        self.assertNotEqual(a0, a1,
                            "sampling must vary with the seed, otherwise the "
                            "multi-seed sweep measures nothing")

    def test_same_seed_is_reproducible(self):
        self.assertEqual(sample_files(seed=3, n=self.N),
                         sample_files(seed=3, n=self.N))

    def test_legacy_is_the_alphabetical_prefix(self):
        fa, fb = sample_files(seed=0, n=self.N, legacy=True)
        self.assertEqual([os.path.basename(p) for p in fa],
                         sorted(os.path.basename(p) for p in fa))
        self.assertEqual([os.path.basename(p) for p in fb],
                         sorted(os.path.basename(p) for p in fb))

    def test_legacy_ignores_the_seed(self):
        self.assertEqual(sample_files(seed=0, n=self.N, legacy=True),
                         sample_files(seed=9, n=self.N, legacy=True))

    def test_random_sample_is_not_the_prefix(self):
        rand, _ = sample_files(seed=0, n=self.N)
        leg, _ = sample_files(seed=0, n=self.N, legacy=True)
        self.assertNotEqual(rand, leg)

    def test_no_stay_is_sampled_twice(self):
        fa, fb = sample_files(seed=0, n=self.N)
        self.assertEqual(len(set(fa)), len(fa))
        self.assertEqual(len(set(fb)), len(fb))


if __name__ == "__main__":
    unittest.main()
