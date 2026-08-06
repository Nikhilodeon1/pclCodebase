# Open items

Deferred deliberately. Each says why it is not urgent and what breaks if it is
forgotten.

## 1. CLAUDE.md's preprocessing description does not match the code

**Status:** filed, not fixed. Low priority relative to remaining detector work.

CLAUDE.md's Data Pipeline Rules state:

> Median-binning, forward-filling (6h limit), and unit variance normalization
> computed only on training data.

The code does something different in two respects:

* **Normalization is fixed-bounds min-max, not unit variance, and it is not
  computed from data at all.** `MinMaxNormalizer.fit()` is a no-op
  (`src/data/preprocessing.py:133`); `transform` rescales each variable by the
  fixed clinical plausibility bounds in `PLAUS` (`src/data/variables.py`). The
  `normalizer.fit(all_raw_ts)` calls in all three loaders are vestigial. The
  `normalizer=None` fallback inside `preprocess_timeseries` applies the same
  PLAUS formula inline.
* Because the bounds are constants, "computed only on training data" is
  vacuous here — there is nothing fitted to leak. The rule is satisfied, but
  not for the reason the document gives.

**Why this matters and why it is filed rather than ignored:** the paper's
methods section is the kind of thing that gets drafted from the project
document rather than from the source. Anyone doing that would describe unit
variance normalization fitted on training data, which is not what produced any
number in this repository.

**Do not "fix" this by changing the code.** The fixed-bounds behaviour is what
every result so far was measured under, and detector 2's external design
explicitly depends on it (see `rebootpcl/tests/test_normalizer_bounds.py`).
Fix the documentation to match the code.

## 2. Should every real-data detector report an uncertainty measure by default?

**Status:** open design decision. Deferred until detectors 3/4 are done, but
must be settled BEFORE the results table is finalized.

Detector 1 now has a bootstrap CI and detector 5 has across-seed variance, but
both were computed because a specific number looked close, not as a matter of
course. The MIMIC exposure (kappa 0.651, 95% CI [0.528, 0.776], P(flag) = 0.216)
would not have been found without deliberately going to look for it.

The question is whether a verdict without an interval should be reportable at
all for detectors that consume real data. Detectors 3 and 4 are deterministic
static analysis and are excluded either way — there is nothing to resample.

Arguments for making it default: the two times uncertainty was measured, it
changed how the result had to be described. Against: it is not free for the
detectors that require training runs, and an interval on a verdict that is not
close adds noise to the table.
