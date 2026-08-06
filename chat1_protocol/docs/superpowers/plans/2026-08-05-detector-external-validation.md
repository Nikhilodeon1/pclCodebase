# Confound-Detector External Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the circularity gap that blocks submission: prove each of the five confound detectors works on pipelines they were not built against, quantify their false-positive rate on legitimate variation, and replace single-seed point estimates with multi-seed distributions.

**Architecture:** The five detectors currently print to stdout and return nothing structured, so no downstream experiment can consume their verdicts. The spine of this plan is a shared harness (`rebootpcl/harness.py`) that every detector reports into as `Case` records. Once detectors return structured verdicts, the four downstream deliverables (multi-seed variance, external-pipeline validation, false-positive rate over legitimate variants, baseline comparison) are all loops over the same interface rather than four bespoke scripts. External validation splits by detector type: checks 3 and 4 are static/lineage analysis and are validated against *other codebases*, while checks 1, 2 and 5 are data/model analysis and are validated against a second, independently-built clinical pipeline (`rebootpcl/external/pipeline.py`).

**Tech Stack:** Python 3, numpy, pandas, torch (CPU only), scikit-learn 1.8 (already installed), stdlib `unittest` and `ast`. No new dependencies. No GPU. No datasets beyond PhysioNet 2019, MIMIC-IV demo, eICU demo.

## Global Constraints

- **CPU only.** No GPU spend. If any task appears to need GPU, stop and flag before proceeding.
- **No new datasets.** PhysioNet 2019, MIMIC-IV demo, eICU demo only. Everything already lives under `chat1_protocol/data/`.
- **Budget ~$50 total, $0 spent.** Every task in this plan is local and free. Flag before anything that is not.
- **No single-seed results reported as final.** Any new experimental number ships with >= 5 seeds and a reported spread.
- **No new pip installs.** `pytest` is NOT installed in the shared venv. Tests use stdlib `unittest`. Gradient boosting uses `sklearn.ensemble.HistGradientBoostingClassifier`, not lightgbm.
- **`chat1_protocol/data/` is a Windows directory junction to `../data`.** Never `rm -rf` it. Remove only with `cmd //c rmdir "chat1_protocol\data"`. Never `git add` a path that walks into it.
- **Interpreter:** `C:/Users/nikhi/Codes/pclCodebase/venv/Scripts/python.exe`. All commands below assume cwd `C:/Users/nikhi/Codes/pclCodebase/chat1_protocol`.
- **Checks 1 and 2 require `PCL_TEST_MODE=1`.**
- **`load_physionet2019(fraction=...)` subsamples the file list before reading.** Always pass a fraction; loading all then slicing reads ~40k PSVs (~25 min).
- **Long runs exceed the 2-minute tool timeout.** Use `nohup ... > out.log 2>&1 &` and poll.

---

## File Structure

**New:**
- `rebootpcl/harness.py` — `Case`, `confusion()`, `seed_sweep()`, `fmt_matrix()`. Shared result vocabulary. One responsibility: turning detector verdicts into confusion counts and tables.
- `rebootpcl/run_all.py` — reproduces the five-row headline table end to end. This is the pending "runner".
- `rebootpcl/external/__init__.py`
- `rebootpcl/external/pipeline.py` — the second, independent clinical pipeline (logistic + histogram gradient boosting for sepsis; small dense autoencoder to host check 2's pretraining probe).
- `rebootpcl/external/run_external.py` — runs checks 1/2/5 against the external pipeline and emits the PCL-vs-external comparison table.
- `rebootpcl/external/repos.py` — declares the third-party repositories checks 3 and 4 are run against, and the manually-established ground truth for each.
- `rebootpcl/specificity/variants.py` — the catalogue of legitimate, non-confounded pipeline variants.
- `rebootpcl/specificity/run_specificity.py` — every detector against every clean variant; emits the FP-rate table.
- `rebootpcl/standard/baseline_checks.py` — k-fold CV, single train/test split sanity check, plain external validation without confound auditing.
- `rebootpcl/standard/run_baselines.py` — baseline detection rate vs. detector detection rate, per failure mode.
- `rebootpcl/lineage_extract.py` — extracts the data-lineage table for check 4 from loader source instead of hand-authoring it.
- `rebootpcl/fixtures/sweep_SYNTH_*.py` — synthetic sweep fixtures giving check 3 positives and negatives beyond the single historical bug.
- `rebootpcl/tests/` — stdlib `unittest` tests, one module per new module.
- `rebootpcl/results/*.json` — machine-readable outputs, one per experiment.
- `.gitignore` (repo root) — one added line.

**Modified:**
- `rebootpcl/checks/check1_label_shift.py` — replace tautological negative control; expose `run(seed) -> list[Case]`.
- `rebootpcl/checks/check2_pretrain_leakage.py` — vary the data split seed; expose `run(...) -> list[Case]`; correct the t threshold.
- `rebootpcl/checks/check3_selection_audit.py` — expose `run(paths) -> list[Case]`.
- `rebootpcl/checks/check4_circularity.py` — expose `run(lineage) -> list[Case]`; consume extracted lineage.
- `rebootpcl/checks/check5_missingness_scale.py` — expose `run(seed, n) -> list[Case]`; randomize the positive-arm file sample.

---

## Phase 0 — Repository safety and correctness blockers

These precede everything. Task 1 prevents a 455 MB mis-commit. Tasks 2 and 3 fix defects that would make any downstream measurement meaningless.

### Task 1: Make the repository safe to commit, then commit the existing work

**Files:**
- Modify: `C:/Users/nikhi/Codes/pclCodebase/.gitignore`
- Commit: `chat1_protocol/rebootpcl/checks/*`, `chat1_protocol/rebootpcl/fixtures/*`, `chat1_protocol/README.md`, `chat1_protocol/src/`, `chat1_protocol/config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a clean `git status` in which `chat1_protocol/data/` is invisible to git.

**Background the implementer needs:** `chat1_protocol/data` is a Windows directory junction pointing at the shared 455 MB dataset. Git follows it. The root `.gitignore` contains `/data/`, which is *anchored to the repository root* and therefore does not match `chat1_protocol/data/`. `git status --porcelain --untracked-files=all chat1_protocol` currently reports 40,406 untracked files. A naive `git add chat1_protocol/` stages the entire dataset.

- [ ] **Step 1: Confirm the hazard is real before changing anything**

```bash
cd /c/Users/nikhi/Codes/pclCodebase && git status --porcelain --untracked-files=all chat1_protocol | grep -c "chat1_protocol/data/"
```

Expected: a number in the tens of thousands (was 40406). If it prints `0`, the ignore rule already exists — skip to Step 4.

- [ ] **Step 2: Add the ignore rule**

Append to `C:/Users/nikhi/Codes/pclCodebase/.gitignore`:

```
chat1_protocol/data/
chat1_protocol/rebootpcl/results/
chat1_protocol/**/__pycache__/
```

- [ ] **Step 3: Verify git no longer walks the junction**

```bash
cd /c/Users/nikhi/Codes/pclCodebase && git status --porcelain --untracked-files=all chat1_protocol | grep -c "chat1_protocol/data/"
```

Expected: `0`.

- [ ] **Step 4: Verify what will actually be staged, before staging it**

```bash
cd /c/Users/nikhi/Codes/pclCodebase && git add --dry-run chat1_protocol 2>&1 | wc -l
```

Expected: a few dozen lines, all under `chat1_protocol/` and none under `chat1_protocol/data/`. If the count is large, STOP and re-check Step 2.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/nikhi/Codes/pclCodebase && git add .gitignore chat1_protocol && git commit -m "feat: add five confound detectors with validation fixtures

Five automated detectors for confounds that manufacture cross-hospital
gains: label-definition shift, pretraining leakage, OOD-contaminated
selection, circular derived constraints, and missingness/scale artifacts.
Each ships with a positive case and a false-positive control.

Ignore chat1_protocol/data/, which is a junction to the shared dataset.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Replace check 1's tautological negative control

**Files:**
- Modify: `rebootpcl/checks/check1_label_shift.py:106-111`
- Test: `rebootpcl/tests/test_check1_control.py`

**Interfaces:**
- Consumes: `src.data.sofa_sepsis.eicu_sofa_sepsis_labels(data_path, patients_df, mode=...)`, which accepts `mode="window"` and `mode="single"` — two valid Sepsis-3 operationalizations of the suspicion window.
- Produces: `check1_label_shift.build_scenarios(ids, icd, sofa_single, sofa_window, s1, s2, audit) -> dict[str, tuple]`, mapping scenario name to `(site1_labels, site2_labels, audit_a, audit_b, expected_flag)`.

**Background: the defect.** The current negative controls call

```python
diagnose("NEGATIVE CONTROL: both sites = ICD", icd[s1], icd[s2], icd[audit], icd[audit])
```

The two audit arguments are *the same array object*. Cohen's kappa is therefore 1.0 by construction, not by measurement. Both negative controls have this property, so the reported `TN=2` is vacuous — it demonstrates that `kappa(x, x) == 1`, which is arithmetic, not evidence about the detector. A reviewer will find this immediately. Note that the *prevalence-ratio* control is unaffected: `icd[s1]` vs `icd[s2]` are genuinely different halves, so the "negative control hits 1.37, near a 1.5 threshold" finding survives and stays in the paper.

**The fix.** A negative control must be two genuinely distinct label computations that a reasonable person would accept as the same criterion. `mode="window"` vs `mode="single"` SOFA are exactly that: both are valid Sepsis-3, differing only in how the suspicion window is scored. If kappa between them is high, the detector correctly stays silent on legitimate implementation variation. If it is low, that is a real and reportable finding — legitimate operationalization choices shift labels as much as ICD-vs-SOFA does — and it belongs in Phase 3's specificity analysis rather than being hidden.

- [ ] **Step 1: Write the failing test**

Create `rebootpcl/tests/test_check1_control.py`:

```python
import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.checks.check1_label_shift import build_scenarios


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
                f"scenario {name!r} compares an audit array to itself; "
                "its kappa is 1.0 by construction, not by measurement")

    def test_a_negative_control_exists_and_uses_two_real_labelers(self):
        negs = {n: v for n, v in self.scenarios().items() if v[4] is False}
        self.assertGreaterEqual(len(negs), 1)

    def test_positive_scenario_is_icd_vs_sofa(self):
        pos = [n for n, v in self.scenarios().items() if v[4] is True]
        self.assertEqual(len(pos), 1)
        self.assertIn("ICD", pos[0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && ../venv/Scripts/python.exe -m unittest rebootpcl.tests.test_check1_control -v
```

Expected: FAIL with `ImportError: cannot import name 'build_scenarios'`.

- [ ] **Step 3: Implement `build_scenarios`**

In `rebootpcl/checks/check1_label_shift.py`, add after `diagnose`:

```python
def build_scenarios(ids, icd, sofa_single, sofa_window, s1, s2, audit):
    """Scenario name -> (site1_labels, site2_labels, audit_a, audit_b, expected_flag).

    The audit pair must always be two INDEPENDENTLY COMPUTED label arrays over
    the same patients. Passing one array twice makes kappa 1.0 by construction
    and proves nothing about the detector, which is what the previous negative
    control did.

    Negative control rationale: SOFA "window" and "single" are both valid
    Sepsis-3 operationalizations, differing only in how the suspicion window is
    scored. A detector that flags this pair is flagging legitimate
    implementation variation, which is a false positive.
    """
    return {
        "positive (ICD vs SOFA)": (
            icd[s1], sofa_single[s2], icd[audit], sofa_single[audit], True),
        "negative (SOFA window vs single)": (
            sofa_window[s1], sofa_single[s2],
            sofa_window[audit], sofa_single[audit], False),
        "negative (ICD vs ICD, disjoint halves)": (
            icd[s1], icd[s2], icd[audit[::2]], icd[audit[1::2]][:len(audit[::2])],
            False),
    }
```

Note on the third scenario: it compares ICD labels on two *disjoint* patient subsets, so the arrays are genuinely different data. Kappa there measures chance agreement between independent draws of the same criterion and is expected to be low-to-moderate — it is included to show that kappa on non-paired patients is NOT a valid diagnostic, which is precisely why the audit subset must hold patients fixed. If it flags, that is the expected and documented behaviour of a misapplied diagnostic; record it in the plan's results notes rather than treating it as a detector failure. If this scenario proves confusing to report, drop it and keep the window-vs-single control alone.

- [ ] **Step 4: Rewire `main()` to use `build_scenarios`**

Replace lines 101-111 of `check1_label_shift.py` with:

```python
    sofa_win_map, _ = eicu_sofa_sepsis_labels(EICU_DIR, pats, mode="window")
    sofa_window = np.array([int(sofa_win_map.get(int(i), 0)) for i in ids])

    results, exp = {}, {}
    for name, (a, b, aa, ab, e) in build_scenarios(
            ids, icd, sofa, sofa_window, s1, s2, audit).items():
        results[name] = diagnose(name, a, b, aa, ab)
        exp[name] = e
```

and delete the now-dead hardcoded `exp = {...}` dict on lines 114-115.

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && ../venv/Scripts/python.exe -m unittest rebootpcl.tests.test_check1_control -v
```

Expected: PASS, 3 tests.

- [ ] **Step 6: Re-run the detector and record the corrected confusion matrix**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && PCL_TEST_MODE=1 nohup ../venv/Scripts/python.exe rebootpcl/checks/check1_label_shift.py > rebootpcl/check1_fixedcontrol.out 2>&1 &
```

Poll the log. Expected: the positive still flags at kappa ~0.358. The window-vs-single control's kappa is a NEW measurement — record whatever it is. **If it comes out below 0.60 the control flags, the detector has a real false positive, and that must be reported, not tuned away.** Do not adjust `KAPPA_FLAG` to make the control pass.

- [ ] **Step 7: Commit**

```bash
cd /c/Users/nikhi/Codes/pclCodebase && git add chat1_protocol/rebootpcl/checks/check1_label_shift.py chat1_protocol/rebootpcl/tests/test_check1_control.py chat1_protocol/rebootpcl/check1_fixedcontrol.out && git commit -m "fix: replace check 1's tautological negative control

The previous negative controls passed the same audit array as both
arguments to Cohen's kappa, making the result 1.0 by construction. The
reported TN=2 measured arithmetic, not detector specificity. Controls now
compare SOFA window-mode against SOFA single-mode: two genuinely distinct
but equally valid Sepsis-3 operationalizations.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Unpin check 2's data split from a single seed

**Files:**
- Modify: `rebootpcl/checks/check2_pretrain_leakage.py:129`, `:168`
- Test: `rebootpcl/tests/test_check2_seeding.py`

**Interfaces:**
- Consumes: `build(stays, seed)` (already exists, already takes a seed).
- Produces: `run_one(...)` unchanged; `main()` gains `--data-seeds`; `flag_from_relative(rel, alpha=0.05) -> tuple[bool, float, float]` returning `(flagged, t, critical_t)`.

**Background: two defects.**

1. **Data split is pinned.** `main()` calls `build(args.stays, seed=0)` once at line 129, outside the seed loop. Only model initialization and the leak subset vary across "seeds". Site membership, the source/target sample, and the probe slice are identical in every run. The reported spread therefore excludes split variance, so "3 seeds" is a 3-model-seed claim, not a 3-seed claim. Given this project's explicit concern about single-seed results, that is exactly the wrong variance to omit.

2. **The t threshold is wrong for the sample size.** Line 168 flags on `t <= -2.0`. With 3 seeds, df = 2, and the one-sided 5% critical value is 2.920, not 2.0. The reported `t = -3.51` at 5% leakage clears 2.920, but `-2.0` as a fixed threshold does not correspond to any stated alpha and will admit non-significant results at small n. Additionally the `or len(rel) < 3` clause flags on sign agreement alone whenever fewer than 3 seeds are run, which would silently produce unsupported positives.

- [ ] **Step 1: Write the failing test**

Create `rebootpcl/tests/test_check2_seeding.py`:

```python
import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.checks.check2_pretrain_leakage import flag_from_relative


class TestFlagThreshold(unittest.TestCase):
    def test_uses_df_appropriate_critical_value_not_fixed_two(self):
        # t = -2.5 with n=3 (df=2) is NOT significant at one-sided 5%
        # (critical 2.920). A fixed -2.0 threshold would wrongly flag it.
        rel = np.array([-0.10, -0.09, -0.02])
        flagged, t, crit = flag_from_relative(rel)
        self.assertGreater(crit, 2.9, "df=2 critical value must exceed 2.9")
        if t > -crit:
            self.assertFalse(flagged)

    def test_clear_effect_at_five_seeds_flags(self):
        rel = np.array([-0.20, -0.18, -0.22, -0.19, -0.21])
        flagged, t, crit = flag_from_relative(rel)
        self.assertTrue(flagged)
        self.assertLess(t, -crit)

    def test_requires_sign_agreement(self):
        rel = np.array([-0.30, -0.28, +0.25, -0.31, -0.29])
        flagged, _, _ = flag_from_relative(rel)
        self.assertFalse(flagged, "must not flag when seeds disagree in sign")

    def test_two_seeds_never_flag(self):
        rel = np.array([-0.40, -0.42])
        flagged, _, _ = flag_from_relative(rel)
        self.assertFalse(flagged, "n=2 is not enough evidence to flag")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && ../venv/Scripts/python.exe -m unittest rebootpcl.tests.test_check2_seeding -v
```

Expected: FAIL with `ImportError: cannot import name 'flag_from_relative'`.

- [ ] **Step 3: Implement `flag_from_relative`**

Add to `check2_pretrain_leakage.py`, above `main()`:

```python
# One-sided t critical values at alpha=0.05, indexed by degrees of freedom.
# Hardcoded so the check has no scipy dependency.
_T_CRIT_05 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943,
              7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812}


def flag_from_relative(rel):
    """Decide detection from per-seed relative deltas.

    Leakage must LOWER probe loss consistently: every seed negative, and the
    paired one-sided t statistic past the df-appropriate 5% critical value.
    A fixed threshold of -2.0 does not correspond to any alpha at small n
    (df=2 needs 2.920), so the critical value is looked up by df.

    Returns (flagged, t, critical_t).
    """
    rel = np.asarray(rel, float)
    n = len(rel)
    if n < 3:
        return False, 0.0, float("inf")
    df = n - 1
    crit = _T_CRIT_05.get(df, 1.645)   # large-df limit
    sd = float(rel.std(ddof=1))
    t = float(rel.mean() / (sd / np.sqrt(n))) if sd > 1e-12 else 0.0
    return bool(np.all(rel < 0) and t <= -crit), t, crit
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && ../venv/Scripts/python.exe -m unittest rebootpcl.tests.test_check2_seeding -v
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Vary the data split seed**

In `main()`, replace the single `build(...)` call and the seed loop. Replace lines 129-143 with:

```python
    print("NOTE: data split varies per seed — site sample and probe slice are "
          "re-drawn, so the reported spread includes split variance.")

    results = {frac: [] for frac in LEVELS}
    for s in range(args.seeds):
        seed = 42 + s
        # Re-draw the split for THIS seed. Previously build() was called once
        # with seed=0 outside this loop, so every "seed" shared one split and
        # split variance was excluded from the reported spread.
        src_ds, leak_ds, probe_ds = build(args.stays, seed=seed)
        if s == 0:
            print(f"source={len(src_ds.samples)}  leak pool={len(leak_ds.samples)}  "
                  f"probe={len(probe_ds.samples)} (probe never pretrained on)")
        for frac in LEVELS:
            l, n_leak = run_one(src_ds, leak_ds, probe_ds, frac, seed,
                                args.epochs, device)
            results[frac].append(l)

    for frac in LEVELS:
        losses = results[frac]
        print(f"  leakage {int(frac*100):>3}%  probe loss {np.mean(losses):.6f} "
              f"+/- {np.std(losses, ddof=1) if len(losses) > 1 else 0:.6f}"
              f"   {[round(x, 5) for x in losses]}")
```

Then delete the now-unused `src_ds, leak_ds, probe_ds = build(args.stays, seed=0)` line and its adjacent print.

**Important:** the probe slice now differs per seed, so probe losses are no longer comparable *across* seeds in absolute terms. This is fine and is exactly why the analysis is paired — the relative within-seed delta `(cur - base) / base` cancels the seed's baseline level. Do not add cross-seed absolute comparisons.

- [ ] **Step 6: Use `flag_from_relative` in the decision loop**

Replace lines 159-171 with:

```python
    detected = {}
    for frac in LEVELS:
        cur = np.array(results[frac])
        rel = (cur - base) / np.maximum(base, 1e-12)
        flagged, t, crit = flag_from_relative(rel)
        flag = bool(frac > 0 and flagged)
        detected[frac] = flag
        print(f"{int(frac*100):>8}%{float(cur.mean()):>13.6f}"
              f"{float(rel.mean())*100:>19.1f}%{t:>8.2f}"
              f"{str(bool(np.all(rel < 0))):>8}{str(flag):>10}"
              f"   (crit {-crit:.3f})")
```

- [ ] **Step 7: Re-run at 5 seeds and record the result**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && PCL_TEST_MODE=1 nohup ../venv/Scripts/python.exe rebootpcl/checks/check2_pretrain_leakage.py --stays 900 --seeds 5 --epochs 3 > rebootpcl/check2_5seed.out 2>&1 &
```

Runtime roughly 5 min at 3 seeds, so budget ~9 min at 5. Poll the log.

Expected: 0% leakage still does not flag. The detection floor may RISE above 5% now that split variance is included and the threshold is correct. **If it does, that is the honest number and it replaces the 5% claim in the paper.** Record it either way.

- [ ] **Step 8: Commit**

```bash
cd /c/Users/nikhi/Codes/pclCodebase && git add chat1_protocol/rebootpcl/checks/check2_pretrain_leakage.py chat1_protocol/rebootpcl/tests/test_check2_seeding.py chat1_protocol/rebootpcl/check2_5seed.out && git commit -m "fix: include split variance and a df-correct threshold in check 2

build() was called once outside the seed loop, so every seed shared one
data split and the reported spread excluded split variance. The flag
threshold was a fixed t <= -2.0, which corresponds to no alpha at df=2
(one-sided 5% needs 2.920). Both corrected; detection floor re-measured.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 1 — The shared harness and the headline-table runner

### Task 4: Build the result harness

**Files:**
- Create: `rebootpcl/harness.py`
- Test: `rebootpcl/tests/test_harness.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Case(name: str, flagged: bool, expected: bool, stats: dict)` — a NamedTuple.
  - `confusion(cases) -> dict` with integer keys `"TP"`, `"FP"`, `"FN"`, `"TN"`.
  - `rates(counts) -> dict` with float keys `"precision"`, `"recall"`, `"fpr"`; `float("nan")` when undefined.
  - `seed_sweep(run_fn, seeds) -> dict[str, dict]` mapping case name to `{"flag_rate": float, "n": int, "flags": list[bool], "expected": bool}`.
  - `fmt_matrix(label, counts) -> str` — one table row.

- [ ] **Step 1: Write the failing test**

Create `rebootpcl/tests/test_harness.py`:

```python
import os, sys, unittest
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.harness import Case, confusion, rates, seed_sweep, fmt_matrix


class TestConfusion(unittest.TestCase):
    def test_counts_all_four_cells(self):
        cases = [Case("a", True,  True,  {}),
                 Case("b", True,  False, {}),
                 Case("c", False, True,  {}),
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


class TestSeedSweep(unittest.TestCase):
    def test_reports_flag_rate_per_case_across_seeds(self):
        def run_fn(seed):
            # case "x" flags on even seeds only; case "y" always flags
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


class TestFmtMatrix(unittest.TestCase):
    def test_row_contains_label_and_all_counts(self):
        row = fmt_matrix("check1", {"TP": 1, "FP": 0, "FN": 0, "TN": 2})
        for token in ("check1", "1", "0", "2"):
            self.assertIn(token, row)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && ../venv/Scripts/python.exe -m unittest rebootpcl.tests.test_harness -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rebootpcl.harness'`.

- [ ] **Step 3: Implement the harness**

Create `rebootpcl/harness.py`:

```python
"""Shared result vocabulary for the confound detectors.

Every detector reports a list of Case records: one per scenario it evaluates,
each carrying what the detector decided and what the ground truth says. Keeping
this uniform is what lets the multi-seed, specificity, external-pipeline and
baseline-comparison experiments be loops over the same interface instead of
four bespoke scripts.
"""
from collections import namedtuple

Case = namedtuple("Case", "name flagged expected stats")


def confusion(cases):
    """Confusion counts over a list of Case records."""
    c = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for x in cases:
        if x.flagged and x.expected:
            c["TP"] += 1
        elif x.flagged and not x.expected:
            c["FP"] += 1
        elif not x.flagged and x.expected:
            c["FN"] += 1
        else:
            c["TN"] += 1
    return c


def rates(counts):
    """Precision, recall and false-positive rate. Undefined ratios are nan, not
    zero: a detector that never fires has undefined precision, not perfect or
    zero precision, and reporting 0.0 would understate it."""
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    nan = float("nan")
    return {
        "precision": tp / (tp + fp) if (tp + fp) else nan,
        "recall": tp / (tp + fn) if (tp + fn) else nan,
        "fpr": fp / (fp + tn) if (fp + tn) else nan,
    }


def seed_sweep(run_fn, seeds):
    """Run a detector across seeds; report per-case flag rate.

    run_fn(seed) must return the SAME set of case names for every seed —
    a case appearing under one seed and not another means the scenario set is
    seed-dependent, which makes the flag rate uninterpretable, so we raise.
    """
    acc, expected, names0 = {}, {}, None
    for s in seeds:
        cases = run_fn(s)
        names = {c.name for c in cases}
        if names0 is None:
            names0 = names
        elif names != names0:
            raise ValueError(
                f"case set changed at seed {s}: {names ^ names0}")
        for c in cases:
            acc.setdefault(c.name, []).append(bool(c.flagged))
            expected[c.name] = bool(c.expected)
    return {n: {"flags": v,
                "n": len(v),
                "flag_rate": sum(v) / len(v),
                "expected": expected[n]}
            for n, v in acc.items()}


def fmt_matrix(label, counts):
    r = rates(counts)
    return (f"{label:<28}TP={counts['TP']:<3}FP={counts['FP']:<3}"
            f"FN={counts['FN']:<3}TN={counts['TN']:<3}"
            f"prec={r['precision']:.2f} rec={r['recall']:.2f} "
            f"fpr={r['fpr']:.2f}")
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && ../venv/Scripts/python.exe -m unittest rebootpcl.tests.test_harness -v
```

Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/nikhi/Codes/pclCodebase && git add chat1_protocol/rebootpcl/harness.py chat1_protocol/rebootpcl/tests/ && git commit -m "feat: add shared Case/confusion harness for the detectors

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Expose `run()` on all five detectors

**Files:**
- Modify: all five of `rebootpcl/checks/check*.py`
- Test: `rebootpcl/tests/test_check_interfaces.py`

**Interfaces:**
- Consumes: `rebootpcl.harness.Case`.
- Produces, on each check module:
  - `check1_label_shift.run(seed=0, verbose=False) -> list[Case]`
  - `check2_pretrain_leakage.run(seed=0, stays=900, epochs=3, seeds=5, verbose=False) -> list[Case]`
  - `check3_selection_audit.run(paths=None, verbose=False) -> list[Case]`
  - `check4_circularity.run(lineage=None, expected=None, verbose=False) -> list[Case]`
  - `check5_missingness_scale.run(seed=0, n=1200, pair=None, verbose=False) -> list[Case]`

  `pair` for check 5 is `(name_a, files_a, name_b, files_b, expected_flag)` or `None` for the default PhysioNet A/B positive plus A-vs-A control. `paths` for check 3 is a list of `(path, expected_flag)`. `lineage`/`expected` for check 4 default to the module-level tables.

  Each `main()` becomes a thin wrapper: call `run(verbose=True)`, then print `fmt_matrix`.

- [ ] **Step 1: Write the failing test**

Create `rebootpcl/tests/test_check_interfaces.py`:

```python
import os, sys, unittest, importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.harness import Case

MODULES = ["check1_label_shift", "check2_pretrain_leakage",
           "check3_selection_audit", "check4_circularity",
           "check5_missingness_scale"]


class TestEveryCheckExposesRun(unittest.TestCase):
    def test_run_exists_and_is_callable(self):
        for name in MODULES:
            m = importlib.import_module(f"rebootpcl.checks.{name}")
            self.assertTrue(hasattr(m, "run"), f"{name} has no run()")
            self.assertTrue(callable(m.run))

    def test_fast_checks_return_cases(self):
        # checks 3 and 4 are static analysis: seconds, no data needed
        for name in ["check3_selection_audit", "check4_circularity"]:
            m = importlib.import_module(f"rebootpcl.checks.{name}")
            cases = m.run(verbose=False)
            self.assertGreater(len(cases), 0, f"{name}.run() returned nothing")
            for c in cases:
                self.assertIsInstance(c, Case)
                self.assertIsInstance(c.flagged, bool)
                self.assertIsInstance(c.expected, bool)
                self.assertIsInstance(c.stats, dict)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && ../venv/Scripts/python.exe -m unittest rebootpcl.tests.test_check_interfaces -v
```

Expected: FAIL — `check1_label_shift has no run()`.

- [ ] **Step 3: Add `run()` to check 4 (simplest; do this one first)**

In `check4_circularity.py`, add before `main()`:

```python
def run(lineage=None, expected=None, verbose=False):
    from rebootpcl.harness import Case
    lineage = LINEAGE if lineage is None else lineage
    expected = EXPECTED if expected is None else expected
    out = []
    for (ds, c), exp in expected.items():
        got, why = is_circular(ds, c, lineage=lineage)
        if verbose:
            print(f"{ds:<12}{c:<16}{str(got):>9}{str(exp):>10}   {why}")
        out.append(Case(f"{ds}/{c}", bool(got), bool(exp), {"reason": why}))
    return out
```

Then rewrite `main()`'s loop body to consume `run(verbose=True)` and print `fmt_matrix("check4", confusion(cases))`.

- [ ] **Step 4: Add `run()` to check 3**

In `check3_selection_audit.py`, add before `main()`:

```python
def run(paths=None, verbose=False):
    """paths: list of (path, expected_flag). Defaults to the two historical
    fixtures — the real buggy sweep and its correction."""
    from rebootpcl.harness import Case
    if paths is None:
        paths = [(os.path.join(FIX, "sweep_BUGGY.py"), True),
                 (os.path.join(FIX, "sweep_FIXED.py"), False)]
    out = []
    for p, exp in paths:
        hits = audit_file(p)          # existing function: list of offending sweeps
        flagged = bool(hits)
        if verbose:
            print(f"{os.path.basename(p):<28}flagged={flagged}  "
                  f"expected={exp}  sweeps={hits}")
        out.append(Case(os.path.basename(p), flagged, bool(exp),
                        {"sweeps": list(hits)}))
    return out
```

**Note for the implementer:** `audit_file` may not be the existing function's name — read `check3_selection_audit.py` and use whatever function currently walks one file's AST and returns the offending sweeps. If that logic is currently inline in `main()`, extract it to `audit_file(path) -> list[str]` first, keeping behaviour identical, and verify by re-running the detector and confirming it still reports the same two flagged sweeps (`ablation_lambda_sweep` and `ablation_constraint_subset`).

- [ ] **Step 5: Add `run()` to checks 1, 2, 5**

Each follows the same shape: the existing `main()` body computes scenario verdicts; move that into `run(...)`, return `Case` records, and reduce `main()` to printing. For check 1, `run(seed)` must thread `seed` into `np.random.default_rng(seed)` at line 93 in place of the hardcoded `0`. For check 5, `run(seed, n)` must likewise replace the hardcoded `default_rng(0)` at line 175, **and** replace the positive arm's `sorted(os.listdir(...))[:2*n]` deterministic prefix with a seeded random sample, so the positive case is not always the same alphabetically-first stays:

```python
    rng = np.random.default_rng(seed)
    all_a = [f for f in sorted(os.listdir(a_dir)) if f.endswith(".psv")]
    all_b = [f for f in sorted(os.listdir(b_dir)) if f.endswith(".psv")]
    fa = [os.path.join(a_dir, all_a[i])
          for i in rng.permutation(len(all_a))[:2 * n]]
    fb = [os.path.join(b_dir, all_b[i])
          for i in rng.permutation(len(all_b))[:n]]
```

For check 2, `run()` wraps the existing `main()` computation and emits one `Case` per leakage level: `expected=False` for 0%, `expected=True` for each non-zero level.

- [ ] **Step 6: Run the test to verify it passes**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && ../venv/Scripts/python.exe -m unittest rebootpcl.tests.test_check_interfaces -v
```

Expected: PASS, 2 tests.

- [ ] **Step 7: Confirm no detector's verdict changed**

Re-run checks 3 and 4 and diff against the committed `.out` logs.

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && ../venv/Scripts/python.exe rebootpcl/checks/check3_selection_audit.py && ../venv/Scripts/python.exe rebootpcl/checks/check4_circularity.py
```

Expected: check 3 still `TP=1 FP=0 FN=0 TN=1` with two flagged sweeps in the buggy fixture; check 4 still `TP=1 FP=0 FN=0 TN=8`. **A refactor that changes a verdict is a bug in the refactor — stop and fix before continuing.**

- [ ] **Step 8: Commit**

```bash
cd /c/Users/nikhi/Codes/pclCodebase && git add chat1_protocol/rebootpcl/ && git commit -m "refactor: expose structured run() on all five detectors

Detectors previously only printed. Returning Case records lets the
multi-seed, specificity, external-validation and baseline experiments
consume verdicts through one interface. Verdicts unchanged.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The headline-table runner

**Files:**
- Create: `rebootpcl/run_all.py`
- Create: `rebootpcl/results/` (gitignored; holds JSON output)

**Interfaces:**
- Consumes: `run()` on all five checks; `harness.confusion`, `harness.fmt_matrix`.
- Produces: `rebootpcl/results/headline.json` and a printed five-row table.

**Design note:** checks 2 and 5 take minutes. The runner defaults to `--fast`, which runs only checks 3 and 4 and reads the last recorded result for 1, 2 and 5 from `results/`. `--full` re-runs everything (roughly 20 minutes). This keeps the runner usable as a smoke test while still being the single command that reproduces the paper table.

- [ ] **Step 1: Write the failing test**

Create `rebootpcl/tests/test_run_all.py`:

```python
import os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.run_all import CHECKS, collect_fast


class TestRunner(unittest.TestCase):
    def test_declares_all_five_checks(self):
        self.assertEqual(sorted(CHECKS), [1, 2, 3, 4, 5])

    def test_fast_mode_runs_the_static_checks_and_matches_known_results(self):
        rows = collect_fast()
        self.assertEqual(rows[3]["TP"], 1)
        self.assertEqual(rows[3]["FP"], 0)
        self.assertEqual(rows[3]["FN"], 0)
        self.assertEqual(rows[3]["TN"], 1)
        self.assertEqual(rows[4]["TP"], 1)
        self.assertEqual(rows[4]["FP"], 0)
        self.assertEqual(rows[4]["FN"], 0)
        self.assertEqual(rows[4]["TN"], 8)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && ../venv/Scripts/python.exe -m unittest rebootpcl.tests.test_run_all -v
```

Expected: FAIL, `No module named 'rebootpcl.run_all'`.

- [ ] **Step 3: Implement the runner**

Create `rebootpcl/run_all.py`:

```python
"""Reproduce the five-row confound-detector table.

    python rebootpcl/run_all.py            # fast: static checks live, rest cached
    python rebootpcl/run_all.py --full     # re-run everything (~20 min)

--full must be run with PCL_TEST_MODE=1 (checks 1 and 2 refuse otherwise).
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rebootpcl.harness import confusion, fmt_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

CHECKS = {
    1: ("label-definition shift", "check1_label_shift", True),
    2: ("pretraining leakage", "check2_pretrain_leakage", True),
    3: ("OOD-contaminated selection", "check3_selection_audit", False),
    4: ("circular constraints", "check4_circularity", False),
    5: ("missingness/scale", "check5_missingness_scale", True),
}
SLOW = {n for n, (_, _, slow) in CHECKS.items() if slow}


def _import(mod):
    import importlib
    return importlib.import_module(f"rebootpcl.checks.{mod}")


def collect_fast():
    """Run the seconds-long static checks; load cached results for the rest."""
    rows = {}
    for n, (_, mod, slow) in CHECKS.items():
        if slow:
            p = os.path.join(RESULTS, f"check{n}.json")
            rows[n] = json.load(open(p))["counts"] if os.path.exists(p) else None
        else:
            rows[n] = confusion(_import(mod).run(verbose=False))
    return rows


def collect_full(seed=0):
    rows = {}
    for n, (_, mod, _) in CHECKS.items():
        cases = _import(mod).run(seed=seed, verbose=True) if n != 3 and n != 4 \
            else _import(mod).run(verbose=True)
        counts = confusion(cases)
        rows[n] = counts
        os.makedirs(RESULTS, exist_ok=True)
        json.dump({"counts": counts,
                   "cases": [c._asdict() for c in cases]},
                  open(os.path.join(RESULTS, f"check{n}.json"), "w"), indent=2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = collect_full(args.seed) if args.full else collect_fast()

    print("=" * 78)
    print("CONFOUND DETECTOR VALIDATION" + ("  (full re-run)" if args.full
                                            else "  (fast: 3,4 live; 1,2,5 cached)"))
    print("=" * 78)
    missing = []
    for n, (label, _, _) in sorted(CHECKS.items()):
        c = rows.get(n)
        if c is None:
            missing.append(n)
            print(f"check{n}  {label:<28}  NO CACHED RESULT — run with --full")
            continue
        print(f"check{n}  " + fmt_matrix(label, c))
    if missing:
        print(f"\nincomplete: checks {missing} have no result. "
              "The table is NOT reproduced.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && ../venv/Scripts/python.exe -m unittest rebootpcl.tests.test_run_all -v
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Produce the full table once**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && PCL_TEST_MODE=1 nohup ../venv/Scripts/python.exe rebootpcl/run_all.py --full > rebootpcl/run_all.out 2>&1 &
```

Poll. Expected: five rows, exit 0, and `results/check{1..5}.json` written.

- [ ] **Step 6: Commit**

```bash
cd /c/Users/nikhi/Codes/pclCodebase && git add chat1_protocol/rebootpcl/run_all.py chat1_protocol/rebootpcl/tests/test_run_all.py chat1_protocol/rebootpcl/run_all.out && git commit -m "feat: add run_all.py reproducing the five-row detector table

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 2 — Multi-seed stability (strategy Task 3)

### Task 7: Seed sweep for checks 1, 2 and 5

**Files:**
- Create: `rebootpcl/run_seeds.py`
- Test: `rebootpcl/tests/test_run_seeds.py`

**Interfaces:**
- Consumes: `harness.seed_sweep`, `check{1,2,5}.run(seed=...)`.
- Produces: `rebootpcl/results/seeds.json`, keyed `check{n} -> case name -> {flag_rate, flags, expected, n}`.

**Why checks 3 and 4 are excluded:** both are deterministic static analysis over fixed source files. There is no random component to vary, so a seed sweep over them would report a flag rate of exactly 0 or 1 with zero variance by construction — the same tautology problem Task 2 fixed. State this explicitly in the paper rather than padding the table with meaningless variance columns.

- [ ] **Step 1: Write the failing test**

Create `rebootpcl/tests/test_run_seeds.py`:

```python
import os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.run_seeds import SEEDED_CHECKS, summarize


class TestSeedSweep(unittest.TestCase):
    def test_only_stochastic_checks_are_swept(self):
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

    def test_stable_case_is_not_flagged_unstable(self):
        sweep = {"pos": {"flags": [True] * 5, "n": 5,
                         "flag_rate": 1.0, "expected": True}}
        self.assertFalse(summarize(sweep)["pos"]["unstable"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && ../venv/Scripts/python.exe -m unittest rebootpcl.tests.test_run_seeds -v
```

Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

Create `rebootpcl/run_seeds.py`:

```python
"""Multi-seed stability for the stochastic detectors (1, 2, 5).

Checks 3 and 4 are deterministic static analysis over fixed files — sweeping
seeds over them would produce a flag rate of exactly 0 or 1 with zero variance
by construction, which is not evidence of anything.

    PCL_TEST_MODE=1 python rebootpcl/run_seeds.py --seeds 5
"""
import os
import sys
import json
import argparse
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rebootpcl.harness import seed_sweep

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SEEDED_CHECKS = {
    1: "check1_label_shift",
    2: "check2_pretrain_leakage",
    5: "check5_missingness_scale",
}


def summarize(sweep):
    """Mark any case whose verdict is not unanimous across seeds as unstable.

    A detector whose verdict depends on the seed cannot support a point-estimate
    confusion matrix, which is the whole reason for this sweep.
    """
    out = {}
    for name, d in sweep.items():
        out[name] = dict(d)
        out[name]["unstable"] = (0.0 < d["flag_rate"] < 1.0)
        out[name]["agrees_with_truth_rate"] = (
            d["flag_rate"] if d["expected"] else 1.0 - d["flag_rate"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=1200, help="stays per arm")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    all_out = {}
    for num, mod in sorted(SEEDED_CHECKS.items()):
        m = importlib.import_module(f"rebootpcl.checks.{mod}")
        print(f"\n=== check{num} over {len(seeds)} seeds ===")
        if num == 5:
            sweep = seed_sweep(lambda s: m.run(seed=s, n=args.n), seeds)
        elif num == 2:
            # check 2 is itself multi-seed internally; the outer seed shifts the
            # whole seed block so the sweep measures reproducibility of the
            # verdict, not of a single training run.
            sweep = seed_sweep(lambda s: m.run(seed=s, stays=900, epochs=3), seeds)
        else:
            sweep = seed_sweep(lambda s: m.run(seed=s), seeds)
        summary = summarize(sweep)
        all_out[f"check{num}"] = summary
        for name, d in summary.items():
            mark = "  <-- UNSTABLE" if d["unstable"] else ""
            print(f"  {name:<40} flag_rate={d['flag_rate']:.2f} "
                  f"({sum(d['flags'])}/{d['n']}) expected={d['expected']}{mark}")

    os.makedirs(RESULTS, exist_ok=True)
    json.dump(all_out, open(os.path.join(RESULTS, "seeds.json"), "w"), indent=2)
    unstable = [(c, n) for c, s in all_out.items()
                for n, d in s.items() if d["unstable"]]
    print("\n" + "-" * 78)
    if unstable:
        print(f"UNSTABLE cases ({len(unstable)}): {unstable}")
        print("These cannot be reported as point estimates. Report flag rate.")
    else:
        print("All cases unanimous across seeds.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && ../venv/Scripts/python.exe -m unittest rebootpcl.tests.test_run_seeds -v
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Run the real sweep**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && PCL_TEST_MODE=1 nohup ../venv/Scripts/python.exe rebootpcl/run_seeds.py --seeds 5 > rebootpcl/seeds.out 2>&1 &
```

Budget generously: check 2 at 5 outer seeds × 5 inner seeds × 4 leakage levels is the dominant cost. If it exceeds ~45 minutes, reduce check 2's outer sweep to 3 and say so in the writeup — do not silently drop it.

**Report any `UNSTABLE` case immediately.** An unstable verdict means the headline table's point estimate for that row is not supportable and the paper must report a flag rate instead.

- [ ] **Step 6: Commit**

```bash
cd /c/Users/nikhi/Codes/pclCodebase && git add chat1_protocol/rebootpcl/run_seeds.py chat1_protocol/rebootpcl/tests/test_run_seeds.py chat1_protocol/rebootpcl/seeds.out && git commit -m "feat: multi-seed stability sweep for the stochastic detectors

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 3 — External validation (strategy Task 1)

**Scope correction the implementer must understand.** The strategy brief proposes one second pipeline for all five detectors. That does not fit what the detectors are:

- Checks **3 and 4** never touch data or a model. Check 3 is an AST walk over a training script; check 4 is a lineage trace over loader source. "External validation" for these means running them against *other people's code*, which is stronger evidence than a synthetic second pipeline and costs nothing.
- Check **2** measures masked-reconstruction probe loss after pretraining. A gradient-boosted or logistic model has no pretraining stage, so it cannot host this diagnostic at all. The external testbed for check 2 must be a different *representation learner* — a small dense autoencoder is enough, and stays on CPU.
- Checks **1 and 5** need only data, not a model. Their external validation is a different dataset pair (MIMIC-IV demo vs eICU demo), which is genuinely independent of the PhysioNet A/B case they were built on.

### Task 8: External validation of checks 3 and 4 against third-party repositories

**Files:**
- Create: `rebootpcl/external/repos.py`
- Create: `rebootpcl/external/run_repos.py`
- Test: `rebootpcl/tests/test_repos.py`

**⚠️ FLAG BEFORE RUNNING:** this task requires cloning third-party public repositories. That is outside the stated dataset guardrail's letter (it is code, not data) but it is new external material and involves downloading and reading files from the internet. **Stop and get sign-off before cloning anything.** Clone into the scratchpad, never into the working tree, and never execute third-party code — checks 3 and 4 parse source, they do not run it.

**Interfaces:**
- Produces: `repos.TARGETS`, a list of `{"name", "url", "commit", "files": [(relpath, expected_flag, rationale)]}`. Ground truth is established by a human reading the file and recording *why* it does or does not select on OOD data, in the `rationale` field. That rationale text is the audit trail; a `expected_flag` without one is not acceptable evidence.

- [ ] **Step 1: Get sign-off, then select repositories**

Candidate criteria: public clinical-ML or domain-generalization repos with a hyperparameter sweep and a multi-site evaluation. Target 4-6 repos giving at least 3 expected-clean and 1-2 expected-flagged files. Record the exact commit SHA for each so the result is reproducible.

- [ ] **Step 2: Write the failing test**

Create `rebootpcl/tests/test_repos.py`:

```python
import os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.external.repos import TARGETS


class TestTargets(unittest.TestCase):
    def test_every_file_has_a_pinned_commit_and_a_rationale(self):
        self.assertGreaterEqual(len(TARGETS), 1)
        for t in TARGETS:
            self.assertRegex(t["commit"], r"^[0-9a-f]{7,40}$",
                             f"{t['name']} needs a pinned commit SHA")
            for rel, exp, why in t["files"]:
                self.assertIsInstance(exp, bool)
                self.assertTrue(why and len(why) > 20,
                                f"{t['name']}/{rel} needs a written rationale, "
                                "not a bare expected flag")

    def test_ground_truth_contains_both_classes(self):
        flags = [exp for t in TARGETS for _, exp, _ in t["files"]]
        self.assertIn(True, flags, "need at least one expected positive")
        self.assertIn(False, flags, "need at least one expected negative")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run to verify it fails, then populate `repos.py` and re-run**

```bash
cd /c/Users/nikhi/Codes/pclCodebase/chat1_protocol && ../venv/Scripts/python.exe -m unittest rebootpcl.tests.test_repos -v
```

- [ ] **Step 4: Run check 3 over the external files and record the confusion matrix**

`run_repos.py` calls `check3_selection_audit.run(paths=[...])` with the external paths, then prints `fmt_matrix("check3 external", confusion(cases))`.

**Expect false positives and false negatives here.** The detector's `VAL_NAMES` / `OOD_NAMES` / `EVAL_FNS` vocabularies are drawn from this project's naming conventions, and other codebases will not use them. That is the single most important finding this task can produce: it tells you whether check 3 is a detector or a project-specific grep. **Do not extend the name lists to make external repos pass and then report the extended detector as if it generalized.** If the vocabularies need extending, report both the pre-extension and post-extension matrices, and be explicit that the second is fitted.

- [ ] **Step 5: Run check 4's lineage extraction over external loaders** (depends on Task 12; defer this step until `lineage_extract.py` exists)

- [ ] **Step 6: Commit**

---

### Task 9: Build the external clinical pipeline

**Files:**
- Create: `rebootpcl/external/pipeline.py`
- Test: `rebootpcl/tests/test_external_pipeline.py`

**Interfaces:**
- Produces:
  - `load_pair(dataset_a, dataset_b, n=800, seed=0) -> (StayTable, StayTable)` where `StayTable` is a NamedTuple `(name, X: np.ndarray, y: np.ndarray, stay_ids: np.ndarray, feature_names: list[str])`. `dataset_a`/`dataset_b` are `"mimic4"` or `"eicu"`.
  - `fit_logistic(train: StayTable, seed) -> model`, `fit_gbm(train: StayTable, seed) -> model` (`sklearn.linear_model.LogisticRegression` and `sklearn.ensemble.HistGradientBoostingClassifier`; NOT lightgbm, which is not installed).
  - `auroc(model, table) -> float`.
  - `fit_autoencoder(X, seed, epochs=15) -> torch.nn.Module` and `recon_loss(model, X) -> float` — a 2-layer dense autoencoder over the per-stay feature vector, existing solely so check 2's pretraining-leakage probe has a representation learner to attach to.

**Design constraint:** this pipeline must be architecturally unlike PCL's 6-layer time-series transformer. Per-stay aggregate features (mean/min/max/last per variable) plus a flat classifier is a genuinely different modelling choice, and it is the standard clinical-ML baseline, which makes it a fair external target. `src/baselines.py:540 _aggregate_features` already does per-stay aggregation for the PCL codebase — read it for the variable conventions, but write a standalone implementation here so the external pipeline does not inherit PCL's preprocessing decisions.

- [ ] **Step 1: Write the failing test**

Create `rebootpcl/tests/test_external_pipeline.py`:

```python
import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.external.pipeline import (load_pair, fit_logistic, fit_gbm,
                                         auroc, StayTable)


class TestPipelineShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a, cls.b = load_pair("mimic4", "eicu", n=200, seed=0)

    def test_tables_are_aligned_on_features(self):
        self.assertEqual(self.a.feature_names, self.b.feature_names)
        self.assertEqual(self.a.X.shape[1], self.b.X.shape[1])

    def test_no_stay_id_appears_in_both_sites(self):
        self.assertEqual(len(set(self.a.stay_ids) & set(self.b.stay_ids)), 0)

    def test_labels_are_binary_and_not_constant(self):
        for t in (self.a, self.b):
            self.assertEqual(set(np.unique(t.y)) - {0, 1}, set())
            self.assertGreater(t.y.mean(), 0.0, f"{t.name} has no positives")
            self.assertLess(t.y.mean(), 1.0, f"{t.name} is all positives")

    def test_models_train_and_score_above_chance_in_domain(self):
        for fit in (fit_logistic, fit_gbm):
            m = fit(self.a, seed=0)
            self.assertGreater(auroc(m, self.a), 0.60,
                               "in-domain AUROC at chance means the pipeline "
                               "is broken, not that the task is hard")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Expected: `ModuleNotFoundError: No module named 'rebootpcl.external.pipeline'`.

- [ ] **Step 3: Implement `pipeline.py`**

Read `src/data/mimic4.py` and `src/data/eicu.py` for the existing loaders and `src/data/variables.py` for the canonical variable names. Build per-stay features as `{mean, min, max, last, n_obs}` per variable, hstacked in a fixed order taken from the intersection of the two datasets' available variables. Impute missing aggregates with the *training-site* median only (never a pooled median — that is itself a leak), and record which entries were imputed, because check 5 needs the availability pattern.

Label with `src.data.sofa_sepsis` so the external pipeline's task is Sepsis-3, matching the paper's primary task.

- [ ] **Step 4: Run to verify the test passes**

- [ ] **Step 5: Commit**

---

### Task 10: Run checks 1, 2 and 5 against the external pipeline

**Files:**
- Create: `rebootpcl/external/run_external.py`
- Test: `rebootpcl/tests/test_run_external.py`

**Interfaces:**
- Consumes: `pipeline.load_pair`, `pipeline.fit_autoencoder`, `check{1,2,5}.run(...)`.
- Produces: `rebootpcl/results/external.json`; the PCL-vs-external side-by-side table.

Per-detector external instantiation:

- **Check 1:** sites are MIMIC-IV demo and eICU demo. Positive = MIMIC labeled by SOFA against eICU labeled by ICD. Negative control = both labeled by SOFA. The audit subset is eICU stays scored under both criteria (MIMIC lacks an ICD sepsis labeler in `src/data/sepsis.py` — verify this before assuming it; if it has one, use MIMIC for the audit instead and say which).
- **Check 2:** source = MIMIC-IV demo, target = eICU demo, model = `fit_autoencoder`, probe = held-out eICU slice never pretrained on, leakage levels unchanged at 0/5/20/100%. Reconstruction loss replaces masked-prediction loss; the paired relative-delta analysis is identical.
- **Check 5:** the two sites are MIMIC-IV demo and eICU demo; the components are the same MAP / Henderson-Hasselbalch / SpO2 residuals, computed from the external pipeline's per-stay tables rather than raw PSVs. Negative control = eICU randomly split against itself.

- [ ] **Step 1: Write the failing test** — assert `run_external.RESULTS_SCHEMA` contains one entry per check in `{1, 2, 5}`, each with both a positive and a negative case, and that no case name collides with a PCL-side case name.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement, wiring each check's `run()` with its external arguments.**

- [ ] **Step 4: Run and record.** Any detector that flags on PCL but not externally, or vice versa, is the headline result of this phase either way. **A detector that fails externally is a finding to report, not a bug to tune out.** Fixing it is legitimate only if the fix is justified independently of making the external case pass, and the pre-fix result is reported alongside.

- [ ] **Step 5: Emit the side-by-side deliverable table** — one row per detector, columns `PCL TP/FP/FN/TN` and `external TP/FP/FN/TN`.

- [ ] **Step 6: Commit.**

---

## Phase 4 — Specificity on legitimate variation (strategy Task 2)

### Task 11: False-positive rate over clean pipeline variants

**Files:**
- Create: `rebootpcl/specificity/variants.py`, `rebootpcl/specificity/run_specificity.py`
- Test: `rebootpcl/tests/test_variants.py`

**Interfaces:**
- Produces: `variants.VARIANTS`, a list of `{"name", "check", "build", "rationale"}` where `build` returns the arguments for that check's `run()` and `rationale` states in prose why this variant is legitimate. Target 12-20 variants spread across the five detectors.

Variant families, all defensible practice with no confound present:

- *Check 1:* SOFA window vs single mode; different suspicion-window widths; different minimum-stay-length filters; two valid ICD code sets.
- *Check 3:* a sweep that evaluates on OOD *alongside* validation and selects on validation (legitimate reporting — must NOT flag); a sweep selecting on a nested inner validation split; a sweep with no OOD loader at all.
- *Check 4:* a constraint whose input is derived by a *different* equation than the constraint's own (legitimate derived feature); a measured variable with a unit conversion applied.
- *Check 5:* two sites with genuinely different physiology but identical recording practice; the same site under two valid resampling intervals.
- *Check 2:* 0% leakage under several distinct pretraining corpora.

**Every variant runs against all five detectors, not just its own family.** A cross-family false positive — check 4 firing on a check-1 variant — is exactly the kind of thing this table exists to surface.

- [ ] **Step 1: Write the failing test** — assert at least 12 variants; every variant has a rationale over 20 characters; every check number 1-5 has at least two variants; no duplicate names.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `variants.py`.**
- [ ] **Step 4: Implement `run_specificity.py`** — cross product of variants × detectors, `expected=False` throughout, emitting per-detector FP rate via `harness.rates`.
- [ ] **Step 5: Run and record `results/specificity.json`.** **Any detector with a non-zero FP rate must be reported in the paper with the specific variant that triggered it named.** Do not raise thresholds to clear the table.
- [ ] **Step 6: Commit.**

---

## Phase 5 — Baseline comparison (strategy Task 5)

### Task 12: What standard practice already catches

**Files:**
- Create: `rebootpcl/standard/baseline_checks.py`, `rebootpcl/standard/run_baselines.py`
- Test: `rebootpcl/tests/test_baseline_checks.py`

**Interfaces:**
- Produces: `baseline_checks.BASELINES`, a dict of name -> callable returning `list[Case]`:
  - `kfold_cv_sanity(table, seed)` — flags when k-fold CV variance across folds exceeds a threshold.
  - `single_split_sanity(table_a, table_b, seed)` — flags when in-domain minus cross-site AUROC exceeds a threshold.
  - `plain_external_validation(table_a, table_b, seed)` — flags when cross-site AUROC falls below an absolute floor.

Each takes the *same* confounded scenarios the five detectors were validated on, so the comparison is like-for-like.

**Expected outcome, stated in advance so it is not read as a result:** these baselines detect *that* something is wrong (performance moved) but cannot say *what*. Under a label-definition shift or a missingness artifact they may not fire at all, because both can leave cross-site AUROC intact while the number means something different. Under pretraining leakage they will fire in the *wrong direction* — leakage makes cross-site performance look better, so a baseline that flags on degradation stays silent exactly when the confound is most damaging. That asymmetry is the argument for purpose-built detectors and it should be the table's headline, not the raw detection counts.

- [ ] **Step 1: Write the failing test** — assert all three baselines exist, are callable, return `Case` lists, and that each returns `expected` matching the scenario's ground truth.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `baseline_checks.py`.**
- [ ] **Step 4: Implement `run_baselines.py`** — baseline detection rate vs. detector detection rate, per failure mode.
- [ ] **Step 5: Run and record `results/baselines.json`.**
- [ ] **Step 6: Commit.**

---

## Phase 6 — Strengthening the n=1 evidence (strategy Task 4)

### Task 13: Extract check 4's lineage from code instead of hand-authoring it

**Files:**
- Create: `rebootpcl/lineage_extract.py`
- Modify: `rebootpcl/checks/check4_circularity.py`
- Test: `rebootpcl/tests/test_lineage_extract.py`

**Why this and not synthetic injection.** Check 4's `LINEAGE` dict at `check4_circularity.py:36-54` is written by hand. Eight of its nine cells are asserted, not measured; only the PhysioNet PaO2 cell is cross-checked against code, by a `verify_lineage_against_code()` grep for `23400`, `cbrt` and `SaO2`. So what the detector currently demonstrates is that a correct lineage table yields a correct verdict — which is true but nearly tautological, and is the real reason check 4 reads as n=1. Injecting a synthetic circular constraint into a hand-written table would not fix this; it would add a second cell the author also wrote. The substantive upgrade is deriving the lineage from loader source, so the table can be wrong and the detector can catch it.

**Interfaces:**
- Produces: `extract_lineage(loader_path) -> dict[str, str | tuple]` in the same format as `LINEAGE`'s per-dataset value, built by walking the loader's AST for assignments to constrained variable names and classifying the right-hand side as `"measured"` (read from the source file/table) or `("derived", equation_name, source_var)`.
- Equation recognition: match RHS against known signatures — Severinghaus by the `23400` constant and a cube root; Henderson-Hasselbalch by `log10` over an `HCO3`/`PaCO2` ratio with the 6.1 and 0.0307 constants; MAP identity by the `DBP + (SBP - DBP)/3` form.

- [ ] **Step 1: Write the failing test** — assert `extract_lineage("src/data/physionet2019.py")["PaO2"] == ("derived", "severinghaus", "SaO2")`; assert `["MAP"] == "measured"`; assert the MIMIC and eICU loaders yield `"measured"` for `PaO2`; assert an inline fixture containing a hand-written Henderson-Hasselbalch inversion is detected.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `extract_lineage`.**
- [ ] **Step 4: Wire check 4 to prefer extracted lineage, falling back to `LINEAGE` with a printed warning when extraction returns nothing for a dataset.** Keep `LINEAGE` as the declared expectation and **report any disagreement between declared and extracted lineage as a finding** — that disagreement is the drift the original grep guard was trying to catch, now generalized.
- [ ] **Step 5: Run and confirm check 4's verdicts are unchanged** (`TP=1 FP=0 FN=0 TN=8`) when driven by extracted lineage.
- [ ] **Step 6: Commit.**

---

### Task 14: Synthetic sweep fixtures for check 3

**Files:**
- Create: `rebootpcl/fixtures/sweep_SYNTH_pos_{1,2,3}.py`, `rebootpcl/fixtures/sweep_SYNTH_neg_{1,2,3}.py`
- Modify: `rebootpcl/checks/check3_selection_audit.py` (default `paths` gains the synthetic fixtures)
- Test: `rebootpcl/tests/test_synth_sweeps.py`

Each synthetic positive expresses OOD-contaminated selection in a *different syntactic form* than the historical bug's conditional-expression shape, since a detector that only handles the one shape it was written against has not been validated:

1. direct `evaluate(model, test_loader)` inside the sweep, selecting on its return;
2. OOD loader reached through a dict lookup built earlier in the function;
3. OOD score written into a list and `argmax`-ed after the loop.

Each negative is the closest legitimate counterpart: same structure, but selection reads the validation score while the OOD score is only logged.

- [ ] **Step 1: Write the failing test** — assert `check3.run()` over the six synthetic fixtures gives `TP=3 FP=0 FN=0 TN=3`.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Write the six fixtures.**
- [ ] **Step 4: Run.** **If forms 2 or 3 are missed, that is a real recall gap in check 3 — report it, then decide whether to extend `_origin` to handle them, reporting pre- and post-fix numbers separately.**
- [ ] **Step 5: Commit.**

---

### Task 15: Write the limitations record

**Files:**
- Create: `rebootpcl/LIMITATIONS.md`

For every detector, record what could not be validated and precisely why, in enough technical detail that the paper can state the limitation accurately rather than glossing it. At minimum this must cover: whether check 3 generalizes past this project's naming conventions (from Task 8); whether check 2's detection floor is stable once split variance is included (from Task 3); whether any detector's verdict is seed-unstable (from Task 7); and any FP found in Phase 4. If a strategy-brief task proved genuinely infeasible, the technical reason goes here — not "did not get to it".

- [ ] **Step 1: Write it from the recorded results, citing the JSON files by name.**
- [ ] **Step 2: Commit.**

---

## Self-Review Notes

- **Spec coverage.** Strategy Task 1 → Phase 3 (Tasks 8-10). Task 2 → Phase 4 (Task 11). Task 3 → Phase 2 (Task 7). Task 4 → Phase 6 (Tasks 13-15). Task 5 → Phase 5 (Task 12). Phases 0-1 are prerequisites the brief did not include but which every downstream task depends on: without structured `run()` output there is nothing for the sweeps to consume, and without the Task 2 and 3 fixes the multi-seed numbers would be measuring a broken control and a pinned split.
- **Deviations from the brief, all stated in-task:** checks 3 and 4 are externally validated against third-party repositories rather than a second pipeline, because they are static analysis; check 2's external testbed is an autoencoder rather than a gradient-boosted model, because a GBM has no pretraining stage to leak into; check 4's n=1 problem is addressed by lineage extraction rather than synthetic injection, because injecting into a hand-written table adds an author-written cell rather than a measurement.
- **Known open risk.** Tasks 9-12 specify interfaces and expected outcomes but their steps are less granular than Tasks 1-7, because their implementation depends on what the external data actually contains. Re-plan Tasks 9-12 in detail once Task 8's external-repo result is known.
