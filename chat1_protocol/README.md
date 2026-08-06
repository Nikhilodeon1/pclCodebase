# Chat 2 — Confound-detection protocol (new methods paper)

Five automated detectors, one per confound type, each validated against known
ground truth. All five run LOCALLY ON CPU. Total compute cost: $0.

## Contents
    rebootpcl/checks/check1_label_shift.py         label-definition shift
    rebootpcl/checks/check2_pretrain_leakage.py    non-zero-shot pretraining leak
    rebootpcl/checks/check3_selection_audit.py     OOD-contaminated selection
    rebootpcl/checks/check4_circularity.py         circular derived constraints
    rebootpcl/checks/check5_missingness_scale.py   missingness/scale artifacts
    rebootpcl/fixtures/sweep_BUGGY.py              real historical bug (commit c7cb42f)
    rebootpcl/fixtures/sweep_FIXED.py              corrected version
    src/, config.py                                model + data loaders the checks import
    data/                                          JUNCTION to ../data (not a copy)

## Run
    python rebootpcl/checks/check3_selection_audit.py      # seconds
    python rebootpcl/checks/check4_circularity.py          # seconds
    PCL_TEST_MODE=1 python rebootpcl/checks/check1_label_shift.py        # ~2 min
    python rebootpcl/checks/check5_missingness_scale.py 1200             # ~4 min
    PCL_TEST_MODE=1 python rebootpcl/checks/check2_pretrain_leakage.py --stays 900 --seeds 3 --epochs 3   # ~5 min

Local runs exceed the 2-minute tool timeout — use
`nohup ... > out.log 2>&1 &` and poll the log.

## Results (all validated)
    check  confound                     TP FP FN TN   headline
    1      label-definition shift        1  0  0  1   kappa 0.358 vs 0.771 control
    2      pretraining leakage           3  0  0  1   detection floor 5%, 5 seeds
    3      OOD-contaminated selection    1  0  0  1   caught the real historical bug
    4      circular constraints          1  0  0  8   precision 1.00 recall 1.00
    5      missingness/scale             0  0  1  1   FALSE NEGATIVE on its own positive case

Check 1's TN was 2 and its controls reported kappa 1.000. Those controls passed
the same audit array as both arguments to Cohen's kappa, so the result was 1.0
by construction. The control is now SOFA window-mode against SOFA single-mode —
two valid Sepsis-3 operationalizations — giving kappa 0.771 (raw agreement
0.962) and one genuinely measured TN.

Check 2 now re-draws the data split per seed, so the spread includes split
variance, and flags against the df-appropriate one-sided 5% t critical value
rather than a fixed -2.0. At 5 seeds the floor is unchanged at 5% leakage
(rel. delta -3.1%, t=-3.96 against crit -2.132).

## Notes that matter
* `data/` is a Windows directory JUNCTION to `../data`. Do NOT `rm -rf` it —
  that can delete the real 455MB dataset. Remove with `cmd //c rmdir` instead.
* `load_physionet2019(fraction=...)` subsamples the FILE LIST before reading.
  Always pass a fraction; loading everything and slicing after reads all 40k
  PSVs and takes ~25 minutes.
* Two of the five detectors initially produced FALSE NEGATIVES (check 5 wrong
  aggregation granularity, check 2 unpaired instead of paired). Both failures are
  paper material: these confounds are invisible unless the diagnostic mirrors the
  aggregation granularity and pairing structure of what it audits.
* Check 2's 5% floor is specific to this scale (0.3M params, 900 stays/site,
  3 epochs) — an upper bound on the floor, not a universal constant.
* Check 5's "composition explains 49% of the gap" DOES NOT HOLD. That number
  was produced with the stay list taken as `sorted(listdir)[:1200]`, an
  alphabetical prefix; PhysioNet filenames are patient IDs, so a prefix is a
  systematic slice of the site, not a sample of it. Measured on seeded random
  samples:

        n=1200   legacy prefix 0.486   random 0.283 +/- 0.082  [0.152, 0.358]  flags 2/5
        n=4000   legacy prefix 0.323   random 0.274 +/- 0.036  [0.233, 0.299]  flags 0/3

  The share converges to ~0.27, and the legacy value lies outside the random
  range at BOTH sizes — it was an artifact of the prefix AND of the small n.
  Since 0.27 sits below the detector's own COMP_EXPLAINS_FLAG of 0.30, check 5
  does not detect its own flagship positive case: TP=0, FN=1. The 2/5 flag rate
  at n=1200 was estimation noise straddling the threshold, not detection.
  The availability-ratio signal is robust throughout (34-41x against a 2.0
  gate); it is the composition-share gate that fails.
  Do NOT lower the threshold to recover the TP — that fits the detector to the
  case it is supposed to detect. See rebootpcl/check5_sampling_sensitivity.py.
