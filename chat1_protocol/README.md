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
    1      label-definition shift        1  0  0  2   kappa 0.358 vs 1.000 controls
    2      pretraining leakage           3  0  0  1   detection floor 5%
    3      OOD-contaminated selection    1  0  0  1   caught the real historical bug
    4      circular constraints          1  0  0  8   precision 1.00 recall 1.00
    5      missingness/scale             1  0  0  1   composition explains 49% of gap

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
* Check 5's gap is ~49% composition / ~51% genuine per-component difference.
  Describe it as PARTIALLY artifactual.
