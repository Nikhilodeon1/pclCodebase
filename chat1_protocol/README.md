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
    1      label-definition shift        1  0  0  1   kappa 0.467 vs 0.771 control
    2      pretraining leakage           3  0  0  1   detection floor 5%, 5 seeds
    3      OOD-contaminated selection    1  0  0  1   caught the real historical bug
    4      circular constraints          1  0  0  8   precision 1.00 recall 1.00
    5      missingness/scale             0  0  1  1   FALSE NEGATIVE on its own positive case

Check 1's TN was 2 and its controls reported kappa 1.000. Those controls passed
the same audit array as both arguments to Cohen's kappa, so the result was 1.0
by construction. The control is now SOFA window-mode against SOFA single-mode —
two valid Sepsis-3 operationalizations — giving kappa 0.771 (raw agreement
0.962) and one genuinely measured TN.

Check 1's reference implementation is WINDOW-mode SOFA, per Sepsis-3 (Singer et
al. 2016), which specifies a SOFA rise over an interval around the suspicion
time rather than a reading at one instant. The positive case is therefore ICD vs
SOFA-window, kappa 0.467. Single-point scoring is a CONTROL VARIANT, not the
comparison target; it previously served as both. The kappa 0.358 figure quoted
earlier is ICD vs SOFA-single-point and belongs only in the control breakdown,
never as a second headline.

External validation and specificity, detector 1 (see rebootpcl/external1.out,
rebootpcl/bootstrap_kappa.out):

    MIMIC-IV demo (external)  TP=1 FP=0 FN=0 TN=3
    eICU (original database)  TP=1 FP=0 FN=0 TN=3

Report the TN side as "TN=3 (1 discriminating, 2 non-discriminating)". The two
suspicion-window WIDTH controls sit at kappa 0.93-1.00 and would pass almost
anything; only window-vs-single-point discriminates. A genuinely hard control
must vary the infection-suspicion criteria, not the window width.

DETECTOR 1 SPECIFICITY EXPOSURE: on MIMIC the window-vs-single-point control
gives kappa 0.651 against a 0.60 threshold. Bootstrapping over patients (5000
resamples) gives 95% CI [0.528, 0.776] — the threshold is INSIDE the interval,
and P(kappa <= 0.60) = 0.216. At n=117 that control is statistically
indistinguishable from a flag: roughly one run in five, the detector would call
a legitimate Sepsis-3 operationalization a label-definition shift. At eICU's
operating point (500-patient audit) the same control gives [0.683, 0.862] with
P(flag) = 0.001, so the exposure is specific to small cohorts, not general.

Check 2 now re-draws the data split per seed, so the spread includes split
variance, and flags against the df-appropriate one-sided 5% t critical value
rather than a fixed -2.0. At 5 seeds the floor is unchanged at 5% leakage
(rel. delta -3.1%, t=-3.96 against crit -2.132).

Detector 2 external validation, MIMIC-IV demo (source) -> eICU demo (target),
5 seeds (rebootpcl/external2.out):

    leakage    probe loss   paired rel. delta      t    detected
        0%      0.025879                0.0%    0.00     False     <- FP control
        5%      0.025243               -2.6%   -2.86     True      <- floor
       20%      0.020641              -19.7%  -20.21     True
      100%      0.011428              -53.2%  -13.49     True

    check2 EXTERNAL  TP=3 FP=0 FN=0 TN=1

The detection floor is 5%, the same as on PhysioNet A -> B.

Report the two floors with BOTH t-values together — t=-2.86 here, t=-3.96 on
PhysioNet, against a -2.132 critical value. Detection at 5% is marginal on both
pairs, and stating them jointly makes that read as a property of the phenomenon
(5% is an upper bound on the floor, not a constant) rather than as one weak
result.

Effect sizes are larger here (-19.7% vs -9.0% at 20% leakage). The leak pool is
capped at the source size, so contamination PROPORTION is matched between the
experiments and does not explain it. A possible reason is that MIMIC-IV and eICU
are separate hospital systems while PhysioNet A and B are two sites of one
challenge dataset. TREAT THIS AS A POST-HOC OBSERVATION AT n=2 SITE-PAIRS, worth
a sentence, NOT as a finding: "detector sensitivity scales with how distant the
sites are" is one hypothesis fitted to two points. Do not go looking for a third
pair to confirm it — that is the same fitting risk as tuning detector 5's
threshold. If a third pair arises incidentally, note whether it is consistent.

Caveat: the MIMIC source is the entire 117-stay demo cohort at every seed, so
the source contributes no split variance; only the target split varies.

DETECTOR 2 SCOPE — state this wherever detector 2's results are reported, not
only in the limitations list. Detector 2 detects DISTRIBUTIONAL OVERLAP between
pretraining data and the target site, with missingness pattern as one channel of
it. It does NOT isolate leakage of physiological values. Measured across 3 seeds
(rebootpcl/check2_fill_audit.out), training-set fill proportion moves
monotonically with the injected leakage, 0.588 at 0% to 0.539 at 100%, because
Site B records more sparsely than Site A. Adding target stays therefore changes
the value distribution and the missingness pattern together, and this design
cannot attribute detection to either alone. The probe slice is identical across
leakage levels within a seed, so its own fill proportion cancels exactly in the
paired difference — the limitation is about what a detection MEANS, not about
the validity of the measurement.

## Task 5 — what standard practice catches (rebootpcl/baselines_n2400.out)

Three standard checks against detector 2's leakage injection, 3 seeds,
2400 stays/site, scored the same TP/FP/FN/TN way as the detectors:

    leakage   kfold sd   in-domain    target      gap
        0%     0.0789      0.7827    0.6654   0.1173
        5%     0.0789      0.7818    0.6715   0.1103
       20%     0.0795      0.7810    0.6704   0.1106
      100%     0.0804      0.7793    0.6762   0.1031

    external_floor        TP=3 FP=1 FN=0 TN=0   prec 0.75  rec 1.00  FPR 1.00
    kfold_cv_instability  TP=3 FP=1 FN=0 TN=0   prec 0.75  rec 1.00  FPR 1.00
    train_test_gap        TP=3 FP=1 FN=0 TN=0   prec 0.75  rec 1.00  FPR 1.00
    detector 2            TP=3 FP=0 FN=0 TN=1   prec 1.00  rec 1.00  FPR 0.00

DO NOT report the baselines' recall of 1.00 as competence. All three fire at
EVERY level INCLUDING 0% leakage: FPR is 1.00 and TN is 0. A check that always
fires has recall 1.00 by construction and carries no information. Precision 0.75
is just 3 positives out of 4 cases.

The directional prediction HELD but its MECHANISM DID NOT, and the paper must
state the measured version, not the predicted one. Cross-site AUROC did rise
with leakage (+0.0061, +0.0050, +0.0108) and the in-domain/cross-site gap did
shrink (0.1173 -> 0.1031), so leakage does make cross-site performance look
better. But the baselines do not therefore "stay silent when the confound is
worst" — they fire indiscriminately. The reason they are useless is that the
ordinary source-to-target domain gap (0.117 at ZERO leakage) is an order of
magnitude larger than the leakage-induced change (~0.01), so the confound is
buried under the baseline domain shift. The argument for purpose-built detectors
is signal-to-nuisance, not direction: detector 2's probe loss moves -3.1% /
-9.0% / -15.6% with a clean 0% control, while the same confound moves downstream
AUROC by about 0.01 against a 0.12 nuisance.

An earlier run at 900 stays/site gave train_test_gap a perfect TP=3 FP=0 FN=0
TN=1. That was an artifact: one seed's in-domain AUROC was undefined (a held-out
split with one class), `nan > threshold` evaluated False, and the abstention was
scored as a correct silence. Baseline checks now return (flagged, decidable) and
an undefined metric is excluded as UNDECIDABLE rather than counted silent — the
same discipline as detector 3's INDETERMINATE.

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
