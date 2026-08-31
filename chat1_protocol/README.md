# Chat 1 — Confound-detection protocol (new methods paper)

Five automated detectors, one per confound type, each validated against known
ground truth. All five run LOCALLY ON CPU. Total compute cost: $0.

## Contents
    detectors/checks/check1_label_shift.py         label-definition shift
    detectors/checks/check2_pretrain_leakage.py    non-zero-shot pretraining leak
    detectors/checks/check3_selection_audit.py     OOD-contaminated selection
    detectors/checks/check4_circularity.py         circular derived constraints
    detectors/checks/check5_missingness_scale.py   missingness/scale artifacts
    detectors/fixtures/sweep_BUGGY.py              real historical bug (commit c7cb42f)
    detectors/fixtures/sweep_FIXED.py              corrected version
    detectors/baselines/                           naive baselines the detectors are compared against
    detectors/results/                             parsed JSON output per check
    detectors/logs/                                raw stdout from each run
    src/, config.py                                model + data loaders the checks import
    data/                                          JUNCTION to ../data (not a copy)

Dependencies: `../requirements.txt` (shared root venv — no separate copy here).

## Run
    python detectors/checks/check3_selection_audit.py      # seconds
    python detectors/checks/check4_circularity.py          # seconds
    PCL_TEST_MODE=1 python detectors/checks/check1_label_shift.py        # ~2 min
    python detectors/checks/check5_missingness_scale.py 1200             # ~4 min
    PCL_TEST_MODE=1 python detectors/checks/check2_pretrain_leakage.py --stays 900 --seeds 3 --epochs 3   # ~5 min

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

External validation and specificity, detector 1 (see detectors/logs/external1.out,
detectors/logs/bootstrap_kappa.out):

    MIMIC-IV demo (external)  TP=1 FP=0 FN=0 TN=3
    eICU (original database)  TP=1 FP=0 FN=0 TN=3

Report the TN side as "TN=3 (1 discriminating, 2 non-discriminating)". The two
suspicion-window WIDTH controls sit at kappa 0.93-1.00 and would pass almost
anything; only window-vs-single-point discriminates. A genuinely hard control
must vary the infection-suspicion criteria, not the window width.

FULL-SCALE RE-RUN (2026-08-06, RunPod, `detectors/logs/full1_probe.out` on the pod's
/workspace volume). Full MIMIC-IV 3.1 and full eICU-CRD 2.0, same code, same
PCL_TEST_MODE=1, only the data changed:

    MIMIC-IV FULL: 74,829 stays (demo: 117), 44,808 with suspected infection
      cohort+ICD read 8.1s | SOFA components read 1357.6s | 4 variants scored
      from ONE read

    case                                      demo      FULL
    MIMIC positive ICD vs SOFA win[-48,+24]   0.150     0.204   flags, PASS
    MIMIC negative vs SOFA single-point       0.651     0.634   clean, PASS
    MIMIC negative vs SOFA win[-24,+12]       0.931     0.928   clean, PASS
    MIMIC negative vs SOFA win[-72,+24]       1.000     0.984   clean, PASS
    eICU  positive ICD vs SOFA win[-48,+24]   0.467     0.514   flags, PASS
    eICU  negative vs SOFA single-point       0.771     0.869   clean, PASS
    eICU  negative vs SOFA win[-24,+12]       0.979     0.990   clean, PASS
    eICU  negative vs SOFA win[-72,+24]       1.000     1.000   clean, PASS

    check1 EXTERNAL (MIMIC)  TP=1 FP=0 FN=0 TN=3   (unchanged from demo)
    check1 eICU              TP=1 FP=0 FN=0 TN=3   (unchanged from demo)

THE NEAR-MISS IS NOT A SMALL-SAMPLE ARTIFACT. At 640x the cohort the MIMIC
window-vs-single-point control moved TOWARD the threshold, not away: kappa 0.651
-> 0.634, margin 0.051 -> 0.034. Do not describe this limitation as "an artifact
of the 117-stay demo" — that hypothesis was tested at full scale and rejected.
MIMIC's window-vs-single-point disagreement is genuinely that large. The
exposure is MIMIC-SPECIFIC, not size-specific: eICU's equivalent control moved
AWAY from danger (0.771 -> 0.869) on the same run.

The near-ceiling framing also survives: the two window-WIDTH controls are still
0.928 and 0.984 at full scale, so "TN=3 (1 discriminating, 2 non-discriminating)"
was not a small-sample artifact either.

FULL-SCALE BOOTSTRAP (`detectors/logs/full1_bootstrap.out`, 5000 resamples).
MIMIC n=74,829, eICU n=132,900. Two intervals per case: over the whole cohort,
and over repeated draws of the 500-patient AUDIT SUBSET, which is what the
detector actually scores and therefore what governs its verdict.

    case                          kappa   cohort CI        audit-500 CI     P(flag)
    MIMIC ctrl vs single-point    0.602   [0.597, 0.607]   [0.539, 0.663]    0.484
    MIMIC ctrl vs win[-24,+12]    0.913   [0.910, 0.916]   [0.876, 0.948]    0.000
    MIMIC ctrl vs win[-72,+24]    0.993   [0.992, 0.994]   [0.980, 1.000]    0.000
    MIMIC POSITIVE ICD vs SOFA    0.174   [0.168, 0.179]   [0.104, 0.243]    1.000
    eICU  ctrl vs single-point    0.826   [0.821, 0.831]   [0.737, 0.901]    0.000
    eICU  ctrl vs win[-24,+12]    0.986   [0.984, 0.987]   [0.959, 1.000]    0.000
    eICU  ctrl vs win[-72,+24]    1.000   [1.000, 1.000]   [1.000, 1.000]    0.000
    eICU  POSITIVE ICD vs SOFA    0.451   [0.444, 0.458]   [0.334, 0.560]    0.996

**DETECTOR 1'S HEADLINE LIMITATION, at full scale.** At the detector's own
operating point the MIMIC window-vs-single-point control flags **48.4% of the
time** — a coin flip. It is not a small-sample effect and it got WORSE with
data, not better (demo P(flag)=0.216 at n=117). Report this number, not the
demo one.

The mechanism is sharper than "noisy": the full-cohort kappa is 0.602 with a CI
of [0.597, 0.607]. The TRUE agreement between two valid Sepsis-3
operationalizations sits essentially exactly ON the 0.60 flag threshold. The
detector cannot separate them because there is nothing to separate — the
population value and the decision boundary coincide. Tightening the interval
with more data cannot fix that; it only measures the coincidence more precisely.

DO NOT move KAPPA_FLAG to make this control pass. 0.60 is the conventional
Landis-Koch "substantial agreement" boundary, chosen before any of this was
measured; moving it after observing the failure is the same fitting error the
detector 5 pre-registration exists to prevent.

Consequence for the results table. "TN=3" must NEVER appear anywhere in the paper
for MIMIC without this breakdown attached:

    MIMIC TN=3  =  2 non-discriminating  +  1 coincidental

    non-discriminating : win[-24,+12] kappa 0.913, win[-72,+24] kappa 0.993,
                         both P(flag)=0.000 — near-ceiling controls that
                         essentially nothing could fail
    coincidental       : win-vs-single-point, population kappa 0.602 against a
                         0.60 threshold, P(flag)=0.484 — NOT a clean negative.
                         The verdict is close to arbitrary; it is scored TN
                         because one draw happened to land above the boundary

There is arguably not one sound true negative in MIMIC's TN=3 at full scale.
This is not the earlier "1 discriminating, 2 non-discriminating" framing — full
scale demoted the one discriminating control to a coin flip.

Positives are unaffected and robust: both databases detect the ICD-vs-SOFA shift
with P(flag) = 1.000 and 0.996 respectively.

The demo-scale figures below are superseded and kept only for the
demo-vs-full comparison.

DETECTOR 1 SPECIFICITY EXPOSURE (DEMO SCALE, n=117 — SUPERSEDED, see above): on
MIMIC the window-vs-single-point control
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

DETECTOR 2 FULL-SCALE EXTERNAL (2026-08-17, `detectors/logs/full2.out`).
Full MIMIC-IV (74,607 stays) as source, full eICU-CRD (130,446 stays) as target,
5 seeds, leak pool capped at the source size so contamination PROPORTION stays
comparable to the PhysioNet experiment:

    source(MIMIC)=74607   leak pool(eICU, capped)=74607   probe(eICU)=32612

    false positive at 0% leakage: False
    detection floor: 5% leakage
    check2 EXTERNAL  TP=3 FP=0 FN=0 TN=1

The confusion matrix and the 5% floor are UNCHANGED from demo scale, on a target
site 83x larger and a source 638x larger. Detector 2 holds.

    leakage   probe loss   paired rel. delta       t   detected   (crit -2.132)
        0%     0.004961                0.0%    0.00     False    <- FP control
        5%     0.004800               -3.2%   -6.65     True     <- floor
       20%     0.004592               -7.4%  -23.24     True
      100%     0.003912              -21.1%  -64.12     True

THE FLOOR IS NO LONGER MARGINAL AT FULL SCALE, and the earlier "marginal on both
pairs" phrasing must be updated rather than repeated. At 5% leakage t=-6.65
against a -2.132 critical value, roughly a 3x margin. The demo-scale figures were
t=-2.86 (MIMIC demo -> eICU demo) and t=-3.96 (PhysioNet A -> B). So the
marginality itself was largely a small-data artifact: the EFFECT at 5% is stable
across all three runs (-2.6%, -3.1%, -3.2%), but its variance collapses with
data, which is what moves t.

Report all three t-values together, and say which scale each came from. The
floor remains an UPPER BOUND on the true floor -- nothing below 5% leakage has
ever been tested -- and that framing is unchanged.

RETIRES AN EARLIER POST-HOC HYPOTHESIS. The demo-scale external run showed much
larger effects than PhysioNet at matched leakage (-19.7% vs -9.0% at 20%), and
this README previously floated, explicitly as an unconfirmed n=2 observation,
that detector sensitivity might scale with how distant the two sites are. Full
scale disconfirms it: with the same site pair and the same capped contamination
proportion, the effects are now -7.4% at 20% and -21.1% at 100%, CLOSE TO
PHYSIONET rather than to the demo-external run. The demo-external inflation is
explained by its 117-stay source -- target data dominated a tiny training set --
not by hospital distance. Do not carry the distance hypothesis into the paper.

Detector 2 external validation, MIMIC-IV demo (source) -> eICU demo (target),
5 seeds (detectors/logs/external2.out):

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
(detectors/logs/check2_fill_audit.out), training-set fill proportion moves
monotonically with the injected leakage, 0.588 at 0% to 0.539 at 100%, because
Site B records more sparsely than Site A. Adding target stays therefore changes
the value distribution and the missingness pattern together, and this design
cannot attribute detection to either alone. The probe slice is identical across
leakage levels within a seed, so its own fill proportion cancels exactly in the
paired difference — the limitation is about what a detection MEANS, not about
the validity of the measurement.

## Finding: a threshold can coincide with the population value it must separate

State this in the DISCUSSION as a contribution, not folded into a limitations
paragraph. Phrase it this plainly:

**Conventional agreement thresholds can coincide with the true population-level
value between two equally valid label definitions, which no amount of data
resolves because it is not a sampling problem.**

Measured instance. Detector 1 flags label-definition shift when Cohen's kappa on
a fixed audit subset falls below 0.60 — the conventional Landis-Koch "substantial
agreement" boundary, fixed before any of this was measured. On full MIMIC-IV
(74,829 stays) the kappa between SOFA window-mode and SOFA single-point scoring,
two equally defensible Sepsis-3 operationalizations, is **0.602 with a 95% CI of
[0.597, 0.607]**. The population value sits on the decision boundary. At the
detector's operating point (repeated 500-patient audit draws) the control
therefore flags 48.4% of the time: the verdict is close to arbitrary.

This is NOT small-sample noise, and the demo-scale data actively understated it:
at n=117 the same control gave P(flag)=0.216, so scaling the cohort 640x more
than DOUBLED the flag probability rather than resolving it. More data narrows the
interval around 0.602; it cannot move 0.602 away from 0.60.

The threshold was not moved. Adjusting it after observing this failure would be
the same fitting error the detector 5 pre-registration exists to prevent, and the
temptation is stronger here precisely because the required adjustment is tiny.

The counterexample is what keeps this from reading as "kappa-based label-shift
detection does not work": on the SAME full-scale run, eICU's equivalent control
moved AWAY from the boundary (0.771 -> 0.826, P(flag)=0.000). The failure is
MIMIC-specific and axis-specific — it is the window-vs-single-point axis on one
database, not a general property of the detector or of the method. Keep that
contrast adjacent to the finding wherever it appears.

## Finding: pre-registration stopped an ungrounded case from deciding a variant

Belongs in METHODS, as a concrete demonstration that the protocol did work —
next to the granularity finding, not buried in a limitations list.

Detector 5's pre-registration fixed four cases in advance and marked E3
(MIMIC-IV vs eICU, a natural cross-database contrast) as DESCRIPTIVE ONLY,
excluded from every confusion matrix, because we had no independent knowledge of
whether the two databases truly differ in recording practice — and deciding that
from the availability ratio would be circular, since that ratio is Variant B's
entire signal.

At full scale, **E3 turned out to be the ONLY case on which Variants A and B
disagree.** A stays silent (its composition gate is not met, gap_ratio 0.198); B
flags (availability 2.4 exceeds its 2.0 gate). Every scored case is identical
between the two variants.

So had E3 been scored, one case with no ground truth behind it would have decided
the entire variant question on its own — and it would have decided it in favour
of the variant we had reason to suspect was fitted. The exclusion was written
down before any external number existed, for reasons that had nothing to do with
this outcome. That is the protocol earning its cost, demonstrably rather than in
principle.

## Finding: the "share" that was not a share, confirmed at scale

Closure on the `composition_gap_ratio` rename. The quantity was originally named
and described as a share of the cross-site gap. At demo scale exactly 1 of 5
cases exceeded 1.0, which read as an edge case. At full scale **4 of 5 exceed
1.0** (1.266, 1.157, 1.120, 1.414). Reporting it as "composition explains X% of
the gap" would therefore be wrong in the majority of cases at the scale that
matters, not in a corner. The rename was not cosmetic.

## Finding: diagnostics need an explicit UNDECIDABLE state

State this as a contribution in its own right, alongside the aggregation-
granularity finding — not as a lesson learned in passing.

**A diagnostic that has no undecidable state will eventually report false
cleanliness.** When a check cannot reach a judgement — the metric is undefined,
the pattern it looks for is absent, the control is too easy to fail — that
outcome silently becomes "no confound found" and is then counted as evidence of
specificity. The absence of a detection gets scored as a detection of absence.

This occurred FOUR times, independently, in five different pieces of code in
this project alone:

1. **check 1's negative control** passed the same audit array as both arguments
   to Cohen's kappa. The result was 1.0 by construction, and TN=2 measured
   arithmetic rather than the detector.
2. **detector 1's suspicion-window controls** sit at kappa 0.93-1.00 —
   near-ceiling, so they would pass almost any detector. Reported bare, TN=3
   implies three tests when only one discriminates.
3. **detector 3's INDETERMINATE** on third-party repositories. Folding it into
   TN would have produced TN=3 from files the detector never analysed at all.
4. **Task 5's train_test_gap**: a seed whose in-domain AUROC was undefined made
   `nan > threshold` evaluate False, and the abstention was scored as a correct
   silence — handing the baseline a perfect TP=3 FP=0 FN=0 TN=1 that would have
   contradicted this paper's own headline comparison.

Four occurrences across independent code is not coincidence. It is evidence that
this failure mode is structurally common in ad hoc evaluation code, because the
natural Python idiom for "did not detect" and for "could not tell" is the same
falsy value. The fix is cheap and mechanical: every check returns a decidability
flag alongside its verdict, and undecidable cases are excluded from the
confusion matrix and reported in their own column rather than defaulting to
either class.

## Archival gap — detector 1's full-scale logs were not preserved

Detector 1's full-scale numbers in this file (the kappa table and the bootstrap
CIs, including the P(flag)=0.484 headline) were TRANSCRIBED from terminal output,
not read from an archived log. `full1_probe.out` and `full1_bootstrap.out` came
back empty when copied off the pod, and are not in this repository.

Detectors 2 and 5 ARE archived: `detectors/logs/full2.out` and
`detectors/logs/full5.out` are committed, and every figure quoted for those two
was verified against them line by line. The gap is detector 1 alone.

This does not make the numbers wrong -- they were read directly off the runs --
but it does mean the reproducibility claim for detector 1's full-scale row rests
on transcription rather than on an artifact. Two ways to close it, in order of
cheapness:

  1. `full_labels.npz` (detector 1's cached label arrays) may still be on the
     pod's /workspace. If so, `bootstrap_kappa.py --labels <npz> --iters 5000`
     regenerates the bootstrap log in SECONDS, no relabelling.
  2. Otherwise the labelling pass repeats (~40 min) before the bootstrap.

Until one of those is done and the logs are committed, state in the paper that
full-scale detector 1 figures are transcribed from run output.

## Task 5 — what standard practice catches (detectors/logs/baselines_n2400.out)

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

SCOPE OF THE BASELINE COMPARISON — state this explicitly in the paper so the
omission does not read as an oversight: baseline comparison was run against
detector 2's leakage scenario; detector 5's confound is not a
performance-degradation quantity and was not in scope for this comparison.
Detector 5's confound is composition/missingness explaining a cross-site gap in
an aggregate violation score, which these three performance-based checks cannot
address without a different design.

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
  case it is supposed to detect. See detectors/check5_sampling_sensitivity.py.
