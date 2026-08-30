# Outcome — detector 5 variant pre-registration

Kept separate from `PREREGISTRATION.md` so the pre-specification stays as
written. Git history records that the pre-registration was committed (`63f5d95`)
before the code that produced these numbers.

## Results

External cases, 5 seeds each, eICU demo (1563 stays) with controlled HCO3
ablation, plus the PhysioNet regression guard. E3 is descriptive and unscored.

    case                          expected   A flags   B flags   avail ratio   share
    E1 ablation 50%               True         1/5       1/5          1.7      0.598
    E1 ablation 80%               True         4/5       5/5          3.5      0.717
    E1 ablation 95%               True         4/5       5/5          inf      0.936
    E2 eICU split, no ablation    False        0/5       0/5          1.1      0.284
    E4 PhysioNet A vs itself      False        0/5       0/5          1.1      1.414

    Variant A (conjunction)        TP=9  FP=0 FN=6 TN=10   precision 1.00  recall 0.60
    Variant B (availability only)  TP=11 FP=0 FN=4 TN=10   precision 1.00  recall 0.73

    E3 (MIMIC-IV demo vs eICU demo, NOT SCORED): both variants flag,
       availability ratio 3.1, share 0.584

## FULL-SCALE RE-RUN (2026-08-17) — B rejected again, for a stronger reason

Full eICU-CRD (130,446 stays) and full MIMIC-IV (74,607 stays), 5 seeds,
`detectors/logs/full5.out`. The demo-scale decision below is UNCHANGED, but the
reason is now different and firmer.

    case                        expected   A flags   B flags   avail    gap_ratio
    E1 ablation 50%             True         0/5       0/5      1.7      1.266
    E1 ablation 80%             True         5/5       5/5      3.4      1.157
    E1 ablation 95%             True         5/5       5/5    240.9      1.120
    E2 eICU split, no ablation  False        0/5       0/5      1.0      0.371
    E4 PhysioNet A vs itself    False        0/5       0/5      1.1      1.414

    Variant A (conjunction)        TP=10 FP=0 FN=5 TN=10  prec 1.00  rec 0.67
    Variant B (availability only)  TP=10 FP=0 FN=5 TN=10  prec 1.00  rec 0.67

    E3 (MIMIC vs eICU, NOT SCORED): A silent, B flags (avail 2.4, gap 0.198)

**A and B are now IDENTICAL on every scored case.** At demo scale B had higher
recall (0.73 vs 0.60) and the rejection rested on the detection-floor criterion.
At full scale there is no difference to adjudicate: both first detect at 80%
ablation, both miss at 50% where the availability ratio is 1.7 against a 2.0
gate, and both are clean on E2 and E4. B is rejected again, now because it is
INDISTINGUISHABLE rather than merely not-better.

**The composition gate never binds at full scale, and that is the substantive
finding.** Every gap_ratio in the table exceeds the 0.30 threshold — 1.266,
1.157, 1.120, 0.371, 1.414 — so Variant A's conjunction reduces exactly to
Variant B's availability-only condition. The gate is doing no work anywhere.

This CONFIRMS the demo-scale conclusion that the availability gate, not the
composition gate, is what limits sensitivity, and strengthens it: at demo scale
the composition gate was at least binding once (it caused the PhysioNet A/B
false negative at 0.27 against 0.30). At full scale it binds nowhere. The
detector's behaviour is entirely determined by `max_avail_ratio > 2.0`.

**Fresh evidence that composition_gap_ratio is not a share.** Four of the five
cases exceed 1.0 (up to 1.414). At demo scale only E4 did. Reporting this
quantity as "composition explains X% of the gap" would now be wrong in most
cases, not just an edge case.

**The only A/B divergence is E3**, which the pre-registration deliberately left
unscored because its ground truth is unknown. A flags nothing there (gap 0.198
below the gate); B flags on availability 2.4. That divergence therefore cannot
and does not enter the decision — which is exactly why E3 was excluded in
advance.

Consistency check that passed: E4's gap_ratio is 1.414 at both demo and full
scale, identical to three decimals, because PhysioNet was already the full
dataset in both runs. Only eICU and MIMIC changed size.

Detection floor is unchanged at 80% ablation.

## Decision: Variant B is NOT adopted

The pre-registered rule required all three of:

1. B detects E1 at a **strictly lower ablation fraction** than A, or detects it
   where A does not detect it at all. **NOT MET.** The lowest level at which
   either variant fires is 50%, and both fire there on the same 1 of 5 seeds. At
   80% and 95% A already detects; B is more reliable there, not more sensitive.
2. Zero false positives on E2 and E4 across every seed. **Met** — B is clean.
3. Advantage on E1 consistent in sign. **Met in the weak sense**: B is never
   worse, but its advantage is zero on most seeds.

Condition 1 fails, so under the rule fixed in advance the conjunction stays.
Detector 5's row in the main table remains Variant A.

B's higher recall (0.73 vs 0.60) is real and it is tempting, which is exactly
why the rule was written down first. Recall alone was never the criterion.

## What the experiment actually established

**The composition gate was never what limited sensitivity — the availability
gate is.** Both variants share `AVAIL_RATIO_FLAG = 2.0`, and at 50% ablation the
availability ratio is 1.7, below that gate for both. Dropping the composition
gate buys reliability at levels that were already detectable; it does not lower
the detection floor. That is a sharper statement than "availability-only is
better", and it is the opposite of what the PhysioNet A/B failure suggested,
where the composition gate looked like the sole obstacle.

State the demo-versus-full progression exactly this way, because full scale
strengthened the claim rather than merely repeating it:

> Demo scale suggested the composition gate might OCCASIONALLY limit
> sensitivity — it bound once, producing the PhysioNet A/B false negative at
> 0.27 against a 0.30 threshold. Full scale shows it does not limit sensitivity
> at all: every gap_ratio clears the threshold, the conjunction collapses into
> availability-only, and **availability is the entire detector**.

This is a real strengthening of the granularity/pairing thesis, not a
consistency check. At realistic n the detector's behaviour is fully determined
by one of its two gates, and the other is decorative — which is precisely the
kind of thing that stays invisible until a diagnostic is examined at the scale
it will actually be used at.

**Detector 5 does work on data it was not built against.** On eICU with
controlled ablation it reaches precision 1.00 and FPR 0.00 with a detection
floor between 50% and 80% ablation, holding across 5 seeds. This stands beside,
and does not overturn, its documented false negative on PhysioNet A/B.

## A defect in the statistic itself, found here

E4 reports `share = 1.414`. The composition-only gap **exceeds** the naive gap,
so the quantity called "share of the gap explained by composition" is not
bounded to [0, 1] and is not a share. It is a ratio of two gaps that can each
move independently, and the two effects can partially cancel in the naive
aggregate, making the denominator small.

This does not change any verdict above — E4 stays silent under both variants
because its availability ratio is 1.1, far under the gate — but the paper must
not describe this quantity as a percentage of the gap without saying it can
exceed 100%. The PhysioNet A/B value of ~0.27 should be read as a ratio, not a
proportion.

## Carried forward

The oxygen term differs between instantiations: PhysioNet compares O2Sat with
SaO2, while the demo datasets record no SaO2 and use the Severinghaus
saturation/tension relation instead. The external component set is analogous,
not identical, and the two results are not a like-for-like replication.

MIMIC-IV demo supplies only 117 stays, which is why it appears solely in the
unscored E3 case rather than in any confusion matrix.
