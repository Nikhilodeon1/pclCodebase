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
