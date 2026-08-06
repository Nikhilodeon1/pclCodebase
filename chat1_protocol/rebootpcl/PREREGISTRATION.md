# Pre-registration — detector 5 variants on external data

**Written before any external (MIMIC-IV demo / eICU demo) result for detector 5
existed.** Committed before the code that produces those results.

## Why this document exists

Detector 5 currently fails its own flagship positive case. On PhysioNet Site A
vs Site B the availability-ratio signal is large and stable (34-41x against a
2.0 gate) while the composition-share signal converges to ~0.27 against a 0.30
gate, so the conjunction never fires: TP=0, FN=1.

An availability-ratio-only variant would recover that TP. Adopting it now would
be fitting the detector to the case it is supposed to detect — the detector's
docstring never justified the conjunction independently of this case, and we
would not have chosen the availability-only form before seeing it fail.

So the two forms are specified here, in advance, and both are run on data whose
answer is not yet known. Whichever wins does so prospectively.

## The two variants

Both consume the same per-stay component statistics. They differ only in the
decision rule.

**Variant A — conjunction (the current detector, unchanged).**

    flag  <=>  max_availability_ratio > 2.0  AND  composition_share > 0.30

**Variant B — availability-only.**

    flag  <=>  max_availability_ratio > 2.0

Variant B reports `composition_share` as a magnitude alongside its verdict but
does not gate on it.

Thresholds are frozen at the values above. Neither is tuned as part of this
experiment. `AVAIL_RATIO_FLAG = 2.0` and `COMP_EXPLAINS_FLAG = 0.30` keep their
existing values.

## Cases, and where ground truth comes from

The natural MIMIC-vs-eICU contrast cannot score either variant, because we have
no independent knowledge of whether the two demo datasets genuinely differ in
recording practice — and establishing it from the availability ratio would be
circular, since that is Variant B's entire signal. Ground truth therefore comes
from controlled ablation on a single external site, where physiology is held
identical by construction and only the recording pattern is changed.

**E1 — injected positive (expected: flag).** Take eICU demo stays, split at
random into two arms. In arm 2 only, delete a fraction `p` of one component's
input values (HCO3, which feeds the Henderson-Hasselbalch term), so that
component becomes computable for fewer stays. Physiology is untouched; only
availability moves. Run at `p` in {0.50, 0.80, 0.95}, reported as a detection
floor rather than a single point, mirroring detector 2's design.

**E2 — negative control (expected: no flag).** The same eICU random split with
no ablation. Identical recording practice on both arms.

**E3 — natural pair (no ground truth; descriptive only).** MIMIC-IV demo vs
eICU demo. Reported with its numbers but **not scored into any confusion
matrix**, because its true state is unknown. It exists to show what the two
variants say about a real cross-database contrast, not to validate them.

**E4 — regression guard (expected: no flag).** PhysioNet Site A split against
itself, the existing negative control. Both variants must stay silent. This is
the case where Variant B is most at risk: dropping the composition gate removes
a constraint, so if Variant B has a specificity cost it should surface here and
in E2.

All cases run at >= 5 seeds. Per the sampling work already done, the stay sample
must be a seeded random sample, never an alphabetical prefix.

## Decision rule, fixed in advance

Variant B replaces Variant A in the paper's main table **only if all three hold**:

1. B detects E1 at a strictly lower ablation fraction than A does, or detects it
   where A does not detect it at all.
2. B produces zero false positives on E2 and E4, across every seed.
3. B's advantage on E1 is consistent in sign across every seed.

If B fires on E2 or E4 at any seed, B is rejected regardless of its performance
on E1, and the paper reports the conjunction with its documented false negative.

If B wins, the paper states plainly that the conjunction was the original form,
that it failed on PhysioNet A/B, that the availability-only form was specified
in advance of the external result, and that this document is the record. If B
loses, that is reported too: an alternative that looked obviously better in
hindsight and did not survive a prospective test.

## What this does not decide

This experiment does not revisit the PhysioNet A/B result. Detector 5's row in
the main table stays TP=0, FP=0, FN=1, TN=1 under Variant A regardless of the
outcome here, because that is what the detector as specified did on the case it
was built for. A Variant B win changes what the paper recommends going forward;
it does not retroactively convert that false negative into a detection.
