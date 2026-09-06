# Open items

Deferred deliberately. Each says why it is not urgent and what breaks if it is
forgotten.

## Methods note — verify asserted resources on the executing machine

Not a paper finding; a working rule worth keeping. A task prompt stated we had
credentialed access to full MIMIC-IV and eICU-CRD. That was true of the pod and
false of the machine actually running the work, where only demos existed. Had it
not been checked first, the run would have produced demo-scale numbers under a
"full-scale" label — the exact class of silent mislabelling this project keeps
catching elsewhere.

Rule: when a prompt asserts a resource is available, verify the assertion holds
in the environment that will execute, not in principle. Cheap, and it is the
same discipline as checking a control is a measurement rather than an identity.

## 0a. DONE — detector 5 full scale (closed 2026-08-17)

The E4 regression guard died on a hardcoded PhysioNet path that ignored
`PHYSIONET_DIR`. Fixed in "fix: detector 5 now honours PHYSIONET_DIR", rerun from
the cached arrays, and the result is recorded in `detectors/logs/full5.out` and
`PREREGISTRATION_OUTCOME.md`. Variant B was rejected a second time, on stronger
grounds than at demo scale. Nothing outstanding.

## 0. Regenerating the per-patient caches (the published reproducibility path)

`full_labels.npz`, `full_d5_arrays.npz` and `full_d2_pools.pkl` are per-stay
artifacts derived from restricted data. They are **deliberately not in this
repository** and are excluded by `.gitignore`; aggregate results are committed
instead, under `detectors/results/*.json`. This section is therefore not a
convenience note any more — it is the path that stands in for the missing files,
so it must stay accurate.

Every flag named below was verified against the argument parsers on 2026-09-06.

Requirements: your own credentialed copies of PhysioNet 2019, MIMIC-IV 3.1 and
eICU-CRD 2.0. `run_all.py` does NOT need any of this — it reproduces the results
table from the committed aggregate JSON on a machine with no clinical data at
all. What follows is only for re-deriving those aggregates from raw data.

    cd <repo>/chat1_protocol
    export PHYSIONET_DIR=<dir CONTAINING training_setA and training_setB>
    export MIMIC_DIR=<.../mimiciv/3.1>
    export EICU_DIR=<.../eicu-crd/2.0>
    export PCL_TEST_MODE=1

`PHYSIONET_DIR` must be the PARENT of `training_setA`, not `training_setA`
itself. Pointing it one level too deep is what killed a multi-hour detector 5
run once; it is the single most common mistake here.

`PCL_TEST_MODE=1` stays 1 deliberately. It does not subset this data
(`load_stays` reads all of `icustays.csv.gz`), but it holds the model at
d=64/2-layer so demo and full-scale rows stay comparable.

### Detector 1 — labels and bootstrap (~2-4 h, then seconds)

    python detectors/external/run_external1.py         --save-labels detectors/results/full_labels.npz         > detectors/logs/full1_probe.out 2>&1

    python detectors/external/bootstrap_kappa.py         --labels detectors/results/full_labels.npz --iters 5000         > detectors/logs/full1_bootstrap.out 2>&1

The npz caches the label arrays, so the bootstrap is seconds and can be rerun at
any number of iterations without relabelling. Keep it locally; do not commit it.

### Detector 5 — arrays and variants (2-5 h first time, minutes after)

    python detectors/external/run_external5.py --seeds 5 --physionet-n 800         --save-arrays detectors/results/full_d5_arrays.npz         > detectors/logs/full5.out 2>&1

Rerun with `--arrays` (not `--save-arrays`) to reuse the cache.

### Detector 2 — pools and leakage sweep

    python detectors/external/run_external2.py --seeds 5         --pool-cache detectors/results/full_d2_pools.pkl         > detectors/logs/full2.out 2>&1

`--pool-cache` is written on the first run and reused after, which skips a
40-60 minute load of both databases. Wire it in before launching, not after
discovering it was needed.

### Machine

8-core/32 GB CPU is enough for all three; nothing here trains anything larger
than a 0.3M-parameter model. eICU's `vitalPeriodic` is ~146M rows / ~35 GB
uncompressed, so watch RSS on detector 5's first pass. If you run this on a
rented pod, write outputs to a persistent volume — container-local paths are
lost on shutdown, which is how detector 1's original full-scale logs were lost
(see the archival gap section of the README).

## 1. DONE — CLAUDE.md's preprocessing description now matches the code

It described "unit variance normalization computed only on training data". The
code applies a FIXED min-max map from the clinical plausibility bounds in `PLAUS`
(`src/data/variables.py`); `MinMaxNormalizer.fit()` is a no-op
(`src/data/preprocessing.py`) and the `normalizer.fit(all_raw_ts)` calls in all
three loaders are vestigial.

The documentation was corrected to match the code, never the reverse: every
number in this repository was measured under the fixed-bounds behaviour, and
detector 2's external design depends on the bounds being shared across datasets.
That dependency is guarded by `detectors/tests/test_normalizer_bounds.py`.

Kept here rather than deleted because the failure it describes is a live hazard
for the paper: a methods section drafted from the project document rather than
the source would state something no result was produced under.

## 2. Detector 2 disambiguation arm — DESIGNED, NOT RUN (rebuttal-ready)

**Status:** deliberately not run. This is a stated limitation in the paper, not
a gap to close before submission. It is written up here so it can be PROPOSED
IN REBUTTAL if a reviewer pushes on it, rather than designed from scratch
against a deadline.

**The limitation.** Detector 2 detects distributional overlap between
pretraining data and the target site, missingness pattern included. It cannot
separate "the encoder saw target values" from "the encoder saw target
missingness pattern", because Site B records more sparsely than Site A, so
adding target stays moves both together. Measured: training fill proportion goes
0.588 -> 0.539 from 0% to 100% leakage, consistently across 3 seeds
(`detectors/check2_fill_audit.out`).

**The experiment that would separate them.** Add a fourth arm to the existing
leakage sweep:

* Arms as now: 0%, 5%, 20%, 100% leakage of genuine Site B stays.
* NEW arm: leak SITE A stays that have been down-sampled to match Site B's fill
  proportion (drop observations at random per variable until the arm's fill
  proportion equals the Site B leak pool's, using the same ablation mechanism as
  `detectors/external/pipeline5.py:ablate`). Same count as the 20% Site B arm.

The new arm carries Site B's MISSINGNESS pattern but Site A's VALUES and site
identity. If probe loss on Site B drops as much for it as for the genuine 20%
Site B arm, the detection is driven by missingness alone. If it drops much less,
value-distribution overlap is doing the work and the current result narrows to
that. Either outcome is reportable; both are more informative than the present
scope statement.

Cost: one extra arm on the existing sweep, CPU only, roughly the runtime of one
additional leakage level (order 15 minutes at 5 seeds, 900 stays/site).

## 3. RESOLVED (2026-09-05) — yes, every stochastic detector reports uncertainty

**Decision: mandatory for stochastic detectors, explicitly N/A for deterministic
ones, and a stochastic detector that reports nothing renders as MISSING rather
than as blank.**

The machinery already existed — `harness.requires_uncertainty` and
`harness.fmt_uncertainty`, guarded by `tests/test_uncertainty_policy.py`. What
was open was whether the MAIN RESULTS TABLE carried the column. It now does, in
both `run_all.py` output and README.

What settled it: the two times uncertainty was measured it changed how the
result had to be described, and the audit found a third case while wiring the
column up. Detector 2's stored entry was `{"mean": -0.031, "sd": 0.018}` — the
DEMO run, carried forward with no provenance comment while every surrounding
number had moved to full scale. Nothing flagged it, because a plausible-looking
number in a table nobody printed is invisible. It is now the full-scale paired t
(-6.65 vs crit -2.132), with the source log named in a comment.

The argument against — that an interval on a not-close verdict adds noise —
turned out to be backwards. The intervals that added the most were on the rows
that looked settled: detector 5's 2/5 flag rate is stable-looking until you see
it is 0/3 at n=4000, i.e. not scale-invariant at all.

**Rules that follow, and must hold in the paper.** Detector 1 is quoted with its
AUDIT-SUBSET interval, never the cohort interval, because the audit subset is
what the detector scores. Detector 2 is quoted as a paired t, not mean +/- sd,
because its unpaired null spread is comparable to the effect size. Detector 5's
flag rate is never quoted without its n.
