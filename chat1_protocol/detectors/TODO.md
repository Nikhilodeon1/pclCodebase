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

## 0a. RESUME HERE — detector 5 full scale, E4 failed (2026-08-17)

**State.** The expensive half SUCCEEDED and is cached. Full eICU (130,446 stays)
and full MIMIC-IV (74,607 stays) were loaded and written to
`full_d5_arrays.npz`. The run then died at the E4 regression guard with
`missing .../chat1_protocol/data/physionet2019/training_setA`, because check5's
ROOT was hardcoded relative to __file__ and ignored PHYSIONET_DIR. That is FIXED
(commit "fix: detector 5 now honours PHYSIONET_DIR"). Nothing else is wrong.

Because the arrays are cached, the rerun is MINUTES, not hours — pass
`--arrays`, NOT `--save-arrays`.

**Step 1 — find PhysioNet on the pod.** This is what killed the run; do not
guess it again.

    find /workspace -maxdepth 4 -type d -name "training_setA" 2>/dev/null | head

PHYSIONET_DIR must be the directory CONTAINING training_setA.

**Step 2 — pull, and move the cache.** The `rebootpcl/` -> `detectors/` rename
landed while the pod was off. The npz is untracked so git leaves it behind in
the old directory:

    cd /workspace/pclCodebase && git pull origin main
    mkdir -p chat1_protocol/detectors/results
    mv chat1_protocol/rebootpcl/results/full_d5_arrays.npz \
       chat1_protocol/detectors/results/ 2>/dev/null
    ls -la chat1_protocol/detectors/results/full_d5_arrays.npz

If that npz is GONE, /workspace was not a persistent network volume and the
load must be repeated with `--save-arrays` (2-5h). Check this before assuming
the short path.

**Step 3 — rerun.**

    cd /workspace/pclCodebase/chat1_protocol
    export MIMIC_DIR=/workspace/physionet.org/files/mimiciv/3.1
    export EICU_DIR=/workspace/physionet.org/files/eicu-crd/2.0
    export PHYSIONET_DIR=<from step 1>
    export PCL_TEST_MODE=1
    python detectors/external/run_external5.py --seeds 5 --physionet-n 800 \
        --arrays detectors/results/full_d5_arrays.npz \
        > detectors/logs/full5.out 2>&1
    tail -45 detectors/logs/full5.out

**What to look for.** The pre-registered rule rejected Variant B at demo scale
because it did not detect at a strictly LOWER ablation fraction — both variants
first fired at 80%, neither at 50%, where the availability ratio was 1.7 against
a 2.0 gate. At full scale those ratios come from vastly more stays. If B now
detects at 50% where A does not, the pre-registered criterion is met and B should
be adopted — that is a DECISION REVERSAL and goes back to the strategy advisor
rather than being applied unilaterally. Also watch E4: its gap_ratio was 1.414 at
demo scale (the case that exposed the unbounded-ratio bug). It should stay silent
on availability grounds; if it flags, that is a new false positive and takes
priority over everything else.

Detector 2 full-scale comes AFTER detector 5's result is reviewed, not before.

**Machine:** 8-core/32GB suffices for the cached rerun. Use 16-core/64GB only if
step 2 shows the npz is gone and the full load must repeat.

## 0. NEXT POD SESSION — exact commands (full-scale, resumes 2026-08-06)

State: full-scale detector 1 ran and PASSED (see README's full-scale block).
Its label arrays were NOT cached, so the bootstrap CI is still demo-scale.
`--save-labels` now exists so this is the last time labelling is paid for.

Pod: RunPod, 8-core/32GB, ~$0.32/h. NO GPU — nothing here trains anything
bigger than a 0.3M-param model. /workspace is a network volume and persists
across pod shutdown, so write outputs there, never to container-local paths.

    # from the laptop
    git push origin main

    # on the pod
    cd /workspace/pclCodebase && git pull origin main && cd chat1_protocol
    export MIMIC_DIR=/workspace/physionet.org/files/mimiciv/3.1
    export EICU_DIR=/workspace/physionet.org/files/eicu-crd/2.0
    export PHYSIONET_DIR=/workspace/physionet2019
    export PCL_TEST_MODE=1

    # ~2-4h. Caches labels so nothing after this needs the raw data again.
    nohup python detectors/external/run_external1.py \
        --save-labels detectors/results/full_labels.npz \
        > detectors/full1_final.out 2>&1 &

    # seconds, once the above finishes
    python detectors/external/bootstrap_kappa.py \
        --labels detectors/results/full_labels.npz --iters 5000 \
        > detectors/full1_bootstrap.out 2>&1

Then COPY `detectors/results/full_labels.npz` off the pod and commit it — it is
small (label arrays only) and makes every future detector 1 analysis free.

PCL_TEST_MODE stays 1 deliberately: it does not subset this task's data
(`load_stays` reads all of icustays.csv.gz) but it holds the model at
d=64/2-layer, so demo and full-scale rows stay comparable.

Unverified: `/workspace/physionet2019` was never confirmed to exist. Detector 1
does not need it; detector 5's regression guard (Task B) does. Check before
starting Task B.

Task B (detectors 5 and 2 at full scale) and Task C are NOT started. Detector 5
loads full eICU through a heavier path than detector 1 — budget several hours
and watch RSS, since eICU's vitalPeriodic is ~146M rows / ~35GB uncompressed.
Priority is now low: detector 1 was the open risk and it is answered.

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
explicitly depends on it (see `detectors/tests/test_normalizer_bounds.py`).
Fix the documentation to match the code.

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

## 3. Should every real-data detector report an uncertainty measure by default?

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
