# Provenance and history

## Authorship

Work by Nikhil Tamvada, with Claude (Anthropic) as a coding and analysis
assistant throughout. Commits carry `Co-Authored-By: Claude` where that applies.
Every result, decision rule and retraction recorded in `README.md` and
`detectors/PREREGISTRATION_OUTCOME.md` was reviewed by the author; the
pre-registration was committed before the code that produced the numbers it
governs (`63f5d95`).

## The git history interleaves two separate papers

This directory is one work-stream inside a shared repository. A second paper
lives in `chat2_papers/` and is not part of this work. For most of the history
the two are cleanly separated by directory, but **two commits touch both**:

    51a311d  2026-08-17  "clean up time"
    1b9beab  2026-08-05  "ig bro"

`1b9beab` is the substantial one: it adds this project's `run_all.py`,
`run_seeds.py` and their tests (then under `rebootpcl/`, renamed to `detectors/`
later) in the same commit as a rebuild of the other paper's `.docx`/`.pdf` and
its figures. `51a311d` is a wide rename/path cleanup across this project that
also changed one line of `chat2_papers/README.md`.

**Nothing is missing or damaged.** Every file is intact at `HEAD`, the test
suite passes, and `run_all.py` reproduces the results table. The history is
recorded here rather than rewritten: rewriting it would invalidate the commit
SHAs that `PREREGISTRATION_OUTCOME.md` and `README.md` cite as evidence of
ordering, and that ordering is part of this paper's argument.

The practical consequence for a reader: `git log -- chat1_protocol/` is the
correct way to see this project's history. A plain `git log` includes the other
paper's commits, and those two SHAs will appear in both views.

An earlier note recorded only `1b9beab`. The second, `51a311d`, was found by
enumerating every commit that touches both directories rather than trusting the
note — the same check the paper argues diagnostics should make on themselves.

## What is deliberately absent

Per-stay caches derived from restricted clinical data (`full_labels.npz`,
`full_d5_arrays.npz`, `full_d2_pools.pkl`) are excluded, and `.gitignore`
enforces it. Aggregate results are committed in `detectors/results/*.json`.
Regenerate the caches with your own credentials via `detectors/TODO.md`
section 0. `run_all.py` does not need them.

Detector 1's full-scale logs (`full1_probe.out`, `full1_bootstrap.out`) came back
empty when copied off the machine that produced them and are genuinely lost;
those figures are transcribed from run output rather than read from an archived
artifact. This is stated in `README.md` under "Archival gap" and should be
stated in the paper too.
