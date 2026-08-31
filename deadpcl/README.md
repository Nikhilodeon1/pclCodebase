# Dead PCL — ML4H 2026 Findings negative-results writeup

Full rigorous negative-result paper on the original PCL investigation. See
[PROJECTS.md](../PROJECTS.md) for status and byline.

## Contents

    pcl_findings_draft.tex     the paper (edit this)
    ref.bib                    bibliography
    paper_temp.tex             untouched copy of the official ML4H template
    scripts/                   GPU-pod experiment code supporting the paper's
                                per-confound ablation and seed-variance sections

## Scripts (run on the RunPod pod — real GPU spend, real data)

    python scripts/extract_seed_variance.py       # READ-ONLY, no GPU, no training.
                                                    # Searches the pod for existing
                                                    # per-seed AUROC data behind the
                                                    # paper's "4-7pp" seed-variance
                                                    # range. Free to run any time.

`scripts/run_confound_ablation.py` (per-confound magnitude, external review
item #7) exists and was verified against the real API (every call matches an
existing function signature; the confound-4 loss variant reuses an
already-exercised code path in `chat1_protocol/src/ablations.py`), but is
**DEFERRED, not planned before submission** — 2026-08-24 decision. Real cost
came out to ~1 GPU-hour for confound 4 alone but ~8-9 GPU-hours for confound
2 (its pretraining pool is ~22x larger — all 4 sites pooled vs. source-only),
call it ~$30 on an H100 SXM pod for both at 3 seeds each. Given real risk
the result lands inside this study's own 4-7pp seed-noise floor (an
individual confound's marginal contribution can easily be smaller than the
noise that already sank the IRM-30 claim) and this is upgrading an
already-solid paper rather than fixing something broken, skipped as not
worth the budget. Reflected honestly in the paper's Limitations rather than
left as a silent gap. Script kept in case the calculus changes (cheaper GPU
time, more budget, or a reviewer makes it a real ask post-submission) —
run seed 42 alone per confound first if it's ever revisited.

Both experiment scripts import `../pod_monitor.py` and call `watch_pod()` —
loud switch-pods alerts either direction, same as pcl-legacy2's scripts. See
`pod_monitor.py`'s docstring at the repo root.

Reuses `chat1_protocol/src` and `config.py` exactly like
`pcl-legacy2/scripts/finetune_mortality.py` does (`sys.path.insert` to the
`chat1_protocol/` directory) — no duplicated model/data/training code.
