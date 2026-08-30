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
                                                    # range. Run this FIRST.

    python scripts/run_confound_ablation.py --confound 2 --seed 42
    python scripts/run_confound_ablation.py --confound 4 --seed 42
                                                    # Real pretraining runs.
                                                    # Reintroduces one evaluation
                                                    # confound at a time (2 =
                                                    # pretraining leakage, 4 =
                                                    # Severinghaus circularity) on
                                                    # top of the otherwise-corrected
                                                    # pipeline, for the per-confound
                                                    # AUROC-drift table. Run seed 42
                                                    # alone first per confound — its
                                                    # wall-clock time is the only
                                                    # real cost estimate before the
                                                    # other two seeds. ~1-2 H100-hours
                                                    # total for both confounds x 3
                                                    # seeds, extrapolated from the
                                                    # documented ~6h/~30-run full
                                                    # suite in the withdrawn draft.

Both experiment scripts import `../pod_monitor.py` and call `watch_pod()` —
loud switch-pods alerts either direction, same as pcl-legacy2's scripts. See
`pod_monitor.py`'s docstring at the repo root.

Reuses `chat1_protocol/src` and `config.py` exactly like
`pcl-legacy2/scripts/finetune_mortality.py` does (`sys.path.insert` to the
`chat1_protocol/` directory) — no duplicated model/data/training code.
