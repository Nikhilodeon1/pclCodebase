# PCL Legacy2

Tests whether the URTC paper's finding (source-validation and reconstruction-error
selection criteria beat OOD-blind baselines) generalizes across clinical tasks,
beyond the single sepsis task URTC covered. Co-developed with AI club members.

Target: ML4H 2026 Proceedings, deadline Sept 10 2026 11:59 PM AoE. Track decision
(Proceedings vs. Findings) made after results are in, not before.

**Not `chat3_generalization/`.** That is a dead, unrelated conformal-prediction
attempt with a similar-sounding name — no shared code, no shared findings,
don't look there for anything in this project.

**Not `chat1_protocol/` or `chat2_papers/`** either — see `../PROJECTS.md` for
those two.

## Status

Blocked on RunPod pod being powered on (network volume persists while pod is
off; the H100 pod itself does not run 24/7). No trained ERM/PCL/DRO checkpoints
or full-scale cohorts exist locally — only demo-scale, wrong-task (LOS not
sepsis), test-mode-sized artifacts exist in `../_archive/results/checkpoints/`,
which are NOT what this project reuses.

## Authorship

Undecided. Spec lists Dr. Lin as co-author (same as URTC), but this is now a
group effort with AI club members contributing. Who is author vs. contributor
needs deciding before submission, not left open until then.

## Plan

1. Locate + verify trained ERM/PCL/DRO checkpoints and preprocessed PhysioNet
   2019 / MIMIC-IV / eICU-CRD cohorts on the RunPod network volume. Reuse only,
   no retraining, no re-preprocessing.
2. Add task labels: mortality first, then LOS, then decompensation
   (Harutyunyan et al. definition) only if budget/time remain after the first two.
3. Fine-tune only (frozen pretrained encoder, two-phase unfreezing head)
   per task x method (ERM/PCL/DRO) x 3 seeds, matching URTC protocol exactly.
4. Extend URTC's selection-criteria comparison with two new inference-time
   baselines: ATC and MMD. Same metrics (Spearman rank correlation, selection
   regret), 3 seeds, 3 target sites, per task.
5. Check reconstruction-error ranking for confound with masked-pretraining vs.
   physiology-constraint reliance, per task. Disclose if real.
6. Tag each task by output structure (static binary / continuous / dense
   time-series) before looking at whether results split by task.

Gate: report back after mortality (steps 1-4 for that task only) before
touching LOS or decompensation.

Flag immediately: any task needing more than light fine-tuning (new output
head), or compute approaching the $50 ceiling.
