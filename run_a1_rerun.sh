#!/usr/bin/env bash
# A1 rerun: 3-seed main OOD (EXP1) + randomization (EXP3) with the new SOFA
# Sepsis-3 labels, then aggregate into camera-ready tables. Gated on EXP1 per
# plan — inspect PCL vs ERM before spending compute on A3.
set -e
[ -f runpod_env.sh ] && source runpod_env.sh   # sets prod mode + data paths

export PCL_SEED_STUDY=1        # EXP1 + EXP3 only (skips heavy EXP2/4-7, classical, high-k IRM)
CACHE=cache_a1_shared          # fresh cache dir => SOFA labels (not stale ICD cache)

for S in 42 43 44; do
  echo "=== SEED $S ==="
  PCL_SEED=$S RESULTS_DIR=results_a1_s$S CHECKPOINT_DIR=results_a1_s$S/ckpt CACHE_DIR=$CACHE \
    python run_paper_experiments.py
done

python scripts/aggregate_seeds.py \
  results_a1_s42/paper_results.json \
  results_a1_s43/paper_results.json \
  results_a1_s44/paper_results.json
echo "Done. Tables: results_a1_s44/aggregated_tables.tex"
