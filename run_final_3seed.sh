#!/usr/bin/env bash
# Headline run: 3 seeds, 17 variables, lambda=0.5, EXP1 only (ERM / PCL / DRO-fixed).
#
# MODEL SELECTION PROTOCOL (state this in any write-up):
#   * Site-A held-out validation  -> early stopping / checkpoint selection.
#   * Site B                      -> VALIDATION DOMAIN used to select lambda
#                                    (full-scale sweep gave lambda*=0.5). Site B is
#                                    therefore NOT a test set in this protocol.
#   * MIMIC-IV and eICU           -> the true zero-shot test sets. Never used for
#                                    any selection decision.
# Site-A validation AUROC is nearly flat across lambda (~1pp) while OOD spans ~11pp,
# so training-domain validation alone cannot select lambda -- hence the validation
# DOMAIN. Report Site B numbers, but label them as selection-domain, not test.
set -e
[ -f runpod_env.sh ] && source runpod_env.sh

python - <<'PY'
import os, sys, torch
if os.environ.get("PCL_TEST_MODE", "1") == "1":
    sys.exit("ABORT: PCL_TEST_MODE!=0 required (source runpod_env.sh).")
if not torch.cuda.is_available():
    sys.exit("ABORT: CUDA unavailable.")
print(f"Preflight OK: prod, cuda ({torch.cuda.get_device_name(0)})")
PY

export PCL_SEED_STUDY=1     # EXP1 (+EXP3) only; no classical/high-k re-runs
export PCL_SKIP_EXP3=1      # EXP1 only
export PCL_EXPANDED_VARS=1  # 17 variables
export PCL_LAMBDA=${PCL_LAMBDA:-0.5}
echo "Using lambda = $PCL_LAMBDA"

for S in 42 43 44; do
  echo "=== final seed $S (lambda=$PCL_LAMBDA) ==="
  OUT=results_final_s$S
  rm -rf "$OUT"
  PCL_SEED=$S PCL_SKIP_AUDIT=1 \
    RESULTS_DIR=$OUT CHECKPOINT_DIR=$OUT/ckpt CACHE_DIR=cache_v17_prod \
    python run_paper_experiments.py
done

mkdir -p agg_final
python scripts/aggregate_seeds.py \
  results_final_s42/paper_results.json \
  results_final_s43/paper_results.json \
  results_final_s44/paper_results.json \
  --out-dir agg_final

echo
echo "==================== FINAL 3-SEED COMPLETE ===================="
echo "Tables: agg_final/aggregated_seeds.md"
echo "Reminder: Site B = selection domain; MIMIC-IV + eICU = zero-shot test."
