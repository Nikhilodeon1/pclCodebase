#!/usr/bin/env bash
# Feature-expansion A/B: 17-variable (expanded) vs 9-variable (core) inputs,
# 3 seeds each, EXP1 main-OOD only. Answers the headline question honestly:
# "do the added sepsis variables (Temp/Resp/WBC/Lactate/BUN/Creatinine/K/Glucose)
# improve cross-hospital transfer?" -- for BOTH ERM and PCL, so the comparison
# stays fair and neither arm is privileged.
#
# Caches are per-arm because the two arms have different tensor widths (17 vs 9);
# they must never be shared. Build the 17-var cache first on a big-RAM CPU box:
#   CACHE_DIR=cache_v17_prod python scripts/build_cache.py
# The 9-var cache is built automatically on first use by the CORE arm.
set -e
[ -f runpod_env.sh ] && source runpod_env.sh

python - <<'PY'
import os, sys, torch
if os.environ.get("PCL_TEST_MODE", "1") == "1":
    sys.exit("ABORT: PCL_TEST_MODE!=0 required (source runpod_env.sh).")
if not torch.cuda.is_available():
    sys.exit("ABORT: CUDA unavailable — would silently run on CPU.")
print(f"Preflight OK: prod, cuda ({torch.cuda.get_device_name(0)})")
PY

export PCL_SEED_STUDY=1     # EXP1 (+EXP3 unless skipped) only
export PCL_SKIP_EXP3=1      # EXP1 only — gate before spending on anything else

run_arm () {                # $1 = arm name, $2 = PCL_EXPANDED_VARS, $3 = cache dir
  local ARM=$1 EXPANDED=$2 CACHE=$3
  echo "################ ARM: $ARM (PCL_EXPANDED_VARS=$EXPANDED) ################"
  for S in 42 43 44; do
    echo "=== $ARM seed $S ==="
    local OUT=results_${ARM}_s$S
    rm -rf "$OUT"
    # EXP0 audit is seed-independent: compute on the first seed only.
    local SKIP=$([ "$S" = 42 ] && echo 0 || echo 1)
    PCL_EXPANDED_VARS=$EXPANDED PCL_SEED=$S PCL_SKIP_AUDIT=$SKIP \
      RESULTS_DIR=$OUT CHECKPOINT_DIR=$OUT/ckpt CACHE_DIR=$CACHE \
      python run_paper_experiments.py
  done
  mkdir -p agg_${ARM}
  python scripts/aggregate_seeds.py \
    results_${ARM}_s42/paper_results.json \
    results_${ARM}_s43/paper_results.json \
    results_${ARM}_s44/paper_results.json \
    --out-dir agg_${ARM}
}

run_arm expanded 1 cache_v17_prod
run_arm core     0 cache_v9_prod

echo
echo "==================== A/B COMPLETE ===================="
echo "EXPANDED (17 vars): agg_expanded/aggregated_seeds.md"
echo "CORE      (9 vars): agg_core/aggregated_seeds.md"
echo "Compare ERM and PCL across arms. Report whichever way it lands."
