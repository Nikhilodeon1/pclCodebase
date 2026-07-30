#!/usr/bin/env bash
# A1 Phase 1: re-preprocess + EXP0 audit (incl. A2 computability) + EXP1 3-seed
# main-OOD, with the new SOFA Sepsis-3 labels. EXP3 and A3 are intentionally
# SKIPPED — gate on EXP1 (inspect PCL vs ERM) before spending on them.
set -e
[ -f runpod_env.sh ] && source runpod_env.sh   # prod mode + data paths

# ── Preflight: fail fast if not prod/CUDA (silent CPU on full data = brutal) ──
python - <<'PY'
import os, sys, torch
test = os.environ.get("PCL_TEST_MODE", "1") == "1"
if test:
    sys.exit("ABORT: PCL_TEST_MODE!=0 — runpod_env.sh not sourced? Prod mode required.")
if not torch.cuda.is_available():
    sys.exit("ABORT: CUDA not available — would fall back to CPU on full data.")
print(f"Preflight OK: prod mode, device=cuda ({torch.cuda.get_device_name(0)})")
PY

export PCL_SEED_STUDY=1     # gates off EXP2/4-7, high-k IRM, classical baselines
export PCL_SKIP_EXP3=1      # EXP1 only for Phase 1
CACHE=cache_a1_shared       # fresh shared cache => new SOFA labels, not stale ICD

# Clean result dirs so resume logic can't skip an experiment. The cache dir
# (cache_a1_shared) is a NEW name that only ever holds SOFA labels, so it is
# NOT wiped — this lets a CPU-prebuilt cache (scripts/build_cache.py) be reused.
rm -rf results_a1_s42 results_a1_s43 results_a1_s44

for S in 42 43 44; do
  echo "=== SEED $S ==="
  # EXP0 audit is seed-independent: run it only on the first seed, skip on 43/44.
  SKIP_AUDIT=$([ "$S" = 42 ] && echo 0 || echo 1)
  PCL_SEED=$S PCL_SKIP_AUDIT=$SKIP_AUDIT \
    RESULTS_DIR=results_a1_s$S CHECKPOINT_DIR=results_a1_s$S/ckpt CACHE_DIR=$CACHE \
    python run_paper_experiments.py
done

python scripts/aggregate_seeds.py \
  results_a1_s42/paper_results.json \
  results_a1_s43/paper_results.json \
  results_a1_s44/paper_results.json
echo "DONE. EXP1 tables: results_a1_s44/aggregated_tables.tex ; audit+A2 in each paper_results.json"
