#!/usr/bin/env bash
# Gate check, seed replication + baselines.
#
# Phase 1 (training, expensive): re-run the lambda sweep at seeds 43 and 44 so the
#   label-free-selection result can be tested for seed stability. Seed 42 already
#   exists in results_lambda17/, so only 2 extra sweeps are needed.
# Phase 2 (inference, minutes): run gate_check.py on all three seeds. It now scores
#   four label-free signals -- violation, entropy, recon_mse, repr_dist -- so the
#   reviewer-requested controls come along at no extra training cost.
#
# The question: does constraint violation on unlabeled target data select lambda
# better than (a) training-domain validation, (b) entropy, (c) plain reconstruction
# error, (d) representation distance -- and does it hold across seeds?
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

export PCL_EXPANDED_VARS=1
export PCL_SEED_STUDY=1     # skip high-k IRM + classical baselines
export PCL_SKIP_AUDIT=1

# ── Phase 1: lambda sweeps at seeds 43, 44 ───────────────────────────────────
# main(z,a,b,c,d,zxcxz,f,g,h): 0=run, 1=skip. Only g (lambda sweep) is enabled.
for S in 43 44; do
  OUT=results_lambda17_s$S
  if [ -f "$OUT/paper_results.json" ] && python -c "
import json,sys; d=json.load(open('$OUT/paper_results.json'))
sys.exit(0 if 'ablation_lambda' in d else 1)" 2>/dev/null; then
    echo "=== seed $S sweep already present, skipping ==="
    continue
  fi
  echo "=== lambda sweep, seed $S ==="
  rm -rf "$OUT"
  PCL_SEED=$S RESULTS_DIR=$OUT CHECKPOINT_DIR=$OUT/ckpt CACHE_DIR=cache_v17_prod \
    python -c "
import run_paper_experiments as r
r.main(1, 1, 1, 1, 1, 1, 1, 0, 1)
"
done

# ── Phase 2: gate check on all three seeds ───────────────────────────────────
run_gate () {   # $1 = label, $2 = results dir
  echo
  echo "################ GATE CHECK: $1 ################"
  PCL_EXPANDED_VARS=1 CACHE_DIR=cache_v17_prod \
    python OOD-iclr/gate_check.py \
      --ckpt-dir "$2/ckpt" --results "$2/paper_results.json" \
      --out "OOD-iclr/gate_$1.json"
}

run_gate seed42 results_lambda17
run_gate seed43 results_lambda17_s43
run_gate seed44 results_lambda17_s44

echo
echo "==================== GATE SEEDS COMPLETE ===================="
echo "Per-seed JSON: OOD-iclr/gate_seed4{2,3,4}.json"
python OOD-iclr/summarize_gate.py || true
