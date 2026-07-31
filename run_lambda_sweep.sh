#!/usr/bin/env bash
# Full-scale, val-selected lambda sweep at 17 variables (+ fixed DRO/IRM).
#
# PCL has so far always run at lambda=1.0, an unjustified config default: the old
# sweep selected on Site B (an OOD site), which is test-peeking. This run sweeps
# lambda in {0, 0.1, 0.5, 1, 2, 5} -- lambda=0 IS the ERM reference -- scores every
# setting on the Site-A held-out VALIDATION split, and picks lambda* by val AUROC
# only. OOD numbers are computed for reporting but never used for selection.
#
# Single seed (42): a sweep is a search, not a final estimate. Re-run the winning
# lambda at 3 seeds afterwards for the headline number.
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

OUT=results_lambda17
rm -rf "$OUT"

# z=EXP1 a=EXP2 b=EXP3 c=EXP4 d=EXP5 zxcxz=EXP6 f=EXP7 g=lambda h=subset
# 0 => run, 1 => skip. Here: EXP1 (for the in-run ERM/PCL/DRO reference) + the
# lambda sweep + the constraint-subset ablation; everything else off.
PCL_EXPANDED_VARS=1 PCL_SEED=42 PCL_SKIP_AUDIT=1 \
  RESULTS_DIR=$OUT CHECKPOINT_DIR=$OUT/ckpt CACHE_DIR=cache_v17_prod \
  python -c "
import run_paper_experiments as r
r.main(0, 1, 1, 1, 1, 1, 1, 0, 0)
"

echo
echo "==================== LAMBDA SWEEP COMPLETE ===================="
python - <<PY
import json, os
p = os.path.join("$OUT", "paper_results.json")
d = json.load(open(p))
lam = d.get("ablation_lambda", {})
best = lam.get("_val_selected_lambda")
print(f"{'lambda':>8} {'val':>8} " + " ".join(f"{s:>12}" for s in ["PhysioNet-B","MIMIC-IV","eICU"]))
for k in sorted([k for k in lam if k != "_val_selected_lambda"], key=float):
    e = lam[k]
    if not isinstance(e, dict):
        continue
    ood = e.get("ood") or {}
    row = f"{k:>8} {e.get('val') or float('nan'):>8.4f} " + " ".join(
        f"{(ood.get(s) if ood.get(s) is not None else float('nan')):>12.4f}"
        for s in ["PhysioNet-B","MIMIC-IV","eICU"])
    print(row + ("   <-- val-selected" if str(best)==str(k) else ""))
print(f"\nval-selected lambda* = {best}   (lambda=0.0 is the ERM reference)")
print("Selection used VALIDATION only; OOD columns are reported, not tuned on.")
PY
echo "Subset ablation + full JSON: $OUT/paper_results.json"
