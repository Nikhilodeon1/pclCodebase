"""
deadpcl, seed-variance extraction (external review item #5).

The paper currently reports seed-to-seed variability as a bare range ("4 to 7
AUROC points"), sourced from a sentence in chat2_papers/paper/build_urtc.py,
not from raw per-seed numbers anyone has looked at directly since. A range
from 3 seeds is closer to anecdotal than a real variance estimate, and the
paper should say min/max/std, not just a range, if the real numbers can be
recovered.

This script does NOT train or touch the GPU. It only searches the pod for
artifacts that might already contain the raw per-seed AUROC values behind
that range, in order of how cheap they are to use if found:

  1. A results JSON with a per-seed breakdown already computed (cheapest --
     just parse it).
  2. Per-sample prediction dumps (results/predictions/*.npz -- probs+labels
     per model x site x seed), if `set_prediction_dump` was ever pointed at
     a directory during the corrected 3-seed run. Cheap: pure CPU numpy/
     sklearn, no model loading.
  3. Nothing usable -- reports exactly what it searched so you know what's
     actually missing, rather than guessing a path with false confidence.

This deliberately searches broadly instead of assuming one hardcoded path:
unlike pcl-legacy2's PRETRAIN_CKPT_DIR (confirmed on the pod in a code
comment), nothing in this repo confirms exactly where the corrected 3-seed
sepsis run's raw output landed -- the naming convention below is inferred
from config.py / finetune_mortality.py's PCL_LEGACY2_PRETRAIN_DIR default
("results_lambda17/ckpt") and from the OLD (pre-split) pipeline's own
save-path convention (_archive/run_paper_experiments.py's "paper_results.json"
+ "results/predictions/*.npz"), not confirmed against the live pod.

Usage (on the pod, after runpod_env.sh):
    python extract_seed_variance.py                       # search /workspace
    python extract_seed_variance.py --search-root /some/other/path
"""
import argparse
import glob
import json
import os
import sys

# Some historical filenames in this codebase contain non-ASCII characters
# (e.g. sigma in noise-level ablation names); on Windows the console defaults
# to cp1252 and a bare print() on those crashes the whole search before it
# can report anything useful. Force UTF-8 with a safe fallback instead of
# failing on a filename.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass  # Python <3.7 or a stream that doesn't support reconfigure; best effort

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEADPCL_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_DEADPCL_ROOT)

SEEDS = (42, 43, 44)
METHODS = ("ERM", "PCL")
SITES = ("Site B", "MIMIC-IV", "eICU")

# Candidate result-JSON locations, in the order most likely to be right given
# what other scripts in this repo already confirm about the pod's layout.
# Relative to --search-root (default /workspace on the pod).
CANDIDATE_JSON_GLOBS = [
    "results_lambda17*/paper_results.json",
    "results_lambda17*_seed*/paper_results.json",
    "**/paper_results.json",
]

# Candidate per-sample prediction-dump directories.
CANDIDATE_NPZ_GLOBS = [
    "results_lambda17*/predictions/*.npz",
    "**/predictions/*.npz",
]


def _find(search_root, patterns):
    hits = []
    for pat in patterns:
        hits.extend(sorted(glob.glob(os.path.join(search_root, pat), recursive=True)))
    # de-dupe, keep order
    seen = set()
    out = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def _try_parse_seeded_json(path):
    """Best-effort parse: looks for a per-seed AUROC breakdown inside a
    paper_results.json-shaped file. Structure is NOT guaranteed (this repo's
    own history has at least two different result-JSON shapes across its
    pipeline versions), so this is deliberately defensive and returns None
    rather than guessing on anything that doesn't look right."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [SKIP] {path}: could not parse ({e})")
        return None

    # Known shape from _archive/run_paper_experiments.py's _serialize_results:
    # {"experiment_1": {"ERM": {"ood": {"Site B": {"auroc": ...}, ...}}, "PCL": {...}}}
    # This shape is PER-RUN, not per-seed -- a single paper_results.json from
    # one seed's run. If found, report it as one seed's worth of data and say
    # so; the caller needs to find the sibling files for the other 2 seeds.
    exp1 = data.get("experiment_1")
    if isinstance(exp1, dict) and "ERM" in exp1 and "PCL" in exp1:
        out = {}
        for method in METHODS:
            m = exp1.get(method, {})
            ood = m.get("ood", {})
            out[method] = {site: ood.get(site, {}).get("auroc") for site in SITES}
        return {"shape": "single_seed_experiment_1", "auroc": out}

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", default=os.environ.get("PCL_SEARCH_ROOT", "/workspace"),
                     help="Root to search under. Default /workspace (the pod's network "
                          "volume mount per runpod_env.sh). Use the repo root for a local "
                          "dry run of the search logic (will find nothing real).")
    args = ap.parse_args()
    root = args.search_root
    if not os.path.isdir(root):
        print(f"[search-root does not exist locally: {root} -- expected on the pod, "
              f"not this machine. Run this ON the pod.]")

    print(f"Searching {root} for per-seed result artifacts...\n")

    json_hits = _find(root, CANDIDATE_JSON_GLOBS)
    print(f"paper_results.json candidates found: {len(json_hits)}")
    for h in json_hits:
        print(f"  {h}")
    npz_hits = _find(root, CANDIDATE_NPZ_GLOBS)
    print(f"\nprediction .npz candidates found: {len(npz_hits)}")
    for h in npz_hits:
        print(f"  {h}")

    per_seed = {}  # seed_guess -> {"ERM": {...}, "PCL": {...}}
    for path in json_hits:
        parsed = _try_parse_seeded_json(path)
        if parsed is None:
            continue
        # Guess the seed from the directory name if it's tagged; otherwise
        # index by path so at least nothing overwrites silently.
        seed_key = next((s for s in SEEDS if f"seed{s}" in path or f"_s{s}" in path), path)
        per_seed[seed_key] = parsed["auroc"]
        print(f"\n[PARSED] {path} -> seed key '{seed_key}':")
        print(f"  {parsed['auroc']}")

    if npz_hits:
        print(f"\n{len(npz_hits)} prediction dump(s) found but not auto-parsed -- their "
              f"naming convention isn't confirmed. If the JSON search above didn't "
              f"produce 3 full seeds, read one filename here and tell me the pattern "
              f"(method/site/seed encoding) so this script can compute AUROC from them "
              f"directly instead (sklearn.roc_auc_score on probs+labels, no model load).")

    print("\n" + "=" * 70)
    if len(per_seed) >= 3:
        print(f"Found {len(per_seed)} seed-tagged results — computing min/max/std per site.")
        import statistics
        for method in METHODS:
            print(f"\n{method}:")
            for site in SITES:
                vals = [per_seed[s][method].get(site) for s in per_seed
                        if per_seed[s].get(method, {}).get(site) is not None]
                if len(vals) < 2:
                    print(f"  {site}: insufficient data ({len(vals)} seed(s))")
                    continue
                print(f"  {site}: n={len(vals)} mean={statistics.mean(vals):.4f} "
                      f"min={min(vals):.4f} max={max(vals):.4f} "
                      f"std={statistics.stdev(vals):.4f}")
        print("\nCopy the numbers above into the paper's Section 5 in place of the bare "
              "'4 to 7 AUROC points' range.")
    else:
        print(f"Found {len(per_seed)}/3 usable seeds. NOT enough to compute min/max/std "
              f"honestly — do not fabricate the missing seeds. Either point this script "
              f"at the right directory with --search-root, or tell me the real path/"
              f"filename pattern once you've looked at the pod yourself, and I'll fix "
              f"the search patterns above rather than guessing again.")


if __name__ == "__main__":
    main()
