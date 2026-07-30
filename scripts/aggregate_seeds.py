"""
Seed aggregation + camera-ready table generator (professor feedback W5).

Reads the per-seed ``paper_results.json`` files produced by
``run_paper_experiments.py`` (one per seed, e.g. PCL_SEED=42/43/44 written to
separate RESULTS_DIRs) and produces, for the main tables:

  * mean +/- std across seeds for AUROC, AUPRC, and Brier at every site;
  * seed-level PAIRED differences vs. ERM (the "+6.4 +/- 1.6 pp" gaps);
  * the per-run bootstrap CIs each seed already carries (auroc_ci/auprc_ci),
    surfaced so the author can report them;
  * LaTeX rows matching Table 2 (main OOD) and Table 4 (randomization).

This script computes NOTHING new about the model -- it only aggregates numbers
the runner already wrote. It never invents seeds: if you pass one JSON, std is
reported as 0 and flagged.

Usage:
    python scripts/aggregate_seeds.py PATH1.json PATH2.json PATH3.json
    python scripts/aggregate_seeds.py --glob "results_seed*/paper_results.json"
    python scripts/aggregate_seeds.py            # defaults to results/paper_results.json
"""
import os
import sys
import glob as globmod
import json
import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SITES = ["val", "PhysioNet-B", "MIMIC-IV", "eICU"]
METRICS = ["auroc", "auprc", "brier"]
METHOD_ORDER = ["ERM", "DRO", "IRM", "ERM+FE",
                "DenoisingAE", "NoiseAug", "DropoutWD", "Mixup",
                "PCL", "PCL+FE"]


# ── metric extraction ────────────────────────────────────────────────────────
def _get_site_block(method_block, site):
    """experiment_1[method] = {"val": {...}, "ood": {site: {...}}}."""
    if method_block is None:
        return None
    if site == "val":
        return method_block.get("val")
    return method_block.get("ood", {}).get(site)


def _collect(seed_results, exp_key="experiment_1"):
    """Returns {method: {site: {metric: [per-seed values]}, '_ci': {...}}}."""
    methods = {}
    for res in seed_results:
        exp = res.get(exp_key, {})
        for method, mblock in exp.items():
            m = methods.setdefault(method, {})
            for site in SITES:
                block = _get_site_block(mblock, site)
                if block is None:
                    continue
                s = m.setdefault(site, {metric: [] for metric in METRICS})
                for metric in METRICS:
                    v = block.get(metric)
                    if v is not None:
                        s[metric].append(float(v))
                # keep the last seen per-run bootstrap CIs for reference
                s.setdefault("_ci", {})
                for ci_key in ("auroc_ci", "auprc_ci"):
                    if block.get(ci_key) is not None:
                        s["_ci"].setdefault(ci_key, []).append(block[ci_key])
    return methods


def _mean_std(vals):
    if not vals:
        return None, None, 0
    arr = np.array(vals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0
    return mean, std, len(arr)


def _paired_diff(method_vals, erm_vals):
    """Seed-level paired difference (method - ERM), aligned by seed order.
    Only pairs positions present in both; returns (mean, std, n, per_seed)."""
    n = min(len(method_vals), len(erm_vals))
    if n == 0:
        return None, None, 0, []
    diffs = [method_vals[i] - erm_vals[i] for i in range(n)]
    arr = np.array(diffs)
    std = float(arr.std(ddof=1)) if n >= 2 else 0.0
    return float(arr.mean()), std, n, diffs


# ── formatting ───────────────────────────────────────────────────────────────
def _pm(mean, std, digits=3):
    """'$0.865{\\pm}.006$' — matches the paper's existing table style."""
    if mean is None:
        return "---"
    m = f"{mean:.{digits}f}"
    s = f"{std:.{digits}f}".lstrip("0")  # ".006"
    return f"${m}{{\\pm}}{s}$"


def _pp(mean, std):
    if mean is None:
        return "---"
    return f"{mean*100:+.1f}{{\\pm}}{std*100:.1f}pp"


# ── main aggregation ─────────────────────────────────────────────────────────
def aggregate(seed_results):
    out = {"n_seeds": len(seed_results), "experiment_1": {}, "paired_vs_ERM": {},
           "experiment_3": {}, "ablation_lambda": {}, "ablation_constraint_subset": {}}

    m1 = _collect(seed_results, "experiment_1")
    # Fold A3 regularization baselines (experiment_8) in as extra methods; they
    # share the {val, ood} schema and are disjoint from EXP1 method names.
    m8 = _collect(seed_results, "experiment_8_regularization")
    for method, sites in m8.items():
        m1.setdefault(method, {}).update(sites)
    erm = m1.get("ERM", {})
    for method, sites in m1.items():
        out["experiment_1"][method] = {}
        for site in SITES:
            if site not in sites:
                continue
            block = {}
            for metric in METRICS:
                mean, std, n = _mean_std(sites[site][metric])
                block[metric] = {"mean": mean, "std": std, "n_seeds": n}
            block["per_run_ci"] = sites[site].get("_ci", {})
            out["experiment_1"][method][site] = block

        # paired diffs vs ERM (AUROC) — the headline gaps
        if method != "ERM" and method in m1:
            out["paired_vs_ERM"].setdefault(method, {})
            for site in SITES:
                if site in sites and site in erm:
                    dm, ds, dn, per = _paired_diff(sites[site]["auroc"], erm[site]["auroc"])
                    if dm is not None:
                        out["paired_vs_ERM"][method][site] = {
                            "mean": dm, "std": ds, "n_seeds": dn, "per_seed": per}

    # experiment_3 (randomization): flat {condition: value}
    cond_vals = {}
    for res in seed_results:
        for cond, v in res.get("experiment_3", {}).items():
            if v is not None:
                cond_vals.setdefault(cond, []).append(float(v))
    for cond, vals in cond_vals.items():
        mean, std, n = _mean_std(vals)
        out["experiment_3"][cond] = {"mean": mean, "std": std, "n_seeds": n}

    # ablations (flat {key: value})
    for abl in ("ablation_lambda", "ablation_constraint_subset"):
        acc = {}
        for res in seed_results:
            for k, v in res.get(abl, {}).items():
                if v is not None:
                    acc.setdefault(k, []).append(float(v))
        for k, vals in acc.items():
            mean, std, n = _mean_std(vals)
            out[abl][k] = {"mean": mean, "std": std, "n_seeds": n}

    return out


# ── LaTeX emission ───────────────────────────────────────────────────────────
def latex_main_table(agg, metric="auroc"):
    lines = []
    label = {"auroc": "AUROC", "auprc": "AUPRC", "brier": "Brier"}[metric]
    lines.append(f"% Main OOD table ({label}), mean+/-std over {agg['n_seeds']} seeds")
    lines.append(r"\begin{tabular}{@{}lcccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Method & Site A (ID) & Site B & MIMIC-IV & eICU \\")
    lines.append(r"\midrule")
    exp1 = agg["experiment_1"]
    for method in METHOD_ORDER:
        if method not in exp1:
            continue
        cells = []
        for site in SITES:
            b = exp1[method].get(site, {}).get(metric)
            cells.append(_pm(b["mean"], b["std"]) if b else "---")
        name = f"\\textbf{{{method} (ours)}}" if method == "PCL" else method
        row = f"{name} & " + " & ".join(cells) + r" \\"
        lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def latex_randomization_table(agg):
    lines = ["% Randomization table (Site B OOD AUROC), mean+/-std"]
    lines.append(r"\begin{tabular}{@{}lc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Constraint Type & OOD AUROC \\")
    lines.append(r"\midrule")
    items = sorted(agg["experiment_3"].items(),
                   key=lambda kv: (kv[1]["mean"] if kv[1]["mean"] is not None else 0))
    for cond, b in items:
        name = f"\\textbf{{{cond}}}" if "real" in cond.lower() else cond
        lines.append(f"{name} & {_pm(b['mean'], b['std'])} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def markdown_summary(agg):
    md = [f"# Seed-aggregated results ({agg['n_seeds']} seeds)\n"]
    if agg["n_seeds"] < 2:
        md.append("> **WARNING: only 1 seed provided — std is 0 and not meaningful.**\n")
    md.append("## Main OOD (mean +/- std)\n")
    for metric in METRICS:
        md.append(f"\n### {metric.upper()}\n")
        md.append("| Method | " + " | ".join(SITES) + " |")
        md.append("|" + "---|" * (len(SITES) + 1))
        for method in METHOD_ORDER:
            if method not in agg["experiment_1"]:
                continue
            cells = []
            for site in SITES:
                b = agg["experiment_1"][method].get(site, {}).get(metric)
                cells.append(f"{b['mean']:.3f}±{b['std']:.3f}" if b and b["mean"] is not None else "—")
            md.append(f"| {method} | " + " | ".join(cells) + " |")
    md.append("\n## Paired difference vs ERM (AUROC, pp)\n")
    md.append("| Method | " + " | ".join(SITES) + " |")
    md.append("|" + "---|" * (len(SITES) + 1))
    for method, sites in agg["paired_vs_ERM"].items():
        cells = []
        for site in SITES:
            d = sites.get(site)
            cells.append(f"{d['mean']*100:+.1f}±{d['std']*100:.1f}" if d else "—")
        md.append(f"| {method} | " + " | ".join(cells) + " |")
    md.append("\n## Randomization (Site B OOD AUROC)\n")
    md.append("| Condition | mean±std |")
    md.append("|---|---|")
    for cond, b in sorted(agg["experiment_3"].items(),
                          key=lambda kv: (kv[1]["mean"] or 0)):
        md.append(f"| {cond} | {b['mean']:.3f}±{b['std']:.3f} |")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="per-seed paper_results.json files")
    ap.add_argument("--glob", help="glob pattern for per-seed JSONs")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    paths = list(args.paths)
    if args.glob:
        paths += globmod.glob(args.glob)
    if not paths:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default = os.path.join(here, "results", "paper_results.json")
        if os.path.exists(default):
            paths = [default]
    if not paths:
        logging.error("No result JSONs found. Pass paths or --glob.")
        sys.exit(1)

    seed_results = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            seed_results.append(json.load(f))
        logging.info("Loaded %s", p)
    if len(seed_results) < 2:
        logging.warning("Only %d seed(s) — std will be 0 and is not a real "
                        "variance estimate.", len(seed_results))

    agg = aggregate(seed_results)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(paths[0]))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "aggregated_seeds.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    tex = []
    for metric in METRICS:
        tex.append(latex_main_table(agg, metric))
        tex.append("")
    tex.append(latex_randomization_table(agg))
    tex_str = "\n".join(tex)
    with open(os.path.join(out_dir, "aggregated_tables.tex"), "w", encoding="utf-8") as f:
        f.write(tex_str)

    md = markdown_summary(agg)
    with open(os.path.join(out_dir, "aggregated_seeds.md"), "w", encoding="utf-8") as f:
        f.write(md)

    logging.info("Wrote aggregated_seeds.json / aggregated_tables.tex / aggregated_seeds.md to %s", out_dir)
    print("\n" + md)
    print("\n" + "=" * 60 + "\nLaTeX (main AUROC table):\n" + "=" * 60)
    print(latex_main_table(agg, "auroc"))


if __name__ == "__main__":
    main()
