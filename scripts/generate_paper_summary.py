"""
Generate paper summary from run_paper_experiments.py output (paper_results.json).
Uses PhysioNet Challenge 2019 as primary dataset per plan.md.

Usage:
    python scripts/generate_paper_summary.py
    (run after: python run_paper_experiments.py)
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _fmt(v, digits=4):
    if v is None:
        return "N/A"
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}"
    return str(v)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_path = os.path.join(RESULTS_DIR, "paper_results.json")
    if not os.path.isfile(results_path):
        logging.error(
            f"{results_path} not found. "
            "Run 'python run_paper_experiments.py' first."
        )
        sys.exit(1)

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    exp1 = results.get("experiment_1", {})
    exp2 = results.get("experiment_2", {})
    exp3 = results.get("experiment_3", {})
    exp4 = results.get("experiment_4", {})
    exp5 = results.get("experiment_5", {})
    exp6 = results.get("experiment_6", {})
    exp7 = results.get("experiment_7", {})
    ablation_lam = results.get("ablation_lambda", {})
    ablation_sub = results.get("ablation_constraint_subset", {})
    classical = results.get("classical_baselines", {})

    md = "## Results Summary (PhysioNet Challenge 2019)\n\n"
    md += "> Primary dataset: PhysioNet Computing in Cardiology Challenge 2019.  \n"
    md += "> Train on Site A, evaluate zero-shot on Site B, MIMIC-IV demo, eICU demo.\n\n"

    # ── Experiment 1: Main OOD Generalization ────────────────────────────────
    md += "### Experiment 1: Main OOD Generalization (LOS > 3 Days)\n\n"
    md += "| Model | Val AUROC | PhysioNet-B | MIMIC-IV | eICU |\n"
    md += "|-------|-----------|-------------|----------|------|\n"

    for model_name in ["ERM", "PCL", "DRO"]:
        data = exp1.get(model_name, {})
        val = data.get("val", {})
        ood = data.get("ood", {})
        md += (
            f"| {model_name} "
            f"| {_fmt(val.get('auroc'))} "
            f"| {_fmt(ood.get('PhysioNet-B', {}).get('auroc'))} "
            f"| {_fmt(ood.get('MIMIC-IV', {}).get('auroc'))} "
            f"| {_fmt(ood.get('eICU', {}).get('auroc'))} |\n"
        )

    # ── Experiment 2: Feature Engineering Demolition ─────────────────────────
    if exp2:
        md += "\n### Experiment 2: Feature Engineering Demolition\n\n"
        md += "| Model | Val AUROC | PhysioNet-B | MIMIC-IV | eICU |\n"
        md += "|-------|-----------|-------------|----------|------|\n"
        for model_name in ["ERM+FE", "PCL+FE"]:
            data = exp2.get(model_name, {})
            val = data.get("val", {})
            ood = data.get("ood", {})
            md += (
                f"| {model_name} "
                f"| {_fmt(val.get('auroc'))} "
                f"| {_fmt(ood.get('PhysioNet-B', {}).get('auroc'))} "
                f"| {_fmt(ood.get('MIMIC-IV', {}).get('auroc'))} "
                f"| {_fmt(ood.get('eICU', {}).get('auroc'))} |\n"
            )

    # ── Experiment 3: Constraint Randomization ───────────────────────────────
    if exp3:
        md += "\n### Experiment 3: Constraint Randomization Ablation\n\n"
        md += "| Condition | OOD AUROC |\n"
        md += "|-----------|----------|\n"
        for cond, auroc in exp3.items():
            marker = " **" if "real" in cond.lower() else ""
            md += f"| {cond}{marker} | {_fmt(auroc)} |\n"

    # ── Experiment 4: Linear Probe ───────────────────────────────────────────
    if exp4:
        md += "\n### Experiment 4: Linear Probe Analysis\n\n"
        md += "| Target | ERM | PCL | Interpretation |\n"
        md += "|--------|-----|-----|----------------|\n"
        erm_probes = exp4.get("ERM", {})
        pcl_probes = exp4.get("PCL", {})
        for target in erm_probes:
            e = erm_probes.get(target)
            p = pcl_probes.get(target)
            if target == "site_id":
                interp = "PCL removes site signal" if (p and e and p < e - 0.05) else "Similar"
            else:
                interp = "Both preserve clinical signal" if (p and e and abs(p - e) < 0.1) else "Divergence"
            md += f"| {target} | {_fmt(e)} | {_fmt(p)} | {interp} |\n"

    # ── Experiment 5: Noise Robustness ───────────────────────────────────────
    if exp5:
        md += "\n### Experiment 5: Measurement Noise Robustness\n\n"
        md += "| Noise (mmHg) |"
        models = list(exp5.keys())
        for m in models:
            md += f" {m} |"
        md += "\n|" + "------|" * (len(models) + 1) + "\n"
        sigmas = sorted(exp5[models[0]].keys(), key=lambda x: float(x))
        for s in sigmas:
            md += f"| {s} |"
            for m in models:
                md += f" {_fmt(exp5[m].get(s))} |"
            md += "\n"

    # ── Experiment 6: OOD Detector ───────────────────────────────────────────
    if exp6:
        md += "\n### Experiment 6: PCL Violation as OOD Detector\n\n"
        md += "| OOD Site | Mean Score | Detection AUROC |\n"
        md += "|----------|-----------|----------------|\n"
        for site, data in exp6.items():
            md += f"| {site} | {_fmt(data.get('mean_score'))} | {_fmt(data.get('detection_auroc'))} |\n"

    # ── Experiment 7: IRM Failure Mode ────────────────────────────────────────
    if exp7:
        md += "\n### Experiment 7: IRM Failure Mode (High-k vs Low-k)\n\n"
        md += "| Method | Environments | Test AUROC |\n"
        md += "|--------|-------------|------------|\n"
        for method, data in exp7.items():
            n_envs = data.get("n_envs", "?")
            auroc = data.get("test_auroc")
            md += f"| {method} | {n_envs} | {_fmt(auroc)} |\n"

    # ── Ablation: Lambda Sensitivity ─────────────────────────────────────────
    if ablation_lam:
        md += "\n### Ablation: Lambda Sensitivity\n\n"
        md += "| λ | OOD AUROC |\n"
        md += "|---|----------|\n"
        for lam_val in sorted(ablation_lam.keys(), key=lambda x: float(x)):
            md += f"| {lam_val} | {_fmt(ablation_lam[lam_val])} |\n"

    # ── Ablation: Constraint Subset ──────────────────────────────────────────
    if ablation_sub:
        md += "\n### Ablation: Constraint Subset\n\n"
        md += "| Condition | OOD AUROC |\n"
        md += "|-----------|----------|\n"
        for cond, auroc in ablation_sub.items():
            md += f"| {cond} | {_fmt(auroc)} |\n"

    # ── Classical Baselines ──────────────────────────────────────────────────
    if classical:
        md += "\n### Classical ML Baselines\n\n"
        for site, tasks in classical.items():
            if not tasks:
                continue
            md += f"**{site}:**\n\n"
            for task_name, models in tasks.items():
                md += f"#### Task: {task_name}\n\n"
                md += "| Model | AUROC | AUPRC | Brier |\n"
                md += "|-------|-------|-------|-------|\n"
                for model_name, metrics in models.items():
                    md += (
                        f"| {model_name} "
                        f"| {_fmt(metrics.get('auroc'))} "
                        f"| {_fmt(metrics.get('auprc'))} "
                        f"| {_fmt(metrics.get('brier'))} |\n"
                    )
                md += "\n"

    # ── Write ────────────────────────────────────────────────────────────────
    md_path = os.path.join(RESULTS_DIR, "paper_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    json_path = os.path.join(RESULTS_DIR, "paper_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logging.info("Wrote %s", md_path)
    logging.info("Wrote %s", json_path)
    try:
        print(md)
    except UnicodeEncodeError:
        print(md.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
