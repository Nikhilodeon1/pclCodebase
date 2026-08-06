"""Detector 3 against third-party repositories: does it generalize?

Reports the PRE-EXTENSION matrix -- detector 3 exactly as it was written for
this project, with no accommodation for other codebases' naming.

One measurement decision matters more than the numbers. Detector 3 returns three
verdicts, not two: CONTAMINATED, OK, and INDETERMINATE (no sweep function was
recognised at all). Folding INDETERMINATE into "true negative" would inflate the
TN count with files the detector never actually analysed -- the same
inflated-denominator problem as counting near-ceiling controls as evidence. A
clean file only counts as a TN when the detector recognised a sweep and judged it
OK. INDETERMINATE is reported in its own column and scored as a MISS on both
sides: on an expected positive it is a false negative, and on an expected
negative it is NOT credit.

    python rebootpcl/external/run_repos.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.checks.check3_selection_audit import (
    DEFAULT_VOCAB, Vocab, audit_file, verdict_for_file)
from rebootpcl.external.repos import TARGETS, resolved_files

# EXTENDED vocabulary -- FITTED to third-party code AFTER seeing detector 3 fail
# on it. Every name here was added because it appears in the repositories under
# test, so any result it produces is fitted, not generalized. It exists to answer
# one question: is detector 3's failure LEXICAL (it does not know other people's
# names) or STRUCTURAL (the code shape it looks for does not occur)? If the
# extended vocabulary still finds nothing, the failure is structural and no
# amount of renaming would have fixed it.
EXTENDED = Vocab(
    val=set(DEFAULT_VOCAB.val) | {
        "val_loader", "valid_loader", "validation_loader", "val_reader",
        "val_data", "val_X", "val_gen", "in_splits", "train_envs",
        "val_acc", "env_out_acc"},
    ood=set(DEFAULT_VOCAB.ood) | {
        "test_loader", "test_reader", "test_data", "test_X", "test_gen",
        "out_splits", "test_envs", "test_env", "test_out_acc", "test_in_acc",
        "target_data"},
    eval_fns=set(DEFAULT_VOCAB.eval_fns) | {
        "predict", "predict_proba", "score", "test", "validate", "run_acc",
        "accuracy", "evaluate_model", "print_metrics_binary", "sweep_acc"},
    train_fns=set(DEFAULT_VOCAB.train_fns) | {
        "train", "fit", "train_epoch", "run_experiment", "main", "train_step",
        "Trainer", "train_model"},
    hints=tuple(set(DEFAULT_VOCAB.hints) | {
        "select", "hparam", "hyperparam", "run", "main", "collect", "acc"}),
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")


def audit(path, vocab=None):
    """(verdict, per-sweep findings). Missing files are reported, not skipped."""
    if not os.path.exists(path):
        return "MISSING", []
    try:
        res = audit_file(path, vocab)
    except SyntaxError as e:          # a py2-era file in a py3 parser
        return f"UNPARSEABLE ({e.__class__.__name__})", []
    return verdict_for_file(res), res


def score(vocab, label, verbose=True):
    """Confusion counts under one vocabulary, keeping INDETERMINATE separate."""
    tp = fp = tn = fn = 0
    indet_pos = indet_neg = 0
    rows = []
    for repo, path, expected, why in resolved_files():
        verdict, res = audit(path, vocab)
        flagged = (verdict == "CONTAMINATED")
        recognised = verdict in ("CONTAMINATED", "OK")

        if expected:
            if flagged:
                tp += 1; outcome = "TP"
            elif recognised:
                fn += 1; outcome = "FN (judged OK)"
            else:
                fn += 1; indet_pos += 1; outcome = "FN (no sweep recognised)"
        else:
            if flagged:
                fp += 1; outcome = "FP"
            elif recognised:
                tn += 1; outcome = "TN"
            else:
                indet_neg += 1; outcome = "no credit (no sweep recognised)"

        if verbose:
            print(f"{repo:<20} {os.path.basename(path):<28} "
                  f"expected={'FLAG' if expected else 'clean':<5} "
                  f"verdict={verdict:<14} -> {outcome}")
            for r in res:
                print(f"      sweep {r['function']}() line {r['line']}: "
                      f"{r['verdict']}")
        rows.append({"repo": repo, "file": os.path.basename(path),
                     "expected": expected, "verdict": verdict,
                     "outcome": outcome, "rationale": why,
                     "sweeps_found": len(res)})

    counts = {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
              "indeterminate_on_positives": indet_pos,
              "indeterminate_on_negatives": indet_neg,
              "recognised": tp + fp + tn + (fn - indet_pos)}
    print(f"\n{label}:  TP={tp}  FP={fp}  FN={fn}  TN={tn}   "
          f"(recognised a sweep in {counts['recognised']}/{len(rows)} files)")
    return counts, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(RESULTS, "external3_repos.json"))
    args = ap.parse_args()

    print("=" * 78)
    print("DETECTOR 3 — third-party repositories (PRE-EXTENSION, unmodified)")
    print("=" * 78)
    for t in TARGETS:
        tag = "clinical" if t["clinical"] else "NOT clinical (disclosed)"
        print(f"  {t['name']:<22} {t['commit'][:12]}  {tag}")
        print(f"      {t['paper']}")
    print()

    pre_counts, pre_rows = score(None, "PRE-EXTENSION (detector as written)")
    print("\nINDETERMINATE is NOT counted as a true negative. The detector did "
          "not analyse those files and judge them clean; it failed to recognise "
          "a sweep in them at all. Counting them as TN would credit "
          "non-detection as specificity.")

    print("\n" + "=" * 78)
    print("POST-EXTENSION (vocabulary FITTED to these repositories)")
    print("=" * 78)
    d = EXTENDED.diff(DEFAULT_VOCAB)
    for k, v in d.items():
        if v:
            print(f"  +{k}: {', '.join(v)}")
    print()
    post_counts, post_rows = score(EXTENDED, "POST-EXTENSION (fitted)")

    print("\n" + "-" * 78)
    if post_counts["recognised"] == 0:
        print("The extended vocabulary recognised a sweep in NO additional "
              "file. Detector 3's failure on external code is therefore "
              "STRUCTURAL, not lexical: it looks for a function whose name "
              "carries a sweep hint, which calls a training function inside a "
              "loop, and which then calls an evaluation function whose loader "
              "argument can be traced to a named val/OOD variable. Third-party "
              "selection code does not take that shape -- DomainBed selects by "
              "argmax over a table of recorded metrics, with no training call "
              "anywhere in the selecting function. Renaming would not have "
              "fixed this, so the post-extension arm is reported as a NEGATIVE "
              "RESULT rather than as a fitted improvement.")
    else:
        print(f"The extended vocabulary recognised "
              f"{post_counts['recognised']} file(s) the original missed. This "
              "arm is FITTED, not generalized: every added name was taken from "
              "the repositories under test after observing the failure. Report "
              "it beside the pre-extension matrix, never instead of it.")

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump({"pre_extension": pre_counts,
                   "post_extension_fitted": post_counts,
                   "vocabulary_diff": EXTENDED.diff(DEFAULT_VOCAB),
                   "rows_pre": pre_rows, "rows_post": post_rows,
                   "repos": [{k: t[k] for k in
                              ("name", "url", "commit", "paper", "clinical")}
                             for t in TARGETS]}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
