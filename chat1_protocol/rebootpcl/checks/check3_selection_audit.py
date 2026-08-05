"""
CHECK 3 — OOD-contaminated hyperparameter selection.

Confound: a hyperparameter sweep that scores each candidate on a TARGET/OOD split
rather than a source-validation split. The reported "zero-shot" numbers are then
chosen with knowledge of the test distribution, which silently inflates them.

Diagnostic: static analysis. Walk the AST of a training script, find functions
that sweep a hyperparameter (a loop over a grid that trains a model per value),
and determine which loader flows into the evaluation call inside that loop. If the
evaluated loader traces back to an OOD/target collection rather than the
validation loader, flag it.

Ground truth is our own history: the same repository contains the buggy sweep
(commit c7cb42f) and the corrected one, so precision and recall are measured
against a real defect rather than a synthetic one.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(os.path.dirname(HERE), "fixtures")

# Parameter names that denote source-validation vs. target/OOD data.
VAL_NAMES = {"val_loader", "valid_loader", "validation_loader"}
OOD_NAMES = {"ood_loaders", "ood_loader", "test_loaders", "target_loader",
             "test_loader"}
EVAL_FNS = {"evaluate_model", "evaluate", "eval_on_loaders", "score_model"}
SWEEP_HINTS = ("lambda", "sweep", "grid", "ablation", "search", "tune")


def _origin(node, env):
    """Resolve an expression back to 'val', 'ood', or None.

    Handles the three shapes that actually occur: a bare name, a subscript such
    as ood_loaders[name], and a conditional expression whose branches disagree
    (the buggy code used `ood_loaders[n] if n else val_loader`, which is OOD
    whenever any OOD site exists).
    """
    if isinstance(node, ast.Name):
        if node.id in VAL_NAMES:
            return "val"
        if node.id in OOD_NAMES:
            return "ood"
        return env.get(node.id)
    if isinstance(node, ast.Subscript):
        return _origin(node.value, env)
    if isinstance(node, ast.IfExp):
        a, b = _origin(node.body, env), _origin(node.orelse, env)
        return "ood" if "ood" in (a, b) else a or b
    if isinstance(node, ast.Call):
        for a in node.args:
            o = _origin(a, env)
            if o:
                return o
    return None


def audit_function(fn):
    """Returns (is_sweep, findings) for one function definition."""
    name = fn.name.lower()
    has_hint = any(h in name for h in SWEEP_HINTS)
    # A sweep trains inside a loop over a grid of values.
    trains_in_loop = any(
        isinstance(n, ast.Call) and getattr(n.func, "id", "") == "train_model"
        for loop in ast.walk(fn) if isinstance(loop, (ast.For, ast.While))
        for n in ast.walk(loop))
    if not (has_hint and trains_in_loop):
        return False, []

    # Track local aliases: eval_loader = <expr>
    env = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            o = _origin(node.value, env)
            if o:
                env[node.targets[0].id] = o

    findings = []
    for loop in ast.walk(fn):
        if not isinstance(loop, (ast.For, ast.While)):
            continue
        for node in ast.walk(loop):
            if not (isinstance(node, ast.Call) and
                    getattr(node.func, "id", "") in EVAL_FNS):
                continue
            # First positional arg after the model is the loader.
            loader_arg = node.args[1] if len(node.args) > 1 else None
            if loader_arg is None:
                continue
            o = _origin(loader_arg, env)
            findings.append({"line": node.lineno, "resolved": o or "unknown",
                             "call": getattr(node.func, "id", "?")})
    return True, findings


def audit_file(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        is_sweep, findings = audit_function(fn)
        if not is_sweep:
            continue
        origins = {f["resolved"] for f in findings}
        # A sweep is contaminated if ANY in-loop evaluation reads OOD data and
        # no evaluation reads validation data.
        verdict = ("CONTAMINATED" if "ood" in origins and "val" not in origins
                   else "OK" if "val" in origins
                   else "INDETERMINATE")
        out.append({"function": fn.name, "line": fn.lineno,
                    "verdict": verdict, "findings": findings})
    return out


def main():
    cases = [("BUGGY (commit c7cb42f)", os.path.join(FIX, "sweep_BUGGY.py"), "CONTAMINATED"),
             ("FIXED (current)", os.path.join(FIX, "sweep_FIXED.py"), "OK")]
    print("=" * 74)
    print("CHECK 3 — OOD-contaminated hyperparameter selection (static audit)")
    print("=" * 74)
    tp = fp = fn_ = tn = 0
    for label, path, expected in cases:
        if not os.path.exists(path):
            print(f"  {label}: fixture missing ({path})")
            continue
        res = audit_file(path)
        print(f"\n{label}")
        if not res:
            print("   no sweep function detected")
        for r in res:
            print(f"   {r['function']}() line {r['line']}: {r['verdict']}")
            for f in r["findings"]:
                print(f"      eval at line {f['line']} reads {f['resolved'].upper()} data")
        got = res[0]["verdict"] if res else "INDETERMINATE"
        ok = (got == expected)
        print(f"   expected {expected} -> {'PASS' if ok else 'FAIL'}")
        if expected == "CONTAMINATED":
            tp += ok; fn_ += (not ok)
        else:
            tn += ok; fp += (not ok)
    print("\n" + "-" * 74)
    print(f"detections: TP={tp} FP={fp} FN={fn_} TN={tn}")
    print("verdict:", "DETECTOR VALIDATED" if (tp == 1 and tn == 1 and fp == 0 and fn_ == 0)
          else "NEEDS WORK")


if __name__ == "__main__":
    main()
