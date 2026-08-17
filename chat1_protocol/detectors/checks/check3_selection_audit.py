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
TRAIN_FNS = {"train_model"}

# The four knobs above are this project's naming conventions. They are grouped
# into a Vocab so an EXTENDED vocabulary can be supplied without editing the
# defaults, which keeps "as written" and "fitted to external code" separable and
# lets the difference between them be reported as a diff.
class Vocab:
    def __init__(self, val=None, ood=None, eval_fns=None, train_fns=None,
                 hints=None):
        self.val = set(val or VAL_NAMES)
        self.ood = set(ood or OOD_NAMES)
        self.eval_fns = set(eval_fns or EVAL_FNS)
        self.train_fns = set(train_fns or TRAIN_FNS)
        self.hints = tuple(hints or SWEEP_HINTS)

    def diff(self, other):
        """What this vocabulary adds relative to `other`."""
        return {"val": sorted(self.val - other.val),
                "ood": sorted(self.ood - other.ood),
                "eval_fns": sorted(self.eval_fns - other.eval_fns),
                "train_fns": sorted(self.train_fns - other.train_fns),
                "hints": sorted(set(self.hints) - set(other.hints))}


DEFAULT_VOCAB = None   # built after the module-level names exist


def _origin(node, env, vocab=None):
    """Resolve an expression back to 'val', 'ood', or None.

    Handles the three shapes that actually occur: a bare name, a subscript such
    as ood_loaders[name], and a conditional expression whose branches disagree
    (the buggy code used `ood_loaders[n] if n else val_loader`, which is OOD
    whenever any OOD site exists).
    """
    vocab = vocab or DEFAULT_VOCAB
    if isinstance(node, ast.Name):
        if node.id in vocab.val:
            return "val"
        if node.id in vocab.ood:
            return "ood"
        return env.get(node.id)
    if isinstance(node, ast.Subscript):
        return _origin(node.value, env, vocab)
    if isinstance(node, ast.IfExp):
        a, b = _origin(node.body, env, vocab), _origin(node.orelse, env, vocab)
        return "ood" if "ood" in (a, b) else a or b
    if isinstance(node, ast.Call):
        for a in node.args:
            o = _origin(a, env, vocab)
            if o:
                return o
    return None


def audit_function(fn, vocab=None):
    """Returns (is_sweep, findings) for one function definition."""
    vocab = vocab or DEFAULT_VOCAB
    name = fn.name.lower()
    has_hint = any(h in name for h in vocab.hints)
    # A sweep trains inside a loop over a grid of values.
    trains_in_loop = any(
        isinstance(n, ast.Call) and getattr(n.func, "id", "") in vocab.train_fns
        for loop in ast.walk(fn) if isinstance(loop, (ast.For, ast.While))
        for n in ast.walk(loop))
    if not (has_hint and trains_in_loop):
        return False, []

    # Track local aliases: eval_loader = <expr>
    env = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            o = _origin(node.value, env, vocab)
            if o:
                env[node.targets[0].id] = o

    findings = []
    for loop in ast.walk(fn):
        if not isinstance(loop, (ast.For, ast.While)):
            continue
        for node in ast.walk(loop):
            if not (isinstance(node, ast.Call) and
                    getattr(node.func, "id", "") in vocab.eval_fns):
                continue
            # First positional arg after the model is the loader.
            loader_arg = node.args[1] if len(node.args) > 1 else None
            if loader_arg is None:
                continue
            o = _origin(loader_arg, env, vocab)
            findings.append({"line": node.lineno, "resolved": o or "unknown",
                             "call": getattr(node.func, "id", "?")})
    return True, findings


def audit_file(path, vocab=None):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        is_sweep, findings = audit_function(fn, vocab)
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


DEFAULT_VOCAB = Vocab()


def verdict_for_file(res):
    """File-level verdict from every sweep in the file.

    One contaminated sweep condemns the file: a script that selects on OOD data
    anywhere has contaminated the numbers it reports. Reading only res[0] would
    let a clean first sweep mask a contaminated second one.
    """
    verdicts = {r["verdict"] for r in res}
    if not verdicts:
        return "INDETERMINATE"
    if "CONTAMINATED" in verdicts:
        return "CONTAMINATED"
    return "OK" if "OK" in verdicts else "INDETERMINATE"


def run(paths=None, verbose=False):
    """paths: list of (path, expected_flag). Defaults to the two historical
    fixtures -- the real buggy sweep and its correction."""
    from detectors.harness import Case
    if paths is None:
        paths = [(os.path.join(FIX, "sweep_BUGGY.py"), True),
                 (os.path.join(FIX, "sweep_FIXED.py"), False)]
    out = []
    for path, exp in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        res = audit_file(path)
        verdict = verdict_for_file(res)
        flagged = (verdict == "CONTAMINATED")
        if verbose:
            print(f"\n{os.path.basename(path)}  -> {verdict}")
            for r in res:
                print(f"   {r['function']}() line {r['line']}: {r['verdict']}")
                for f in r["findings"]:
                    print(f"      eval at line {f['line']} reads "
                          f"{f['resolved'].upper()} data")
        out.append(Case(os.path.basename(path), flagged, bool(exp),
                        {"verdict": verdict,
                         "sweeps": [r["function"] for r in res],
                         "contaminated": [r["function"] for r in res
                                          if r["verdict"] == "CONTAMINATED"]}))
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
        got = verdict_for_file(res)
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
