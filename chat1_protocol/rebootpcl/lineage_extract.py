"""Extract detector 4's data lineage from loader source instead of asserting it.

Detector 4 currently reads a HAND-WRITTEN lineage table. Eight of its nine cells
are asserted rather than measured, and only one is cross-checked against code
(a grep for the Severinghaus constants). So what it demonstrates is that a
correct table yields a correct verdict -- true, but close to tautological, and
the real reason its evidence reads as n=1.

This module derives the table from the loader's AST, so the table can be wrong
and the detector can catch it.

GUARD-CONDITION REASONING IS BUILT IN FROM THE START, not added after a false
positive. It has to be: physionet2019.py contains

    if "HCO3" not in df.columns and "BaseExcess" in df.columns:
        ts[valid, hco3_idx] = 24.0 + 0.5 * be[valid]

which derives HCO3 -- an input to Henderson-Hasselbalch -- from base excess. A
naive extractor reports HCO3 as derived and detector 4 gains a false positive.
Every PhysioNet PSV carries an HCO3 column, so that branch never executes and
the hand-written table's "measured" is correct. An extractor that cannot reason
about the guard would be WRONG on the one dataset whose answer we already know,
so guards are part of the design rather than hardening bolted on later.

Extraction therefore reports a derivation together with the predicates guarding
it; `resolve()` then decides reachability against the columns a dataset actually
has. Separating the two keeps the static claim ("this code can derive X from Y")
apart from the data-dependent one ("and it does so here").
"""
import ast
import os

# Equation signatures: constants and calls that identify a formula in source.
# Matching on numeric constants is deliberate -- variable names differ between
# codebases but 23400 and 0.0307 are the equations themselves.
# `constants` must all be present. `names_any`/`names_all` add a required
# identifier context for signatures whose constants are too generic to stand
# alone -- 24.0 and 0.5 or a bare 3.0 would otherwise match ordinary
# arithmetic all over a codebase, and a false positive here is precisely the
# failure detector 4 must not have. `calls` is CONFIRMATION ONLY, never
# required: real derivations are split across statements (the Severinghaus
# constant and its cube root sit on different lines in physionet2019.py), so
# demanding both in one expression finds nothing.
EQUATION_SIGNATURES = {
    "severinghaus": {"constants": {23400.0}, "calls": {"cbrt"},
                     "names_any": set(), "names_all": set()},
    "henderson_hasselbalch": {"constants": {6.1, 0.0307}, "calls": {"log10"},
                              "names_any": set(), "names_all": set()},
    "base_excess_to_hco3": {"constants": {24.0, 0.5}, "calls": set(),
                            "names_any": {"be", "BaseExcess", "base_excess"},
                            "names_all": set()},
    "map_identity": {"constants": {3.0}, "calls": set(),
                     "names_any": set(), "names_all": {"SBP", "DBP"}},
}


def _constants(node):
    return {float(n.value) for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)}


def _calls(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def _names(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.add(n.value)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def match_equation(node):
    """Which known equation, if any, this expression implements."""
    consts, calls, names = _constants(node), _calls(node), _names(node)
    best, best_score = None, -1
    for eq, sig in EQUATION_SIGNATURES.items():
        if not sig["constants"] <= consts:
            continue
        if sig["names_any"] and not sig["names_any"] & names:
            continue
        if not sig["names_all"] <= names:
            continue
        # Specificity: number of required constants, plus a point when the
        # confirming call is also present in the same expression.
        score = len(sig["constants"]) + bool(sig["calls"] & calls)
        if score > best_score:
            best, best_score = eq, score
    return best


def classify_guard(test):
    """Classify one `if` predicate.

    Returns (kind, column) where kind is:
      "column_absent"  -- `"X" not in df.columns`, so the body runs only when
                          the dataset LACKS column X
      "column_present" -- `"X" in df.columns`
      "unknown"        -- anything else; treated as possibly-reachable, because
                          assuming unreachable would silently drop real
                          derivations
    """
    out = []
    for node in ast.walk(test):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        op = node.ops[0]
        if not isinstance(op, (ast.In, ast.NotIn)):
            continue
        left = node.left
        if not (isinstance(left, ast.Constant) and isinstance(left.value, str)):
            continue
        target = node.comparators[0]
        # only column-membership tests count; `x in some_list` does not
        if not (isinstance(target, ast.Attribute) and target.attr == "columns"):
            continue
        out.append(("column_absent" if isinstance(op, ast.NotIn)
                    else "column_present", left.value))
    return out or [("unknown", None)]


def _var_for_index(node, index_names):
    """Map an assignment target back to a canonical variable.

    Handles the two shapes loaders actually use: a subscript by an index
    variable (`ts[valid, hco3_idx]`, with `hco3_idx = VAR_TO_IDX["HCO3"]`
    earlier) and a direct string subscript (`df["HCO3"]`).
    """
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in index_names:
            return index_names[n.id]
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value
    return None


def extract_lineage(path, variables):
    """{variable: record} for `variables` found assigned in `path`.

    A record is {"kind": "derived", "equation":..., "sources":[...],
    "guards":[...], "line": n}. Variables never assigned from an equation are
    absent, which the caller reads as "measured".
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    # index-variable -> canonical name, e.g. hco3_idx = VAR_TO_IDX["HCO3"]
    index_names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            for s in ast.walk(node.value):
                if isinstance(s, ast.Subscript) and isinstance(
                        s.value, ast.Name) and "IDX" in s.value.id.upper():
                    key = s.slice
                    if isinstance(key, ast.Constant) and \
                            isinstance(key.value, str):
                        index_names[node.targets[0].id] = key.value

    # Walk statements IN ORDER, carrying the stack of enclosing `if` tests and a
    # taint map of intermediate locals. Real loaders spread a derivation over
    # several statements -- physionet2019.py computes the Severinghaus inversion
    # as `inner = 23400.0 * ...`, then `coeff = np.cbrt(inner)`, then writes
    # `ts[valid, pao2_idx] = np.clip(coeff, ...)`. The constants never appear in
    # the statement that assigns the variable, so matching single expressions in
    # isolation finds nothing. Tainting propagates the equation through the
    # intermediates to the write that matters.
    found = {}

    def visit(body, guards, taint):
        for stmt in body:
            if isinstance(stmt, ast.If):
                g = guards + classify_guard(stmt.test)
                visit(stmt.body, g, dict(taint))
                visit(stmt.orelse, guards, dict(taint))
                continue
            if isinstance(stmt, (ast.For, ast.While, ast.With)):
                visit(stmt.body, guards, taint)
                continue
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                visit(stmt.body, guards, {})     # new scope, fresh taint
                continue
            if not isinstance(stmt, ast.Assign):
                continue

            eq = match_equation(stmt.value)
            if eq is None:
                # inherit an equation from any tainted local it reads
                for n in ast.walk(stmt.value):
                    if isinstance(n, ast.Name) and n.id in taint:
                        eq = taint[n.id]
                        break
            if eq is None:
                continue

            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    taint[tgt.id] = eq          # intermediate, keep propagating
                var = _var_for_index(tgt, index_names)
                if var in variables:
                    found[var] = {
                        "kind": "derived", "equation": eq,
                        "sources": sorted((_names(stmt.value) | set(taint))
                                          & set(variables)),
                        "guards": guards, "line": stmt.lineno}

    visit(tree.body, [], {})
    return found


def resolve(lineage, available_columns):
    """Decide which derivations actually execute for a dataset.

    A derivation guarded by `"X" not in df.columns` is UNREACHABLE when X is
    present, and vice versa. Unknown guards are treated as reachable: assuming
    otherwise would silently discard real derivations, and a false negative here
    is exactly the failure mode detector 4 exists to prevent.
    """
    out = {}
    cols = set(available_columns)
    for var, rec in lineage.items():
        reachable, why = True, []
        for kind, col in rec["guards"]:
            if kind == "column_absent" and col in cols:
                reachable = False
                why.append(f'guarded by "{col}" not in columns, but "{col}" '
                           f'IS present')
            elif kind == "column_present" and col is not None and col not in cols:
                reachable = False
                why.append(f'guarded by "{col}" in columns, but "{col}" is absent')
        out[var] = dict(rec, reachable=reachable, reason="; ".join(why))
    return out


def to_detector4_lineage(resolved, measured_vars):
    """Convert to the shape check 4's LINEAGE table uses."""
    table = {v: "measured" for v in measured_vars}
    for var, rec in resolved.items():
        if rec["reachable"]:
            src = rec["sources"][0] if rec["sources"] else "unknown"
            table[var] = ("derived", rec["equation"], src)
    return table
