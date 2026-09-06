# Does the undecidable-state failure generalize outside clinical ML?

The failure this project hit four times: a computation that could not produce an
answer returned `nan`, `None`, or an empty container, and the surrounding code
consumed that falsy/undefined value as though it meant "no problem found". Our
own case was `harness.rates()` scoring an empty confusion cell as a clean pass,
which manufactured a perfect `TP=3 FP=0 FN=0 TN=1`.

This asks whether the same mechanism appears in public benchmark code in a
NON-clinical domain, so the finding can be stated as a property of evaluation
code rather than of ICU pipelines.

## Corpus

Three repositories, cloned read-only and READ. None was executed. The only code
run for this note was four lines of plain-Python arithmetic reproducing `max()`
behaviour on `nan` (below) — no repository import, no repository entry point,
same parse-only discipline as the detector 3/4 external validation.

    repo        commit     domain                        why
    DomainBed   b93c22a1cf vision domain generalization  already pinned from the detector 3 scan
    robustdg    3eee1730ae vision domain generalization  already pinned; independent implementation
    AIF360      3487791    tabular fairness              has an explicit DETECTOR module, i.e. real verdicts

AIF360 was added specifically because the first two mostly PRINT numbers. The
mechanism needs something that makes a decision, and AIF360's `detectors/mdss`
does. It was sparse-checked out to the `aif360/` package because the full tree
exceeds Windows' path limit.

## Headline: the pattern needs a verdict, and most benchmark code has none

The clearest result is structural rather than a bug list. **A falsy value can
only be misread as "clean" where something asks a yes/no question.** DomainBed
and robustdg overwhelmingly compute a number, format it, and print it for a
human. A `nan` there is loud: it prints as `nan` in the results table. It is
ugly, it is not silent, and no automated conclusion rides on it.

Our detectors are different in kind — they emit verdicts — which is exactly why
we were exposed. That is a real limitation on how far the finding generalizes,
and it should be stated as such rather than smoothed over.

The interesting corollary: benchmark code DOES make one decision, model
selection, and that is precisely where the guarding turns out to be
inconsistent.

## Finding 1 (DomainBed) — the guard exists in one method and not its neighbour

`domainbed/lib/query.py` contains both of these on the same class:

    def argmax(self, selector):
        selector = make_selector_fn(selector)
        return max(self._list, key=selector)

    def sorted(self, key=None):
        ...
        def key2(x):
            x = key(x)
            if isinstance(x, (np.floating, float)) and np.isnan(x):
                return float('-inf')
            else:
                return x

`sorted` explicitly maps `nan` to `-inf`. A `filter_not_nan` helper exists as
well. So `nan` was anticipated in this file. `argmax` — used by three of the four
selection methods in `model_selection.py` to pick the winning model — has no such
guard.

`max(..., key=...)` keeps its incumbent unless a later element compares strictly
greater, and every comparison against `nan` is False. Verified in plain Python:

    nan first  -> the nan record is returned as the maximum
    nan second -> the nan record is skipped

So the selected model depends on **where in the list the undefined value
landed**, and nothing is printed either way. This is the same shape as our own
bug: not a wrong number, but a decision taken on a value that carried no
information.

Status: **mechanism confirmed, exploitation not reachable with shipped
configurations.** See finding 3 for why.

## Finding 2 (DomainBed) — an emptiness guard that is vacuously satisfied

`model_selection.LeaveOneOutSelectionMethod._step_acc`:

    val_accs = np.zeros(n_envs) - 1
    ... fill in the entries that were measured ...
    val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:])
    if any([v == -1 for v in val_accs]):
        return None
    val_acc = np.sum(val_accs) / (n_envs - 1)

The `-1` sentinel guard is real and correct for its intended case: any unmeasured
validation environment aborts the record. But with `n_envs == 1` the slicing
leaves `val_accs == []`, `any([])` is **False**, the guard passes, and

    np.sum([]) / (1 - 1)  ->  nan

is returned inside a live `{'val_acc': ..., 'test_acc': ...}` dict. The caller's
`.filter_not_none()` keeps it, because `nan` is not `None`, and it flows into the
unguarded `argmax` of finding 1.

This is the closest external analogue to our own failure: not a missing guard,
but a guard that answers "no problem" on input it was never asked about. Ours was
`rates()` returning clean on an empty cell; theirs is `any([])`.

## Finding 3 — reachability, stated honestly

Neither finding fires on DomainBed as shipped. Every dataset in `DATASETS`
declares at least three environments (`ColoredMNIST` has 3, `DomainNet` has 6,
the minimum across the file is 3), so `val_env_keys` is never empty and `n_envs`
is never 1. Reaching either site requires a user-supplied single-environment
dataset, which the framework permits but does not provide.

**Both findings are therefore structurally present and not currently
exploitable.** Report them that way. Claiming a live bug in DomainBed would be
the same overreach this project keeps catching in others.

For contrast, and worth one line in the paper: DomainBed's *aggregation* layer is
well guarded. `collect_results.format_mean` returns the literal string `"X"` for
an empty series, the per-algorithm average prints `"X"` if any cell is `None`,
and `misc.accuracy` on an empty loader raises `ZeroDivisionError` rather than
returning a number. Those are three explicit undecidable states in one file. The
gap is confined to model selection.

## Finding 4 (robustdg) — silent omission rather than silent nan

`evaluation/per_domain_acc.py` guards zero support correctly:

    if y_c.shape[0]:
        acc = ...

A domain that receives no examples is simply never inserted into
`acc_per_domain`, so it produces no row. The aggregation in `test.py` then does:

    keys = final_metric_score[0].keys()
    for key in keys:
        for item in final_metric_score:
            curr_metric_score.append( item[key] )

The reported key set is taken from **run 0 alone**. The consequence is
asymmetric, and the asymmetry is the finding:

* a domain missing from run 0 is **silently absent from the entire multi-seed
  report**, even if every other run measured it;
* the same domain missing from a LATER run raises `KeyError` — loud, immediate.

So identical degeneracies produce a crash or a quiet gap depending only on which
seed hit them first. The undefined state is never represented as undefined; it is
either an exception or an absence.

## Finding 5 (AIF360) — searched, no hit; the empty path is defensible

`detectors/mdss/ScoringFunctions/BerkJones.qmle` returns `0` for empty
`expectations`, bypassing the `q < alpha -> return alpha` floor every non-empty
path receives. That `0` reaches `compute_qs`, whose score is `-penalty`, which
fails `> 0`, giving `exist = 0`, `q_min = q_max = 0`; `MDSS.py:122`'s
`(q_min < threshold) & (q_max > threshold)` is then False and the subgroup is not
flagged.

An empty subgroup not being flagged as anomalous is correct, so this is recorded
as **searched and not a finding**, not as a near miss. It is written down because
the absence is informative: the one library in the corpus that emits real
verdicts handles its empty case deliberately.

## What this supports in the paper, and what it does not

Supported: the mechanism is not clinical and not ours. It appears in a
Facebook-authored vision benchmark, in two forms, one of which (`any([])`) is the
same vacuous-guard shape as our own. It appears again in an independent vision
codebase as silent omission. The inconsistency inside `query.py` — `sorted`
guarded, `argmax` not, in one file — is the strongest single piece of evidence,
because it shows the hazard was known and still missed one call site.

Not supported: any claim that these are live bugs producing wrong published
numbers. They are not reachable as shipped. Do not write "DomainBed's results may
be affected".

Also not supported: that the pattern is common in benchmark code generally. The
search suggests the opposite, for a reason worth stating — benchmarks report
numbers to humans, and only a component that decides something can mistake
"undefined" for "fine". That is an argument for why DIAGNOSTIC tooling
specifically needs an explicit undecidable state, which is the claim this paper
makes.
