"""Do standard checks catch what the purpose-built detectors catch?

Scope, fixed in advance: the two scenarios that already have injection-based
ground truth -- detector 2's pretraining leakage (0/5/20/100%) and detector 5's
PhysioNet A vs B case. Running baselines against detectors whose ground truth is
observational would compare a measurement to an assumption.

Detector 2 arm. For each leakage level the encoder is pretrained exactly as
detector 2 pretrains it, then frozen and used to embed stays. A logistic
regression is fitted on SOURCE embeddings and scored three ways: k-fold on the
source, a held-out source split, and the target site. The baselines see only
those AUROCs, which is all standard practice ever sees.

The directional hypothesis under test: leakage should IMPROVE cross-site AUROC,
so a degradation-triggered baseline stays silent precisely when the confound is
worst. That is a prediction. It is scored as TP/FP/FN/TN like everything else,
and the run reports whether it held.

    PCL_TEST_MODE=1 python rebootpcl/standard/run_baselines.py --seeds 3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rebootpcl.harness import Case, confusion, fmt_matrix
from rebootpcl.checks.check2_pretrain_leakage import LEVELS, build
from rebootpcl.standard.baseline_checks import BASELINES, auroc

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")


@torch.no_grad()
def embed(model, dataset, device, batch=64):
    """Mean-pooled encoder output per stay, plus the stay's binary label."""
    from torch.utils.data import DataLoader
    model.eval()
    X, y = [], []
    for b in DataLoader(dataset, batch_size=batch, shuffle=False):
        x, m = b["x"].to(device), b["mask"].to(device)
        h = model.encode(x, m)
        X.append(h.mean(dim=1).cpu().numpy())
        y.append(b["sepsis"].cpu().numpy())
    return np.concatenate(X), np.concatenate(y)


def fit_logistic(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(sc.transform(X), y)
    return sc, clf


def score(sc, clf, X):
    return clf.predict_proba(sc.transform(X))[:, 1]


def kfold_sd(X, y, seed=0, k=5):
    """Standard k-fold CV AUROC spread on the source site."""
    from sklearn.model_selection import StratifiedKFold
    if len(np.unique(y)) < 2:
        return float("nan")
    aurocs = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        sc, clf = fit_logistic(X[tr], y[tr], seed)
        aurocs.append(auroc(y[te], score(sc, clf, X[te])))
    a = np.array([v for v in aurocs if np.isfinite(v)])
    return float(a.std(ddof=1)) if len(a) > 1 else float("nan")


def metrics_for_level(frac, seed, stays, epochs, device):
    """Train under `frac` leakage, then produce the AUROCs standard practice sees."""
    from src.baselines import fresh_model, run_erm_pretraining
    from torch.utils.data import DataLoader, Subset, ConcatDataset
    from config import BATCH_SIZE

    src_ds, leak_ds, probe_ds = build(stays, seed=seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    n_leak = int(round(frac * len(leak_ds)))
    if n_leak > 0:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(leak_ds))[:n_leak]
        train = ConcatDataset([src_ds, Subset(leak_ds, idx.tolist())])
    else:
        train = src_ds

    model = fresh_model(seed=seed).to(device)
    ckpt = os.path.join(RESULTS, f"_bl_{int(frac * 100)}_{seed}.pt")
    os.makedirs(RESULTS, exist_ok=True)
    run_erm_pretraining(model,
                        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                                   drop_last=True),
                        DataLoader(probe_ds, batch_size=BATCH_SIZE),
                        n_epochs=epochs, device=device, save_path=ckpt)
    if os.path.exists(ckpt):
        os.remove(ckpt)

    Xs, ys = embed(model, src_ds, device)
    Xt, yt = embed(model, probe_ds, device)

    # held-out source split for the in-domain figure
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xs))
    cut = int(0.75 * len(perm))
    tr, te = perm[:cut], perm[cut:]
    sc, clf = fit_logistic(Xs[tr], ys[tr], seed)

    return {"kfold_sd": kfold_sd(Xs, ys, seed),
            "indomain_auroc": auroc(ys[te], score(sc, clf, Xs[te])),
            "target_auroc": auroc(yt, score(sc, clf, Xt)),
            "n_source": int(len(ys)), "n_target": int(len(yt)),
            "target_prevalence": float(np.mean(yt))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stays", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    from config import TEST_MODE
    if not TEST_MODE:
        sys.exit("Run with PCL_TEST_MODE=1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print("TASK 5 — standard checks vs purpose-built detectors")
    print("=" * 78)
    print("scenario: detector 2's pretraining leakage, injection-based ground "
          f"truth | {args.seeds} seeds\n", flush=True)

    per_level = {f: [] for f in LEVELS}
    for s in range(args.seeds):
        seed = 42 + s
        for frac in LEVELS:
            per_level[frac].append(
                metrics_for_level(frac, seed, args.stays, args.epochs, device))
        print(f"  seed {seed} done", flush=True)

    print(f"\n{'leakage':>9}{'kfold sd':>11}{'in-domain':>12}{'target':>10}"
          f"{'gap':>9}")
    summary = {}
    for frac in LEVELS:
        m = per_level[frac]
        ks = np.nanmean([x["kfold_sd"] for x in m])
        ind = np.nanmean([x["indomain_auroc"] for x in m])
        tgt = np.nanmean([x["target_auroc"] for x in m])
        summary[frac] = {"kfold_sd": float(ks), "indomain": float(ind),
                         "target": float(tgt), "gap": float(ind - tgt)}
        print(f"{int(frac * 100):>8}%{ks:>11.4f}{ind:>12.4f}{tgt:>10.4f}"
              f"{ind - tgt:>9.4f}")

    print("\n" + "-" * 78)
    print("BASELINE VERDICTS (expected: flag whenever leakage > 0)")
    report = {}
    for name, fn in sorted(BASELINES.items()):
        cases, undecidable, labels = [], 0, []
        for frac in LEVELS:
            res = [fn(x) for x in per_level[frac]]
            decided = [f for f, ok in res if ok]
            n_undec = len(res) - len(decided)
            undecidable += n_undec
            if not decided:
                labels.append(f"{int(frac * 100)}%:UNDECIDABLE")
                continue        # excluded from the matrix, never scored silent
            flagged = sum(decided) > len(decided) / 2.0
            cases.append(Case(f"leakage {int(frac * 100)}%", bool(flagged),
                              bool(frac > 0),
                              {"flag_rate": sum(decided) / len(decided),
                               "n_decided": len(decided),
                               "n_undecidable": n_undec}))
            labels.append(f"{int(frac * 100)}%:"
                          f"{'FLAG' if flagged else 'silent'}"
                          f"{f'({n_undec} undec)' if n_undec else ''}")
        counts = confusion(cases)
        report[name] = {"counts": counts, "undecidable_seed_levels": undecidable,
                        "per_level": {str(c.name): c.stats for c in cases}}
        print("  " + fmt_matrix(name, counts))
        print("      " + "  ".join(labels))
        if undecidable:
            print(f"      {undecidable} seed-level(s) undecidable (AUROC "
                  f"undefined); excluded rather than scored as silent")

    print("\n  " + fmt_matrix("detector 2 (for comparison)",
                              {"TP": 3, "FP": 0, "FN": 0, "TN": 1}))

    print("\n" + "=" * 78)
    print("DIRECTIONAL HYPOTHESIS — did cross-site AUROC IMPROVE with leakage?")
    print("=" * 78)
    base_t = summary[0.0]["target"]
    improved = []
    for frac in LEVELS:
        if frac == 0:
            continue
        d = summary[frac]["target"] - base_t
        improved.append(d > 0)
        print(f"  leakage {int(frac * 100):>3}%: target AUROC "
              f"{summary[frac]['target']:.4f} vs {base_t:.4f} at 0%  "
              f"(delta {d:+.4f}) {'IMPROVED' if d > 0 else 'did not improve'}")
    if all(improved):
        print("\nHypothesis HELD at every level: leakage made cross-site "
              "performance look BETTER, so a degradation-triggered check has "
              "nothing to fire on.")
    elif any(improved):
        print("\nHypothesis HELD ONLY IN PART. Report the levels where it did "
              "and did not, and do not generalise beyond them.")
    else:
        print("\nHypothesis DID NOT HOLD: cross-site AUROC did not improve "
              "under leakage at any level. The predicted mechanism is wrong "
              "and must not be stated in the paper. Whatever the baselines do "
              "or miss here, they do not miss it for the predicted reason.")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "baselines.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"summary": {str(k): v for k, v in summary.items()},
                   "baselines": report,
                   "raw": {str(k): v for k, v in per_level.items()}},
                  fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
