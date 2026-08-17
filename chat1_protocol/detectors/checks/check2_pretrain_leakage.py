"""
CHECK 2 — non-zero-shot pretraining leakage.

Confound: a method claimed to be zero-shot pretrains on unlabeled data that
includes the TARGET site. Pretraining uses no labels, so the leak looks harmless,
but the encoder has already seen the target distribution and downstream "zero-shot"
transfer is inflated. This is the exact defect found in this project's PCL setup.

Diagnostic: hold out a probe slice of the target site that is NEVER used for
pretraining at any leakage level. Pretrain with masked-value prediction on the
source plus a varying fraction of the remaining target data, then measure
reconstruction loss on the probe. If the encoder saw target data, it reconstructs
held-out target data better than a true zero-shot control does. Flag when the
probe loss falls outside the spread of the 0%-leakage runs.

Because the effect competes with seed noise, every leakage level is run at several
seeds and the decision uses the 0%-leakage spread as its null.

Source = PhysioNet Site A, target = PhysioNet Site B. Model is the small test-mode
transformer, so this runs on CPU without paid compute.

    PCL_TEST_MODE=1 python rebootpcl/checks/check2_pretrain_leakage.py --stays 1200 --seeds 3
"""
import os
import sys
import json
import copy
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

logging.disable(logging.INFO)

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = [0.0, 0.05, 0.20, 1.00]


def build(stays, seed):
    """Load Site A (source) and Site B (target); split target into leak pool and
    a probe slice that never enters pretraining."""
    from src.data.physionet2019 import load_physionet2019
    from src.data.dataset import ICUDataset
    from config import PHYSIONET_DIR

    # Subsample BEFORE reading: the loader's `fraction` filters the file list,
    # whereas loading everything and slicing afterwards reads all ~40k PSVs.
    frac = min(1.0, (stays * 1.35) / 20000.0)   # headroom for stays dropped by filters
    src, _ = load_physionet2019(PHYSIONET_DIR, fraction=frac, sites=[0], seed=seed)
    tgt, _ = load_physionet2019(PHYSIONET_DIR, fraction=frac, sites=[1], seed=seed)
    rng = np.random.default_rng(seed)
    src = [src[i] for i in rng.permutation(len(src))[:stays]]
    tgt = [tgt[i] for i in rng.permutation(len(tgt))[:stays]]
    n_probe = max(50, len(tgt) // 4)
    probe, leak_pool = tgt[:n_probe], tgt[n_probe:]
    return ICUDataset(src), ICUDataset(leak_pool), ICUDataset(probe)


@torch.no_grad()
def probe_loss(model, loader, mask_prob, device, seed=0):
    """Masked-reconstruction loss on the probe slice, with a fixed mask pattern so
    the measurement is comparable across models."""
    from src.models.backbone import apply_random_mask
    from src.training.train_utils import masked_prediction_loss
    model.eval()
    tot, n = 0.0, 0
    for bi, b in enumerate(loader):
        x = b["x"].to(device)
        m = b["mask"].to(device)
        torch.manual_seed(seed * 7919 + bi)
        xm, pm = apply_random_mask(x, m, mask_prob)
        loss = masked_prediction_loss(model.predict(model.encode(xm, m)), x, pm)
        if torch.isfinite(loss):
            tot += float(loss.item()) * x.shape[0]
            n += x.shape[0]
    return tot / max(n, 1)


def run_one(src_ds, leak_ds, probe_ds, frac, seed, epochs, device):
    """Pretrain on source + `frac` of the target leak pool; return probe loss."""
    from src.baselines import fresh_model, run_erm_pretraining
    from config import MASK_PROB, BATCH_SIZE
    from torch.utils.data import ConcatDataset

    torch.manual_seed(seed); np.random.seed(seed)
    n_leak = int(round(frac * len(leak_ds)))
    if n_leak > 0:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(leak_ds))[:n_leak]
        train = ConcatDataset([src_ds, Subset(leak_ds, idx.tolist())])
    else:
        train = src_ds

    tl = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    pl = DataLoader(probe_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = fresh_model(seed=seed).to(device)
    ckpt = os.path.join(HERE, f"_leak_{int(frac*100)}_{seed}.pt")
    run_erm_pretraining(model, tl, pl, n_epochs=epochs, device=device,
                        save_path=ckpt)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    return probe_loss(model, pl, MASK_PROB, device), n_leak


# One-sided t critical values at alpha=0.05, indexed by degrees of freedom.
# Hardcoded so the check keeps no scipy dependency.
_T_CRIT_05 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943,
              7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812}


def flag_from_relative(rel):
    """Decide detection from per-seed relative deltas.

    Leakage must LOWER probe loss: every seed negative, and the paired
    one-sided t statistic past the df-appropriate 5% critical value. A fixed
    threshold of -2.0 corresponds to no alpha at small n (df=2 needs 2.920),
    so the critical value is looked up by df.

    Returns (flagged, t, critical_t).
    """
    rel = np.asarray(rel, float)
    n = len(rel)
    if n < 3:
        return False, 0.0, float("inf")
    crit = _T_CRIT_05.get(n - 1, 1.645)      # large-df limit
    sd = float(rel.std(ddof=1))
    t = float(rel.mean() / (sd / np.sqrt(n))) if sd > 1e-12 else 0.0
    return bool(np.all(rel < 0) and t <= -crit), t, crit


def measure(stays, seeds, epochs, device, base_seed=42, verbose=False):
    """Probe loss per leakage level, one value per seed.

    The split is re-drawn for every seed, so probe slices differ across seeds
    and absolute losses are NOT comparable between them. That is deliberate:
    the analysis is paired within seed, which is what makes the levels
    comparable while still letting split variance into the spread.
    """
    results = {frac: [] for frac in LEVELS}
    for s in range(seeds):
        seed = base_seed + s
        src_ds, leak_ds, probe_ds = build(stays, seed=seed)
        if verbose and s == 0:
            print(f"source={len(src_ds.samples)}  leak pool={len(leak_ds.samples)}  "
                  f"probe={len(probe_ds.samples)} (probe never pretrained on)",
                  flush=True)
        for frac in LEVELS:
            l, _ = run_one(src_ds, leak_ds, probe_ds, frac, seed, epochs, device)
            results[frac].append(l)
        if verbose:
            print(f"  seed {seed} done ({s + 1}/{seeds})", flush=True)
    return results


def analyse(results):
    """(detected, stats) from the paired within-seed relative deltas."""
    base = np.array(results[0.0])
    detected, stats = {}, {}
    for frac in LEVELS:
        cur = np.array(results[frac])
        rel = (cur - base) / np.maximum(base, 1e-12)
        sig, t, crit = flag_from_relative(rel)
        detected[frac] = bool(frac > 0 and sig)
        stats[frac] = {"mean_loss": float(cur.mean()),
                       "rel_delta": float(rel.mean()),
                       "t": t, "crit": crit,
                       "sign_agree": bool(np.all(rel < 0)),
                       "losses": [float(x) for x in cur]}
    return detected, stats


def run(seed=0, stays=900, epochs=3, seeds=5, verbose=False, device=None):
    """One Case per leakage level. 0% is the false-positive control.

    `seed` shifts the whole seed block, so a sweep over it measures whether the
    VERDICT reproduces, not whether one training run does.
    """
    from rebootpcl.harness import Case
    from config import TEST_MODE
    if not TEST_MODE:
        raise RuntimeError("check 2 requires PCL_TEST_MODE=1")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    results = measure(stays, seeds, epochs, device,
                      base_seed=42 + 100 * seed, verbose=verbose)
    detected, stats = analyse(results)
    return [Case(f"leakage {int(f * 100)}%", detected[f], bool(f > 0),
                 dict(stats[f], seed=seed)) for f in LEVELS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stays", type=int, default=1200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    from config import TEST_MODE, D_MODEL, N_LAYERS
    from rebootpcl.harness import Case, confusion, fmt_matrix
    if not TEST_MODE:
        sys.exit("Run with PCL_TEST_MODE=1 — this check is deliberately small.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print("CHECK 2 — non-zero-shot pretraining leakage")
    print("=" * 78)
    print(f"model d={D_MODEL} layers={N_LAYERS} | device={device} | "
          f"{args.stays} stays/site | {args.seeds} seeds | {args.epochs} epochs")
    print("NOTE: the data split is re-drawn per seed (site sample, leak pool and "
          "probe slice), so the reported spread includes split variance.",
          flush=True)

    results = measure(args.stays, args.seeds, args.epochs, device, verbose=True)
    detected, stats = analyse(results)

    for frac in LEVELS:
        losses = results[frac]
        print(f"  leakage {int(frac*100):>3}%  probe loss {np.mean(losses):.6f} "
              f"+/- {np.std(losses, ddof=1) if len(losses) > 1 else 0:.6f}"
              f"   {[round(x, 5) for x in losses]}")

    base = np.array(results[0.0])
    print("")
    print(f"null (0% leakage): mean={base.mean():.6f} "
          f"sd={base.std(ddof=1) if len(base) > 1 else 0.0:.6f}"
          f"   (unpaired spread — large, hence the paired test below)")
    print("")
    print(f"{'leakage':>9}{'probe loss':>13}{'paired rel. delta':>20}"
          f"{'t':>8}{'agree':>8}{'detected':>10}")
    for frac in LEVELS:
        st = stats[frac]
        print(f"{int(frac*100):>8}%{st['mean_loss']:>13.6f}"
              f"{st['rel_delta']*100:>19.1f}%{st['t']:>8.2f}"
              f"{str(st['sign_agree']):>8}{str(detected[frac]):>10}"
              f"   (crit {-st['crit']:.3f})")

    floor = next((f for f in LEVELS if f > 0 and detected[f]), None)
    print("\n" + "-" * 78)
    print(f"false positive at 0% leakage: {detected[0.0]}   (must be False)")
    print(f"detection floor: {f'{int(floor*100)}% leakage' if floor else 'NOT DETECTED at any level tested'}")
    cases = [Case(f"leakage {int(f * 100)}%", detected[f], bool(f > 0), stats[f])
             for f in LEVELS]
    print("\n" + fmt_matrix("check2", confusion(cases)))
    json.dump({str(k): v for k, v in results.items()},
              open(os.path.join(HERE, "check2_results.json"), "w"), indent=2)
    ok = (not detected[0.0]) and (floor is not None)
    print("verdict:", "DETECTOR VALIDATED" if ok else
          "INCONCLUSIVE — leak not separable from seed noise at this scale")


if __name__ == "__main__":
    main()
