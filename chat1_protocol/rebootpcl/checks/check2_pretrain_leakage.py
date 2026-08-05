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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stays", type=int, default=1200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    from config import TEST_MODE, D_MODEL, N_LAYERS
    if not TEST_MODE:
        sys.exit("Run with PCL_TEST_MODE=1 — this check is deliberately small.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print("CHECK 2 — non-zero-shot pretraining leakage")
    print("=" * 78)
    print(f"model d={D_MODEL} layers={N_LAYERS} | device={device} | "
          f"{args.stays} stays/site | {args.seeds} seeds | {args.epochs} epochs")

    src_ds, leak_ds, probe_ds = build(args.stays, seed=0)
    print(f"source={len(src_ds.samples)}  leak pool={len(leak_ds.samples)}  "
          f"probe={len(probe_ds.samples)} (probe never pretrained on)")

    results = {}
    for frac in LEVELS:
        losses = []
        for s in range(args.seeds):
            l, n_leak = run_one(src_ds, leak_ds, probe_ds, frac, 42 + s,
                                args.epochs, device)
            losses.append(l)
        results[frac] = losses
        print(f"  leakage {int(frac*100):>3}%  (n_leak={n_leak:>4})  "
              f"probe loss {np.mean(losses):.6f} +/- {np.std(losses, ddof=1) if len(losses)>1 else 0:.6f}"
              f"   {[round(x,5) for x in losses]}")

    # PAIRED analysis. Every leakage level is run with the SAME seeds, and the
    # seed effect is a large shared nuisance (one seed sits above another at every
    # level). Comparing group means against the 0% spread therefore buries a real
    # effect in seed variance; differencing within seed removes it. We use the
    # RELATIVE change so a seed's baseline loss level cancels too.
    base = np.array(results[0.0])
    print("")
    print(f"null (0% leakage): mean={base.mean():.6f} "
          f"sd={base.std(ddof=1) if len(base) > 1 else 0.0:.6f}"
          f"   (unpaired spread — large, hence the paired test below)")
    print("")
    print(f"{'leakage':>9}{'probe loss':>13}{'paired rel. delta':>20}"
          f"{'t':>8}{'agree':>8}{'detected':>10}")
    detected = {}
    for frac in LEVELS:
        cur = np.array(results[frac])
        rel = (cur - base) / np.maximum(base, 1e-12)      # per-seed relative change
        mu = float(cur.mean())
        rmu = float(rel.mean())
        rsd = float(rel.std(ddof=1)) if len(rel) > 1 else 0.0
        t = rmu / (rsd / np.sqrt(len(rel))) if rsd > 1e-12 else 0.0
        agree = bool(np.all(rel < 0)) and frac > 0
        # Leakage must LOWER probe loss, consistently across every seed.
        flag = bool(frac > 0 and agree and (t <= -2.0 or len(rel) < 3))
        detected[frac] = flag
        print(f"{int(frac*100):>8}%{mu:>13.6f}{rmu*100:>19.1f}%{t:>8.2f}"
              f"{str(agree):>8}{str(flag):>10}")

    floor = next((f for f in LEVELS if f > 0 and detected[f]), None)
    print("\n" + "-" * 78)
    print(f"false positive at 0% leakage: {detected[0.0]}   (must be False)")
    print(f"detection floor: {f'{int(floor*100)}% leakage' if floor else 'NOT DETECTED at any level tested'}")
    json.dump({str(k): v for k, v in results.items()},
              open(os.path.join(HERE, "check2_results.json"), "w"), indent=2)
    ok = (not detected[0.0]) and (floor is not None)
    print("verdict:", "DETECTOR VALIDATED" if ok else
          "INCONCLUSIVE — leak not separable from seed noise at this scale")


if __name__ == "__main__":
    main()
