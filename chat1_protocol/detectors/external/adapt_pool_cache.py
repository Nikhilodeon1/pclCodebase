"""Reuse another project's per-site sample cache as detector 2's pool cache.

Another script on the same pod (`finetune_mortality.py --cache-only`) already
paid ~57 minutes to load and pickle full MIMIC-IV and eICU, one file per site.
Detector 2's --pool-cache expects a single pickle holding a (source, target)
tuple. The underlying objects are the same processed sample lists, so this
converts rather than reloads.

It VERIFIES before converting. A silently incompatible cache would produce a
detector 2 result computed on the wrong objects, which is exactly the class of
error this project exists to catch, so a mismatch refuses loudly instead of
falling through to something plausible-looking.

    python detectors/external/adapt_pool_cache.py \
        --mimic <dir>/mimic_frac1.0.pkl \
        --eicu  <dir>/eicu_frac1.0.pkl \
        --out   detectors/results/full_d2_pools.pkl
"""
import argparse
import os
import pickle
import sys

# Keys ICUDataset reads off each sample. Missing any of these means the cache
# was written by a pipeline that does not produce what detector 2 consumes.
REQUIRED = ["values", "mask", "abg_mask", "c_mask", "label", "site_id"]


def load_site(path, label):
    if not os.path.exists(path):
        sys.exit(f"missing {label} cache: {path}")
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    # tolerate (samples, normalizer) tuples as well as a bare list
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], list):
        obj = obj[0]
    if not isinstance(obj, list) or not obj:
        sys.exit(f"{label} cache is not a non-empty list of samples: {type(obj)}")
    missing = [k for k in REQUIRED if k not in obj[0]]
    if missing:
        sys.exit(f"{label} sample is missing {missing}. This cache was not "
                 f"written by the loader detector 2 consumes; do NOT use it. "
                 f"Keys present: {sorted(obj[0])}")
    import numpy as np
    shape = np.asarray(obj[0]["values"]).shape
    print(f"{label:<6} {len(obj):>7} samples   values shape {shape}   OK")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic", required=True)
    ap.add_argument("--eicu", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("verifying source caches ...")
    src = load_site(args.mimic, "MIMIC")
    tgt = load_site(args.eicu, "eICU")

    if len(src) >= len(tgt):
        print(f"\nWARNING: MIMIC ({len(src)}) is not smaller than eICU "
              f"({len(tgt)}). Detector 2 uses MIMIC as SOURCE and eICU as "
              f"TARGET and caps the leak pool at the source size; check the "
              f"two paths are not swapped before trusting the result.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump((src, tgt), fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nwrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB)")
    print("detector 2 will now skip the full load. Run it with "
          f"--pool-cache {args.out}")


if __name__ == "__main__":
    main()
