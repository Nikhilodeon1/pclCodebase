"""
Download and unzip the three ICU datasets from Google Drive.
Skips any dataset whose target directory already exists and is non-empty.
"""
import os
import zipfile
import logging

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_DATASETS = [
    {
        "name": "PhysioNet 2019",
        "file_id": "1GH88GziKSfesFHFS_yUIK_4RU5g7ZkKu",
        "zip_name": "physionet2019.zip",
        "target_dir": os.path.join(_BASE, "physionet2019"),
    },
    {
        "name": "MIMIC-IV Demo",
        "file_id": "1nuD0KHtgRh1Y2FTt0eCly5eNdgy_oibn",
        "zip_name": "mimic4-demo.zip",
        "target_dir": os.path.join(_BASE, "mimic4-demo"),
    },
    {
        "name": "eICU Demo",
        "file_id": "1glNLUd5Oashi4t-HRIZ2FEBe-JFYpOye",
        "zip_name": "eICU-demo.zip",
        "target_dir": os.path.join(_BASE, "eICU-demo"),
    },
]


def _dir_has_files(path):
    if not os.path.isdir(path):
        return False
    for _, _, files in os.walk(path):
        if files:
            return True
    return False


def ensure_datasets():
    """
    Download and unzip demo datasets into data/ if they are missing.

    On RunPod (or any environment where PHYSIONET_DIR / MIMIC_DIR / EICU_DIR
    are set via env vars pointing to the full datasets), skip downloading
    entirely — the real data is already present at those paths.
    """
    # If all three env-var paths exist and are non-empty, nothing to do.
    env_physionet = os.environ.get("PHYSIONET_DIR", "")
    env_mimic     = os.environ.get("MIMIC_DIR", "")
    env_eicu      = os.environ.get("EICU_DIR", "")

    if env_physionet and env_mimic and env_eicu:
        all_present = (
            _dir_has_files(env_physionet)
            and _dir_has_files(env_mimic)
            and _dir_has_files(env_eicu)
        )
        if all_present:
            logging.info("All datasets found at configured env-var paths — skipping demo download.")
            return
        missing = [p for p in [env_physionet, env_mimic, env_eicu] if not _dir_has_files(p)]
        logging.warning(f"Env-var paths set but these are empty/missing: {missing}")
        logging.warning("Falling through to demo download for any missing datasets.")

    try:
        import gdown
    except ImportError:
        raise ImportError("gdown not installed. Run: pip install gdown")

    os.makedirs(_BASE, exist_ok=True)

    for ds in _DATASETS:
        if _dir_has_files(ds["target_dir"]):
            logging.info(f"Dataset present, skipping download: {ds['name']}")
            continue

        zip_path = os.path.join(_BASE, ds["zip_name"])
        logging.info(f"Downloading {ds['name']} ...")
        url = f"https://drive.google.com/uc?id={ds['file_id']}"
        gdown.download(url, zip_path, quiet=False)

        if not os.path.exists(zip_path):
            logging.error(f"Download failed for {ds['name']} — zip not found at {zip_path}")
            continue

        logging.info(f"Extracting {ds['zip_name']} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(_BASE)
        os.remove(zip_path)
        logging.info(f"Done: {ds['name']} → {ds['target_dir']}")
