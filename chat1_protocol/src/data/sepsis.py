"""
ICD-code-based sepsis labeling for MIMIC-IV and eICU.

Matches the paper's §4.1 claim that sepsis labels for these datasets are
extracted from ICD diagnosis codes (explicit-sepsis definition), rather than
the hourly Sepsis-3 labels shipped with PhysioNet 2019. This gives a real,
non-zero, admission-level binary sepsis label across all three sites.

Explicit-sepsis code set (ICD-9 + ICD-10):
  ICD-9 : 038.x  (septicemia), 995.91 (sepsis), 995.92 (severe sepsis),
          785.52 (septic shock)
  ICD-10: A40.x  (streptococcal sepsis), A41.x (other sepsis),
          R65.20 (severe sepsis), R65.21 (septic shock)
"""
import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

_SEPSIS_ICD9_EXACT = {"99591", "99592", "78552"}


def _norm_icd(code):
    """Normalize an ICD code: uppercase, strip dots/spaces (99591, A419, R6520)."""
    return str(code).upper().replace(".", "").replace(" ", "").strip()


def _is_sepsis_code(n):
    if n in _SEPSIS_ICD9_EXACT:
        return True
    if n.startswith("038"):                              # ICD-9 septicemia
        return True
    if n.startswith("A40") or n.startswith("A41"):       # ICD-10 sepsis
        return True
    if n.startswith("R6520") or n.startswith("R6521"):   # severe sepsis / septic shock
        return True
    return False


def _resolve(path):
    if os.path.exists(path):
        return path
    alt = path.replace(".csv.gz", ".csv")
    return alt if os.path.exists(alt) else None


def mimic_sepsis_hadm_ids(data_path):
    """Set of MIMIC hadm_ids carrying an explicit-sepsis ICD code."""
    path = _resolve(os.path.join(data_path, "hosp", "diagnoses_icd.csv.gz"))
    if path is None:
        logging.warning(
            "MIMIC hosp/diagnoses_icd.csv.gz NOT FOUND — sepsis labels will all be 0. "
            "Download it to enable ICD-based sepsis labeling."
        )
        return set()
    d = pd.read_csv(path, usecols=["hadm_id", "icd_code"], dtype={"icd_code": str},
                    encoding_errors="replace")
    d = d[d["icd_code"].notna()]
    mask = d["icd_code"].map(lambda c: _is_sepsis_code(_norm_icd(c)))
    ids = set(d.loc[mask, "hadm_id"].astype(int))
    logging.info(f"MIMIC sepsis labeling: {len(ids)} admissions flagged sepsis+ (ICD)")
    return ids


def eicu_sepsis_stay_ids(data_path):
    """Set of eICU patientunitstayid carrying an explicit-sepsis ICD code.

    eICU's `icd9code` column holds comma-separated, dotted codes mixing ICD-9
    and ICD-10 (e.g. "995.91, A41.9"), so each is split and normalized.
    """
    path = _resolve(os.path.join(data_path, "diagnosis.csv.gz"))
    if path is None:
        logging.warning(
            "eICU diagnosis.csv.gz NOT FOUND — sepsis labels will all be 0. "
            "Download it to enable ICD-based sepsis labeling."
        )
        return set()
    d = pd.read_csv(path, usecols=["patientunitstayid", "icd9code"],
                    encoding_errors="replace")
    d = d[d["icd9code"].notna()]

    def row_sepsis(code):
        return any(_is_sepsis_code(_norm_icd(part)) for part in str(code).split(","))

    mask = d["icd9code"].map(row_sepsis)
    ids = set(d.loc[mask, "patientunitstayid"].astype(int))
    logging.info(f"eICU sepsis labeling: {len(ids)} stays flagged sepsis+ (ICD)")
    return ids
