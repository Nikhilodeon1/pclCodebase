"""
SOFA-based Sepsis-3 re-derivation for MIMIC-IV and eICU (professor action A1).

Replaces the ICD-code sepsis label (src/data/sepsis.py) with a clinical
Sepsis-3 label matching the reference implementation, to remove the
label-definition confound between train (clinical Sepsis-3 on PhysioNet) and
external eval (administrative ICD codes on MIMIC/eICU).

Methodology follows MIT-LCP/mimic-iv `concepts/sepsis/sepsis3.sql` (+ `sofa.sql`,
`suspicion_of_infection.sql`), the standard reference in this space:

  Sepsis-3 = suspected infection  AND  SOFA >= 2 in the infection window.

  * Baseline SOFA is assumed 0 for every patient (mimic-code does the same:
    it does not know pre-ICU organ dysfunction), so "acute rise >= 2" reduces to
    "SOFA >= 2" -- but ONLY when SOFA is evaluated at the right time.
  * SOFA is computed HOURLY with a 24-HOUR ROLLING LOOKBACK per organ system
    (worst value in the preceding 24 h), for all six systems (respiration,
    coagulation, liver, cardiovascular, CNS, renal).
  * The hourly SOFA series is then restricted to the suspected-infection window
    [t_suspicion - 48 h, t_suspicion + 24 h]; sepsis = max SOFA in that window
    >= 2. This coupling (not whole-stay "SOFA ever >= 2") is what keeps the rate
    realistic even with baseline 0 -- almost everyone hits SOFA>=2 eventually,
    but few do so exactly during their infection workup.
  * Suspected infection uses the ASYMMETRIC Sepsis-3 window: antibiotic first ->
    culture within +24 h; culture first -> antibiotic within +72 h. t_suspicion
    is the earlier event of the qualifying pair.
  * Label granularity is PER-STAY BINARY (positive if criteria met), matching the
    existing pipeline's "positive if ever met" harmonization.
  * Missing SOFA components score 0 (standard SOFA-on-EHR practice). We report the
    observed-vs-imputed fraction per component.

KNOWN SIMPLIFICATION: cardiovascular uses MAP + vasopressor *presence*; full
dose-based 3/4 stratification needs weight-normalized rates and is approximated
as "any vasopressor infusion => score 3" (flagged via cardio_dose_approx=True).
"""
import os
import logging

import numpy as np
import pandas as pd

from .preprocessing import make_heartbeat

logging.basicConfig(level=logging.INFO)

SOFA_WINDOW_PRE_H = 48.0     # t_suspicion - 48h
SOFA_WINDOW_POST_H = 24.0    # t_suspicion + 24h
SOFA_LOOKBACK_H = 24.0       # rolling worst-value lookback per organ

# ── MIMIC-IV item IDs ────────────────────────────────────────────────────────
MIMIC_LAB_IDS = {"creatinine": [50912], "bilirubin": [50885], "platelets": [51265]}
MIMIC_CHART_IDS = {
    "fio2":       [223835],
    "gcs_eye":    [220739], "gcs_verbal": [223900], "gcs_motor": [223901],
    "pao2":       [220224],
}
MIMIC_VASOPRESSOR_IDS = [221906, 221289, 221662, 221653, 222315, 221749]
MIMIC_URINE_IDS = [226559, 226560, 226561, 226584, 226563, 226564, 226565,
                   226567, 226557, 226558, 227488, 227489]

# eICU string keys (labname / drugname fragments, lower-cased)
EICU_LABNAMES = {"creatinine": "creatinine", "bilirubin": "total bilirubin",
                 "platelets": "platelets x 1000", "pao2": "pao2", "fio2": "fio2"}
VASOPRESSOR_PATTERNS = ["norepinephrine", "levophed", "epinephrine", "dopamine",
                        "dobutamine", "vasopressin", "phenylephrine", "neosynephrine"]

# eICU-specific: culture data (microLab) is sparsely charted, so suspected
# infection is corroborated by a culture draw OR an infection diagnosis
# (strategy-reviewed adaptation of option (b); disclose in the label-shift
# limitation). Broad infection-adjacent list; narrow to explicit sepsis codes
# only if the eICU positive rate comes out too high.
INFECTION_DX_PATTERNS = ["sepsis", "septic", "septicemia", "bacteremia",
                         "pneumonia", "urinary tract", "uti", "cellulitis",
                         "meningitis", "endocarditis", "abscess", "infection",
                         "peritonitis", "cholangitis", "empyema", "osteomyelitis"]


def is_vasopressor(name):
    if not isinstance(name, str):
        return False
    n = name.lower()
    return any(p in n for p in VASOPRESSOR_PATTERNS)

ANTIBIOTIC_PATTERNS = [
    "vancomycin", "piperacillin", "tazobactam", "cefepime", "ceftriaxone",
    "ceftazidime", "cefazolin", "meropenem", "imipenem", "ertapenem",
    "ciprofloxacin", "levofloxacin", "moxifloxacin", "azithromycin",
    "clindamycin", "metronidazole", "gentamicin", "tobramycin", "amikacin",
    "ampicillin", "amoxicillin", "penicillin", "aztreonam", "linezolid",
    "daptomycin", "doxycycline", "tigecycline", "colistin", "nafcillin",
    "oxacillin", "cefuroxime", "cefotaxime", "trimethoprim", "sulfamethoxazole",
]


def is_antibiotic(name):
    if not isinstance(name, str):
        return False
    n = name.lower()
    return any(p in n for p in ANTIBIOTIC_PATTERNS)


# ── SOFA component scoring (0-4 each) ────────────────────────────────────────
def sofa_respiration(pf):
    if pf is None or np.isnan(pf): return 0
    if pf < 100: return 4
    if pf < 200: return 3
    if pf < 300: return 2
    if pf < 400: return 1
    return 0

def sofa_coagulation(plt):
    if plt is None or np.isnan(plt): return 0
    if plt < 20: return 4
    if plt < 50: return 3
    if plt < 100: return 2
    if plt < 150: return 1
    return 0

def sofa_liver(bili):
    if bili is None or np.isnan(bili): return 0
    if bili >= 12.0: return 4
    if bili >= 6.0: return 3
    if bili >= 2.0: return 2
    if bili >= 1.2: return 1
    return 0

def sofa_cardiovascular(map_min, any_vaso):
    if any_vaso: return 3
    if map_min is not None and not np.isnan(map_min) and map_min < 70: return 1
    return 0

def sofa_cns(gcs):
    if gcs is None or np.isnan(gcs): return 0
    if gcs < 6: return 4
    if gcs < 10: return 3
    if gcs < 13: return 2
    if gcs < 15: return 1
    return 0

def sofa_renal(creat, urine_ml_24h):
    score = 0
    if creat is not None and not np.isnan(creat):
        if creat >= 5.0: score = 4
        elif creat >= 3.5: score = max(score, 3)
        elif creat >= 2.0: score = max(score, 2)
        elif creat >= 1.2: score = max(score, 1)
    if urine_ml_24h is not None and not np.isnan(urine_ml_24h):
        if urine_ml_24h < 200: score = max(score, 4)
        elif urine_ml_24h < 500: score = max(score, 3)
    return score


# ── hourly, window-restricted SOFA ───────────────────────────────────────────
def _worst(h, v, t, agg):
    """Worst value in the 24h lookback (t-24h, t] for a component time series."""
    if len(h) == 0:
        return np.nan
    m = (h > t - SOFA_LOOKBACK_H) & (h <= t)
    if not m.any():
        return np.nan
    return v[m].max() if agg == "max" else v[m].min()


def _sepsis_from_series(C, si_h, pre_h=SOFA_WINDOW_PRE_H, post_h=SOFA_WINDOW_POST_H):
    """C: per-component (hours, values) arrays; si_h: suspicion time (hours since
    intime). Evaluates hourly SOFA (24h lookback) over [si-pre_h, si+post_h] and
    returns (is_sepsis, max_sofa_in_window). pre_h=post_h=0 -> single-point
    (SOFA at exactly the suspicion time), the stricter mimic-code-literal variant."""
    lo, hi = si_h - pre_h, si_h + post_h
    best = 0
    best_cardio = 0
    t = lo
    while t <= hi:
        pao2 = _worst(*C["pao2"], t, "min")
        fio2 = _worst(*C["fio2"], t, "max")
        pf = np.nan
        if not np.isnan(pao2) and not np.isnan(fio2) and fio2 > 0:
            f = fio2 / 100.0 if fio2 > 1.0 else fio2
            pf = pao2 / f if f > 0 else np.nan
        # urine: rolling 24h sum == mL/24h directly
        hu, vu = C["urine"]
        mu = (hu > t - SOFA_LOOKBACK_H) & (hu <= t)
        urine = vu[mu].sum() if mu.any() else np.nan
        # vaso: any infusion overlapping the lookback window
        vs, ve = C["vaso"]
        vaso = bool(np.any((vs <= t) & (ve > t - SOFA_LOOKBACK_H))) if len(vs) else False
        cardio = sofa_cardiovascular(_worst(*C["map"], t, "min"), vaso)
        sofa = (sofa_respiration(pf)
                + sofa_coagulation(_worst(*C["platelets"], t, "min"))
                + sofa_liver(_worst(*C["bilirubin"], t, "max"))
                + cardio
                + sofa_cns(_worst(*C["gcs"], t, "min"))
                + sofa_renal(_worst(*C["creatinine"], t, "max"), urine))
        if sofa > best:
            best = sofa
            best_cardio = cardio
        t += 1.0
    # cardio_alone: SOFA>=2 reached, but drops below 2 without the cardiovascular
    # component (checks the vasopressor-presence approximation isn't inflating).
    cardio_alone = int(best >= 2 and (best - best_cardio) < 2)
    return int(best >= 2), best, cardio_alone


def _score_stays(suspicion, C_by, all_ids, mode="window",
                 pre_h=SOFA_WINDOW_PRE_H, post_h=SOFA_WINDOW_POST_H):
    """Score every suspected-infection stay under BOTH the window and the
    single-point variant, so one data-read pass reports both rates. `mode`
    selects which becomes the returned label.

    `pre_h`/`post_h` set the suspicion window. They default to the [-48,+24]
    hours of the original Sepsis-3 operationalization; other widths within the
    accepted range are equally valid, and varying them is how detector 1's
    specificity against legitimate labeling variation is measured."""
    labels = {sid: 0 for sid in all_ids}
    n_win = n_pt = 0
    n_cardio_alone = 0      # positive labels (chosen mode) driven by cardio alone
    n_3plus_imputed = 0     # positive stays with >=3 of 6 organs fully unobserved
    for sid, si_h in suspicion.items():
        w, _, _ = _sepsis_from_series(C_by[sid], si_h, pre_h, post_h)
        p, _, ca = _sepsis_from_series(C_by[sid], si_h, 0.0, 0.0)
        n_win += w
        n_pt += p
        chosen = w if mode == "window" else p
        labels[sid] = chosen
        if chosen:
            # cardio_alone from the single-point eval (ca); for window mode this is
            # a lower bound, adequate for the sanity check.
            n_cardio_alone += ca
            C = C_by[sid]
            organ_obs = [
                len(C["pao2"][0]) and len(C["fio2"][0]),   # respiration
                len(C["platelets"][0]),                    # coagulation
                len(C["bilirubin"][0]),                    # liver
                len(C["map"][0]) or len(C["vaso"][0]),     # cardiovascular
                len(C["gcs"][0]),                          # CNS
                len(C["creatinine"][0]) or len(C["urine"][0]),  # renal
            ]
            if sum(1 for o in organ_obs if not o) >= 3:
                n_3plus_imputed += 1
    extras = {"n_cardio_alone_pos": n_cardio_alone,
              "n_3plus_organs_imputed_pos": n_3plus_imputed}
    return labels, n_win, n_pt, extras


# ── data readers ─────────────────────────────────────────────────────────────
def _find(path):
    """Resolve a table path tolerating .csv(.gz) and filename CASE differences.
    The eICU demo ships lower-cased names (infusiondrug.csv.gz) while the full
    eICU-CRD uses camelCase (infusionDrug.csv.gz); match either."""
    for cand in (path, path.replace(".csv.gz", ".csv")):
        if os.path.exists(cand):
            return cand
    d, base = os.path.split(path)
    alt = os.path.basename(path.replace(".csv.gz", ".csv"))
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower() in (base.lower(), alt.lower()):
                return os.path.join(d, f)
    return None


def _read_csv(path, **kw):
    p = _find(path)
    if p is None:
        logging.warning(f"[A1/SOFA] missing {os.path.basename(path)} — components absent")
        return None
    return pd.read_csv(p, encoding_errors="replace", **kw)


def _read_filtered(path, usecols, id_col, keep_ids, extra=None, chunk=1_000_000):
    """Chunk-read a large CSV, filtering to keep_ids (and an optional `extra`
    per-chunk filter) immediately so peak memory stays bounded on full-scale
    MIMIC/eICU (chartevents/nurseCharting/vitalPeriodic are tens of GB)."""
    p = _find(path)
    if p is None:
        logging.warning(f"[A1/SOFA] missing {os.path.basename(path)} — components absent")
        return None
    keep_ids = set(keep_ids)
    parts, beat, n = [], make_heartbeat(f"A1 read {os.path.basename(path)}"), 0
    for c in pd.read_csv(p, usecols=usecols, chunksize=chunk,
                         encoding_errors="replace", low_memory=False):
        n += 1
        c = c[c[id_col].isin(keep_ids)]
        if extra is not None:
            c = extra(c)
        if len(c):
            parts.append(c)
        beat(n)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)


def _hours_since(times, intime):
    return (pd.to_datetime(times) - intime).dt.total_seconds().values / 3600.0


def mimic_sofa_sepsis_labels(data_path, stays_df, mode="window",
                             pre_h=SOFA_WINDOW_PRE_H, post_h=SOFA_WINDOW_POST_H):
    """Per-stay Sepsis-3 labels for MIMIC-IV.

    stays_df needs: stay_id, hadm_id, subject_id, intime (datetime).
    Returns (labels {stay_id:0/1}, diagnostics dict).
    """
    stays = stays_df.copy()
    stays["intime"] = pd.to_datetime(stays["intime"])
    stay_ids = set(stays["stay_id"].astype(int))
    intime = stays.set_index("stay_id")["intime"].to_dict()
    hadm_to_stay = (stays.dropna(subset=["hadm_id"]).set_index("hadm_id")["stay_id"]
                    .astype(int).to_dict())

    # 1) suspected infection -> {stay_id: si_hours_since_intime}. Only these
    #    stays can be positive, so SOFA is computed for them alone (major speedup).
    suspicion = _mimic_suspicion(data_path, stays, hadm_to_stay, intime)
    infected = set(suspicion)

    # 2) timestamped SOFA components for infected stays only.
    C_by_stay = {sid: {k: (np.array([]), np.array([])) for k in
                       ["creatinine", "bilirubin", "platelets", "fio2", "pao2",
                        "gcs", "map", "urine"]} | {"vaso": (np.array([]), np.array([]))}
                 for sid in infected}
    observed = {k: set() for k in
                ["creatinine", "bilirubin", "platelets", "fio2", "pao2", "gcs",
                 "map", "urine", "vaso"]}
    if infected:
        _mimic_fill_components(data_path, stays, infected, hadm_to_stay, intime,
                               C_by_stay, observed)

    # 3) score (both modes; `mode` picks the returned label)
    labels, n_win, n_pt, extras = _score_stays(suspicion, C_by_stay, stay_ids, mode=mode,
                                              pre_h=pre_h, post_h=post_h)
    n_pos = sum(labels.values())

    n = max(1, len(stay_ids))
    ninf = max(1, len(infected))
    diagnostics = {
        "n_stays": len(stay_ids),
        "mode": mode,
        "n_positive": n_pos,
        "positive_rate": n_pos / n,
        "positive_rate_window": n_win / n,
        "positive_rate_singlepoint": n_pt / n,
        "n_suspected_infection": len(infected),
        # observed fraction among the infected stays where SOFA is evaluated
        "observed_fraction": {k: len(observed[k]) / ninf for k in observed},
        "cardio_dose_approx": True,
        "frac_pos_cardio_alone": extras["n_cardio_alone_pos"] / max(1, n_pos),
        "frac_pos_3plus_organs_imputed": extras["n_3plus_organs_imputed_pos"] / max(1, n_pos),
    }
    logging.info(f"[A1/SOFA MIMIC mode={mode}] {n_pos}/{len(stay_ids)} sepsis+ "
                 f"({diagnostics['positive_rate']:.1%}); window={n_win/n:.1%} "
                 f"single-point={n_pt/n:.1%}; infection in {len(infected)} stays")
    return labels, diagnostics


def _mimic_suspicion(data_path, stays, hadm_to_stay, intime):
    """Asymmetric Sepsis-3 window. Returns {stay_id: si_hours_since_intime}."""
    hadm_ids = set(stays["hadm_id"].dropna().astype(int))

    presc = _read_filtered(os.path.join(data_path, "hosp", "prescriptions.csv.gz"),
                           ["hadm_id", "drug", "starttime"], "hadm_id", hadm_ids,
                           extra=lambda c: c[c["drug"].map(is_antibiotic)])
    abx = {}
    if presc is not None and len(presc):
        presc = presc.copy()
        presc["starttime"] = pd.to_datetime(presc["starttime"], errors="coerce")
        for hid, t in presc.dropna(subset=["starttime"]).groupby("hadm_id")["starttime"]:
            abx[int(hid)] = sorted(t)

    micro = _read_filtered(os.path.join(data_path, "hosp", "microbiologyevents.csv.gz"),
                           ["hadm_id", "charttime", "chartdate"], "hadm_id", hadm_ids)
    cx = {}
    if micro is not None and len(micro):
        t = pd.to_datetime(micro["charttime"], errors="coerce")
        t = t.fillna(pd.to_datetime(micro["chartdate"], errors="coerce"))
        micro = micro.assign(t=t).dropna(subset=["t"])
        for hid, tt in micro.groupby("hadm_id")["t"]:
            cx[int(hid)] = sorted(tt)

    out = {}
    for hid in hadm_ids:
        a, c = abx.get(hid, []), cx.get(hid, [])
        if not a or not c or hid not in hadm_to_stay:
            continue
        si_time = None
        for ta in a:                       # abx first -> culture within +24h
            for tc in c:
                if ta <= tc <= ta + pd.Timedelta(hours=24):
                    si_time = ta if si_time is None else min(si_time, ta)
        for tc in c:                       # culture first -> abx within +72h
            for ta in a:
                if tc <= ta <= tc + pd.Timedelta(hours=72):
                    si_time = tc if si_time is None else min(si_time, tc)
        if si_time is not None:
            sid = int(hadm_to_stay[hid])
            out[sid] = (si_time - intime[sid]).total_seconds() / 3600.0
    return out


def _mimic_fill_components(data_path, stays, infected, hadm_to_stay, intime,
                           C_by_stay, observed):
    from .variables import MIMIC_ITEM_IDS
    inf_hadm = set(h for h, s in hadm_to_stay.items() if s in infected)

    def _store(sid, key, hours, vals):
        good = ~np.isnan(hours) & ~np.isnan(vals)
        if good.any():
            C_by_stay[sid][key] = (hours[good], vals[good])
            observed[key].add(sid)

    # labevents (creatinine, bilirubin, platelets) — hadm-keyed, needs charttime
    lab_map = {iid: name for name, ids in MIMIC_LAB_IDS.items() for iid in ids}
    labs = _read_filtered(
        os.path.join(data_path, "hosp", "labevents.csv.gz"),
        ["hadm_id", "itemid", "charttime", "valuenum"], "hadm_id", inf_hadm,
        extra=lambda c: c[c["itemid"].isin(lab_map) & c["valuenum"].notna()])
    if labs is not None and len(labs):
        labs["name"] = labs["itemid"].map(lab_map)
        labs["stay_id"] = labs["hadm_id"].map(hadm_to_stay)
        for (sid, name), g in labs.groupby(["stay_id", "name"]):
            sid = int(sid)
            _store(sid, name, _hours_since(g["charttime"], intime[sid]), g["valuenum"].values)

    # chartevents (pao2, fio2, gcs components, MAP)
    chart_ids = {iid: k for k, ids in MIMIC_CHART_IDS.items() for iid in ids}
    map_ids = {iid: "map" for iid in MIMIC_ITEM_IDS["MAP"]}
    chart_ids = {**chart_ids, **map_ids}
    ce = _read_filtered(
        os.path.join(data_path, "icu", "chartevents.csv.gz"),
        ["stay_id", "itemid", "charttime", "valuenum"], "stay_id", infected,
        extra=lambda c: c[c["itemid"].isin(chart_ids) & c["valuenum"].notna()])
    if ce is not None and len(ce):
        ce["k"] = ce["itemid"].map(chart_ids)
        for k in ["pao2", "fio2", "map"]:
            for sid, g in ce[ce["k"] == k].groupby("stay_id"):
                sid = int(sid)
                _store(sid, k, _hours_since(g["charttime"], intime[sid]), g["valuenum"].values)
        # GCS total per charttime = eye+verbal+motor
        gcs = ce[ce["k"].isin(["gcs_eye", "gcs_verbal", "gcs_motor"])]
        for sid, g in gcs.groupby("stay_id"):
            sid = int(sid)
            piv = g.pivot_table(index="charttime", columns="k", values="valuenum", aggfunc="min")
            tot = piv.sum(axis=1, min_count=1).dropna()
            if len(tot):
                _store(sid, "gcs", _hours_since(pd.Series(tot.index), intime[sid]), tot.values)

    # outputevents (urine)
    oe = _read_filtered(
        os.path.join(data_path, "icu", "outputevents.csv.gz"),
        ["stay_id", "itemid", "charttime", "value"], "stay_id", infected,
        extra=lambda c: c[c["itemid"].isin(MIMIC_URINE_IDS) & c["value"].notna()])
    if oe is not None and len(oe):
        for sid, g in oe.groupby("stay_id"):
            sid = int(sid)
            _store(sid, "urine", _hours_since(g["charttime"], intime[sid]), g["value"].values)

    # inputevents (vasopressor intervals: start/end hours)
    ie = _read_filtered(
        os.path.join(data_path, "icu", "inputevents.csv.gz"),
        ["stay_id", "itemid", "starttime", "endtime"], "stay_id", infected,
        extra=lambda c: c[c["itemid"].isin(MIMIC_VASOPRESSOR_IDS)])
    if ie is not None and len(ie):
        for sid, g in ie.groupby("stay_id"):
            sid = int(sid)
            vs = _hours_since(g["starttime"], intime[sid])
            ve = _hours_since(g["endtime"], intime[sid])
            good = ~np.isnan(vs)
            ve = np.where(np.isnan(ve), vs, ve)
            if good.any():
                C_by_stay[sid]["vaso"] = (vs[good], ve[good])
                observed["vaso"].add(sid)


# ── eICU end-to-end labeler ──────────────────────────────────────────────────
def _eicu_offset_hours(offset_min):
    return pd.to_numeric(offset_min, errors="coerce").values / 60.0


def eicu_sofa_sepsis_labels(data_path, patients_df, mode="window",
                            pre_h=SOFA_WINDOW_PRE_H, post_h=SOFA_WINDOW_POST_H):
    """Per-stay Sepsis-3 labels for eICU. patients_df needs patientunitstayid.
    All eICU times are minute-offsets from unit admission (0h). Returns
    (labels {stay_id:0/1}, diagnostics)."""
    pid_set = set(patients_df["patientunitstayid"].astype(int))

    suspicion = _eicu_suspicion(data_path, pid_set)   # {pid: si_hours}
    infected = set(suspicion)

    keys = ["creatinine", "bilirubin", "platelets", "fio2", "pao2", "gcs",
            "map", "urine"]
    C_by = {pid: {k: (np.array([]), np.array([])) for k in keys}
                 | {"vaso": (np.array([]), np.array([]))} for pid in infected}
    observed = {k: set() for k in keys + ["vaso"]}
    if infected:
        _eicu_fill_components(data_path, infected, C_by, observed)

    labels, n_win, n_pt, extras = _score_stays(suspicion, C_by, pid_set, mode=mode,
                                              pre_h=pre_h, post_h=post_h)
    n_pos = sum(labels.values())

    n = max(1, len(pid_set)); ninf = max(1, len(infected))
    diagnostics = {
        "n_stays": len(pid_set), "mode": mode, "n_positive": n_pos,
        "positive_rate": n_pos / n,
        "positive_rate_window": n_win / n,
        "positive_rate_singlepoint": n_pt / n,
        "n_suspected_infection": len(infected),
        "observed_fraction": {k: len(observed[k]) / ninf for k in observed},
        "cardio_dose_approx": True,
        "frac_pos_cardio_alone": extras["n_cardio_alone_pos"] / max(1, n_pos),
        "frac_pos_3plus_organs_imputed": extras["n_3plus_organs_imputed_pos"] / max(1, n_pos),
    }
    logging.info(f"[A1/SOFA eICU mode={mode}] {n_pos}/{len(pid_set)} sepsis+ "
                 f"({diagnostics['positive_rate']:.1%}); window={n_win/n:.1%} "
                 f"single-point={n_pt/n:.1%}; infection in {len(infected)} stays")
    return labels, diagnostics


def _eicu_suspicion(data_path, pid_set):
    """Asymmetric window. Returns {pid: si_hours}. abx from medication.drugstartoffset,
    cultures from microLab.culturetakenoffset (minutes)."""
    med = _read_filtered(os.path.join(data_path, "medication.csv.gz"),
                         ["patientunitstayid", "drugname", "drugstartoffset"],
                         "patientunitstayid", pid_set,
                         extra=lambda c: c[c["drugname"].map(is_antibiotic)])
    abx = {}
    if med is not None and len(med):
        med = med.copy()
        med["h"] = _eicu_offset_hours(med["drugstartoffset"])
        for pid, g in med.dropna(subset=["h"]).groupby("patientunitstayid"):
            abx[int(pid)] = sorted(g["h"].tolist())

    # Corroborating infection evidence = culture draw OR infection diagnosis.
    cx = {}
    mic = _read_filtered(os.path.join(data_path, "microLab.csv.gz"),
                         ["patientunitstayid", "culturetakenoffset"],
                         "patientunitstayid", pid_set)
    if mic is not None and len(mic):
        mic = mic.copy()
        mic["h"] = _eicu_offset_hours(mic["culturetakenoffset"])
        for pid, g in mic.dropna(subset=["h"]).groupby("patientunitstayid"):
            cx.setdefault(int(pid), []).extend(g["h"].tolist())

    dx = _read_filtered(
        os.path.join(data_path, "diagnosis.csv.gz"),
        ["patientunitstayid", "diagnosisstring", "diagnosisoffset"],
        "patientunitstayid", pid_set,
        extra=lambda c: c[c["diagnosisstring"].astype(str).str.lower().apply(
            lambda s: any(p in s for p in INFECTION_DX_PATTERNS))])
    if dx is not None and len(dx):
        dx = dx.copy()
        dx["h"] = _eicu_offset_hours(dx["diagnosisoffset"])
        for pid, g in dx.dropna(subset=["h"]).groupby("patientunitstayid"):
            cx.setdefault(int(pid), []).extend(g["h"].tolist())

    cx = {pid: sorted(v) for pid, v in cx.items()}

    out = {}
    for pid in pid_set:
        a, c = abx.get(pid, []), cx.get(pid, [])
        if not a or not c:
            continue
        si = None
        for ta in a:                         # abx first -> culture within +24h
            for tc in c:
                if ta <= tc <= ta + 24:
                    si = ta if si is None else min(si, ta)
        for tc in c:                         # culture first -> abx within +72h
            for ta in a:
                if tc <= ta <= tc + 72:
                    si = tc if si is None else min(si, tc)
        if si is not None:
            out[int(pid)] = si
    return out


def _eicu_fill_components(data_path, infected, C_by, observed):
    def _store(pid, key, hours, vals):
        hours = np.asarray(hours, float); vals = np.asarray(vals, float)
        good = ~np.isnan(hours) & ~np.isnan(vals)
        if good.any():
            C_by[pid][key] = (hours[good], vals[good]); observed[key].add(pid)

    # lab: creatinine/bilirubin/platelets/pao2/fio2
    lab = _read_filtered(os.path.join(data_path, "lab.csv.gz"),
                         ["patientunitstayid", "labname", "labresult", "labresultoffset"],
                         "patientunitstayid", infected)
    if lab is not None and len(lab):
        lab["ln"] = lab["labname"].astype(str).str.strip().str.lower()
        name_map = {v: k for k, v in EICU_LABNAMES.items()}
        lab["key"] = lab["ln"].map(name_map)
        lab = lab[lab["key"].notna()]
        lab["h"] = _eicu_offset_hours(lab["labresultoffset"])
        for (pid, key), g in lab.groupby(["patientunitstayid", "key"]):
            _store(int(pid), key, g["h"].values, pd.to_numeric(g["labresult"], errors="coerce").values)

    # GCS total from nurseCharting
    nc = _read_filtered(os.path.join(data_path, "nurseCharting.csv.gz"),
                        ["patientunitstayid", "nursingchartoffset",
                         "nursingchartcelltypevallabel", "nursingchartvalue"],
                        "patientunitstayid", infected)
    if nc is not None and len(nc):
        lbl = nc["nursingchartcelltypevallabel"].astype(str).str.lower()
        nc = nc[lbl.str.contains("glasgow coma score") | lbl.str.contains("score (glasgow", regex=False)]
        nc["h"] = _eicu_offset_hours(nc["nursingchartoffset"])
        for pid, g in nc.groupby("patientunitstayid"):
            _store(int(pid), "gcs", g["h"].values, pd.to_numeric(g["nursingchartvalue"], errors="coerce").values)

    # urine from intakeOutput (celllabel contains 'urine')
    io = _read_filtered(
        os.path.join(data_path, "intakeOutput.csv.gz"),
        ["patientunitstayid", "intakeoutputoffset", "celllabel", "cellvaluenumeric"],
        "patientunitstayid", infected,
        extra=lambda c: c[c["celllabel"].astype(str).str.lower().str.contains("urine")])
    if io is not None and len(io):
        io = io.copy()
        io["h"] = _eicu_offset_hours(io["intakeoutputoffset"])
        for pid, g in io.groupby("patientunitstayid"):
            _store(int(pid), "urine", g["h"].values, pd.to_numeric(g["cellvaluenumeric"], errors="coerce").values)

    # MAP from vitalAperiodic (noninvasivemean) — smaller than vitalPeriodic
    va = _read_filtered(os.path.join(data_path, "vitalAperiodic.csv.gz"),
                        ["patientunitstayid", "observationoffset", "noninvasivemean"],
                        "patientunitstayid", infected,
                        extra=lambda c: c[c["noninvasivemean"].notna()])
    if va is not None and len(va):
        va["h"] = _eicu_offset_hours(va["observationoffset"])
        for pid, g in va.groupby("patientunitstayid"):
            _store(int(pid), "map", g["h"].values, pd.to_numeric(g["noninvasivemean"], errors="coerce").values)

    # vasopressor presence from infusiondrug (point events -> start==end)
    inf = _read_filtered(os.path.join(data_path, "infusiondrug.csv.gz"),
                         ["patientunitstayid", "infusionoffset", "drugname"],
                         "patientunitstayid", infected,
                         extra=lambda c: c[c["drugname"].map(is_vasopressor)])
    if inf is not None and len(inf):
        inf = inf.copy()
        inf["h"] = _eicu_offset_hours(inf["infusionoffset"])
        for pid, g in inf.dropna(subset=["h"]).groupby("patientunitstayid"):
            hrs = g["h"].values
            C_by[int(pid)]["vaso"] = (hrs, hrs)  # point events; overlap==presence in lookback
            observed["vaso"].add(int(pid))
