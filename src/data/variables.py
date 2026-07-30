CANONICAL_VARIABLES = ["HR", "SBP", "DBP", "MAP", "SpO2", "pH", "HCO3", "pCO2", "PaO2"]

VAR_TO_IDX = {v: i for i, v in enumerate(CANONICAL_VARIABLES)}

PLAUS = {
    "HR":   (20, 300),
    "SBP":  (40, 300),
    "DBP":  (20, 200),
    "MAP":  (20, 200),
    "SpO2": (50, 100),
    "pH":   (6.5, 7.9),
    "HCO3": (5, 60),
    "pCO2": (10, 150),
    "PaO2": (20, 700),
}

ABG_VARIABLES = ["pH", "HCO3", "pCO2", "PaO2"]
HEMO_VARIABLES = ["SBP", "DBP", "MAP"]

PHYSIONET_MAPPING = {
    "HR": "HR",
    "SBP": "SBP",
    "DBP": "DBP",
    "MAP": "MAP",
    "SpO2": "O2Sat",
    "pH": "pH",
    "HCO3": "HCO3",
    "pCO2": "PaCO2",
}
PHYSIONET_EXTRA = {
    "SaO2": "SaO2",
}

MIMIC_ITEM_IDS = {
    "SBP": [220179, 220050],
    "DBP": [220180, 220051],
    "MAP": [220181, 220052],
    "HR":  [220045],
    "SpO2": [220277],
    "pH":  [50820],
    "HCO3": [50882],
    "pCO2": [50818],
    "PaO2": [50821],
}

EICU_VITAL_MAPPING = {
    "HR": "heartrate",
    "SBP": "systemicsystolic",
    "DBP": "systemicdiastolic",
    "MAP": "systemicmean",
    "SpO2": "sao2",
}

EICU_VITAL_FALLBACK = {
    "SBP": "noninvasivesystolic",
    "DBP": "noninvasivediastolic",
    "MAP": "noninvasivemean",
}

EICU_LAB_MAPPING = {
    "pH": "pH",
    "HCO3": "bicarbonate",
    "pCO2": "paCO2",
    "PaO2": "paO2",
}
