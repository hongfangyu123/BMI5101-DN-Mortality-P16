"""Predictor definitions shared by all stages (55 predictors listed in the paper's Data extraction)."""

FEATURES = [
    "age", "gender_male",
    "myocardial_infarct", "congestive_heart_failure", "chronic_pulmonary_disease", "renal_disease",
    "hypertension", "obesity",
    "glucose_mean", "aniongap_min", "aniongap_max", "bicarbonate_min", "bicarbonate_max",
    "creatinine_min", "creatinine_max", "bun_min", "bun_max",
    "hemoglobin_mean", "platelets_mean", "wbc_mean", "rbc_mean", "rdw_mean",
    "lymphocytes_mean", "neutrophils_mean", "monocytes_mean",
    "calcium_min", "calcium_max", "chloride_min", "chloride_max", "sodium_min", "sodium_max",
    "potassium_min", "potassium_max", "ptt_min", "ptt_max", "pt_min", "pt_max", "inr_min", "inr_max",
    "heart_rate_mean", "mbp_mean", "resp_rate_mean", "temperature_mean",
    "sapsii", "apsiii", "lods", "oasis", "gcs", "cci",
    "nlr", "plr", "lmr", "pnr", "pwr", "sii",
]
BINARY = ["gender_male", "myocardial_infarct", "congestive_heart_failure", "chronic_pulmonary_disease",
          "renal_disease", "hypertension", "obesity"]
COUNT = ["gcs", "cci"]          # ordinal integer scores -> mode imputation ("counting data")
RATIOS = ["nlr", "plr", "lmr", "pnr", "pwr", "sii"]
MODE_VARS = BINARY + COUNT
assert len(FEATURES) == 55
