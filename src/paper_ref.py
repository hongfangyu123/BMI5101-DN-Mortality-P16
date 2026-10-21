"""Published reference values from Huang et al. 2025 (Table 1 'Total' column, results text)."""

# feature -> (summary type, value, q1_or_sd, q3)   type: 'median' -> median (Q1, Q3); 'mean' -> mean ± SD; 'pct' -> % yes
TABLE1_TOTAL = {
    "age": ("mean", 65.85, 13.65, None),
    "gender_male": ("pct", 58.57, None, None),
    "myocardial_infarct": ("pct", 31.15, None, None),
    "congestive_heart_failure": ("pct", 42.04, None, None),
    "chronic_pulmonary_disease": ("pct", 25.67, None, None),
    "renal_disease": ("pct", 45.32, None, None),
    "hypertension": ("pct", 37.32, None, None),
    "obesity": ("pct", 23.00, None, None),
    "aniongap_min": ("median", 13.00, 11.00, 15.00),
    "aniongap_max": ("median", 17.00, 14.00, 21.00),
    "bicarbonate_min": ("median", 21.00, 18.00, 24.00),
    "bicarbonate_max": ("median", 24.00, 22.00, 27.00),
    "creatinine_min": ("median", 1.20, 0.80, 2.10),
    "creatinine_max": ("median", 1.40, 1.00, 2.60),
    "bun_min": ("median", 24.00, 16.00, 39.00),
    "bun_max": ("median", 30.00, 19.00, 49.00),
    "calcium_min": ("median", 8.17, 7.80, 8.60),
    "calcium_max": ("median", 8.69, 8.20, 9.10),
    "chloride_min": ("median", 101.00, 96.00, 105.00),
    "chloride_max": ("mean", 105.03, 6.48, None),
    "sodium_min": ("median", 136.00, 133.00, 139.00),
    "sodium_max": ("median", 139.00, 137.00, 142.00),
    "potassium_min": ("mean", 4.04, 0.60, None),
    "potassium_max": ("median", 4.70, 4.30, 5.30),
    "ptt_min": ("median", 28.80, 25.90, 31.75),
    "ptt_max": ("median", 32.90, 28.30, 43.78),
    "pt_min": ("median", 13.10, 12.00, 14.45),
    "pt_max": ("median", 14.40, 12.70, 17.19),
    "inr_min": ("median", 1.20, 1.10, 1.32),
    "inr_max": ("median", 1.30, 1.10, 1.59),
    "glucose_mean": ("median", 163.00, 133.49, 210.75),
    "hemoglobin_mean": ("mean", 10.34, 1.89, None),
    "platelets_mean": ("median", 195.33, 144.29, 256.42),
    "wbc_mean": ("median", 11.26, 8.25, 15.20),
    "rbc_mean": ("mean", 3.53, 0.68, None),
    "rdw_mean": ("median", 14.60, 13.60, 16.05),
    "lymphocytes_mean": ("median", 3.69, 1.27, 118.32),
    "neutrophils_mean": ("median", 23.90, 8.63, 831.30),
    "monocytes_mean": ("median", 2.00, 0.60, 43.64),
    "heart_rate_mean": ("mean", 83.81, 14.63, None),
    "mbp_mean": ("median", 75.70, 70.15, 83.64),
    "resp_rate_mean": ("median", 18.46, 16.57, 20.53),
    "temperature_mean": ("mean", 36.84, 0.49, None),
    "sapsii": ("mean", 37.00, 13.24, None),
    "apsiii": ("median", 47.00, 37.00, 62.00),
    "lods": ("median", 5.00, 3.00, 7.00),
    "oasis": ("median", 30.00, 25.00, 37.00),
    "gcs": ("median", 14.00, 13.00, 15.00),
    "cci": ("mean", 7.80, 2.54, None),
    "nlr": ("median", 6.80, 3.89, 12.64),
    "plr": ("median", 36.48, 1.83, 148.66),
    "lmr": ("median", 2.42, 1.32, 4.14),
    "pnr": ("median", 6.58, 0.25, 20.23),
    "pwr": ("median", 17.40, 11.82, 24.92),
    "sii": ("median", 1341.68, 671.58, 2671.13),
}

COHORT_FLOW = [("DN patients", 4137), ("first ICU at first hospitalization", 2144), ("LOS > 48h", 2046),
               ("age > 18", 2046), ("CBC complete", 1313)]
N_SURVIVAL, N_DEATH = 812, 501

LASSO_SELECTED = ["age", "obesity", "aniongap_min", "rdw_mean", "neutrophils_mean", "chloride_max",
                  "ptt_min", "pt_min", "apsiii", "cci", "lmr", "pwr"]
LASSO_LAMBDA_MIN, LASSO_LAMBDA_1SE = 0.01055914, 0.02687268

AUC_TRAIN = {"RF": (0.999, 0.998, 1.000), "SVM": (0.762, 0.735, 0.788), "XGBoost": (0.837, 0.813, 0.861),
             "LR": (0.792, 0.765, 0.819)}
AUC_VALID = {"RF": (0.780, 0.723, 0.837), "SVM": (0.679, 0.625, 0.734), "XGBoost": (0.732, 0.672, 0.793),
             "LR": (0.773, 0.718, 0.829)}
# RF validation metrics as reported (they correspond to survival = positive class, see decisions.md)
RF_VALID_METRICS = {"accuracy": 0.721, "precision": 0.735, "recall": 0.858, "f1": 0.792}
SHAP_TOP5 = ["rdw_mean", "neutrophils_mean", "cci", "chloride_max", "pt_min"]
