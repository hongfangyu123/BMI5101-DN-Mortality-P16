"""Model definitions shared by stage 4 (fitting / tuning) and stage 4c (thresholds, model comparison)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

SCALED_MODELS = {"SVM", "LR"}   # fitted on standardised features; tree models on imputed raw values


def paper_models(seed: int) -> dict:
    """Hyper-parameters reported in the paper, mapped to scikit-learn / xgboost (see decisions.md D4.4)."""
    return {
        "RF": RandomForestClassifier(n_estimators=9, max_features=2, max_depth=10, min_samples_split=30,
                                     min_samples_leaf=10, random_state=seed, n_jobs=1),
        "RF (R defaults)": RandomForestClassifier(n_estimators=9, max_features=2, max_depth=None, min_samples_split=2,
                                                  min_samples_leaf=1, random_state=seed, n_jobs=1),
        "XGBoost": XGBClassifier(n_estimators=50, max_depth=1, learning_rate=0.3, gamma=0, colsample_bytree=0.6,
                                 min_child_weight=1, subsample=1, objective="binary:logistic", eval_metric="logloss",
                                 random_state=seed, n_jobs=1),
        "SVM": SVC(kernel="rbf", gamma=0.1, C=1.0, probability=True, random_state=seed),
        "LR": LogisticRegression(C=np.inf, max_iter=10000),
    }


def tuning_grids(p: int, seed: int) -> dict:
    mf = sorted({1, 2, 3, max(1, int(np.sqrt(p))), max(1, p // 2)})
    return {
        "RF": (RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=1),
               {"max_features": mf, "min_samples_leaf": [1, 3, 5, 10, 20], "max_depth": [None, 4, 6, 10]}),
        "XGBoost": (XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=seed, n_jobs=1),
                    {"n_estimators": [50, 100, 200, 400], "max_depth": [1, 2, 3], "learning_rate": [0.03, 0.1, 0.3],
                     "colsample_bytree": [0.6, 1.0], "subsample": [0.8, 1.0], "min_child_weight": [1, 5]}),
        "SVM": (SVC(kernel="rbf", random_state=seed),
                {"C": [0.01, 0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1, "scale"]}),
        "LR": (LogisticRegression(l1_ratio=0.0, max_iter=10000),
               {"C": [0.001, 0.01, 0.1, 1, 10, 100, np.inf]}),
    }


def tuned_model(name: str, params: dict, p: int, seed: int):
    """Rebuild a tuned model from `tuned_hyperparameters.json` (SVM with Platt probabilities, as refitted in stage 4)."""
    params = {k: (np.inf if v == "inf" else v) for k, v in params.items()}
    est = tuning_grids(p, seed)[name][0]
    if name == "SVM":
        return SVC(kernel="rbf", probability=True, random_state=seed, **params)
    return est.set_params(**params)
