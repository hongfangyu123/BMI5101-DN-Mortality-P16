"""Imputation and standardisation shared by stage 3, the nested CV (04d) and the sensitivity analysis (04e).

Continuous -> mean; binary / ordinal count scores -> mode (decisions.md D3.3); z-score with population SD.
"""
from __future__ import annotations

import pandas as pd

from src.variables import MODE_VARS


def impute_fit(X: pd.DataFrame) -> pd.Series:
    fill = {}
    for c in X.columns:
        fill[c] = X[c].mode(dropna=True).iloc[0] if c in MODE_VARS else X[c].mean()
    return pd.Series(fill)


def scale_fit(X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return X.mean(), X.std(ddof=0)


def impute_scale(X: pd.DataFrame, fit_rows) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit imputer and scaler on `fit_rows` only; return (imputed raw-scale, standardised) for all rows."""
    fill = impute_fit(X.loc[fit_rows])
    Xi = X.fillna(fill)
    mu, sd = scale_fit(Xi.loc[fit_rows])
    return Xi, (Xi - mu) / sd


def missing_filter(X: pd.DataFrame, fit_rows, threshold: float = 0.20) -> list[str]:
    """Columns whose missing fraction on `fit_rows` is <= threshold (D3.2 rule; computed on training rows only, D8.4)."""
    frac = X.loc[fit_rows].isna().mean()
    return [c for c in X.columns if frac[c] <= threshold]
