"""Evaluation helpers: DeLong AUC CI and paired test, threshold metrics (Youden cut-off), decision curve and
clinical impact curve."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def _midrank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def delong_auc_ci(y: np.ndarray, p: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """AUC with DeLong (1988) variance, fast algorithm of Sun & Xu (2014)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    var = np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(var)
    return float(auc), float(max(0.0, auc - z * se)), float(min(1.0, auc + z * se))


def threshold_metrics(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(p) >= threshold).astype(int)
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat, zero_division=0),
            "specificity": recall_score(1 - y, 1 - yhat, zero_division=0)}


def youden_threshold(y, p) -> tuple[float, float]:
    """Cut-off maximising sensitivity + specificity − 1 (positive if p >= cut-off); returns (threshold, J)."""
    fpr, tpr, thr = roc_curve(np.asarray(y).astype(int), np.asarray(p, float))
    j = tpr - fpr
    i = int(np.argmax(j[1:])) + 1          # skip the +inf threshold of roc_curve
    return float(thr[i]), float(j[i])


def delong_roc_test(y, p1, p2) -> dict:
    """Paired DeLong test for two correlated AUCs on the same cases (Sun & Xu 2014 structural components)."""
    y = np.asarray(y).astype(int)
    P = np.vstack([np.asarray(p1, float), np.asarray(p2, float)])
    pos, neg = P[:, y == 1], P[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    aucs, v01, v10 = [], [], []
    for r in range(2):
        tx, ty, tz = _midrank(pos[r]), _midrank(neg[r]), _midrank(np.concatenate([pos[r], neg[r]]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[:m] - tx) / n)
        v10.append(1 - (tz[m:] - ty) / m)
    S = np.cov(np.array(v01)) / m + np.cov(np.array(v10)) / n
    diff = aucs[0] - aucs[1]
    se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    z = diff / se
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "diff": float(diff), "se_diff": float(se),
            "diff_lo": float(diff - 1.959964 * se), "diff_hi": float(diff + 1.959964 * se),
            "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}


def evaluate(y, p, threshold: float = 0.5) -> dict:
    auc, lo, hi = delong_auc_ci(y, p)
    assert abs(auc - roc_auc_score(y, p)) < 1e-9
    return {"auc": auc, "auc_lo": lo, "auc_hi": hi, **threshold_metrics(y, p, threshold)}


def auprc(y, p) -> float:
    """Area under the precision-recall curve as average precision (step interpolation, no trapezoid)."""
    return float(average_precision_score(np.asarray(y).astype(int), np.asarray(p, float)))


def scaled_brier(y, p) -> float:
    """1 - Brier / Brier of the prevalence-only model, prevalence taken from the same data (D8.6)."""
    y = np.asarray(y).astype(int)
    prev = y.mean()
    return float(1 - brier_score_loss(y, p) / (prev * (1 - prev)))


def nadeau_bengio(diffs, n_train: float, n_test: float, k: int, r: int, alpha: float = 0.05) -> dict:
    """Corrected repeated k-fold CV t-test (Nadeau & Bengio 2003; Bouckaert & Frank 2004) for paired differences.

    diffs: the k*r per-fold paired differences (model A - model B) on the same splits.
    variance = (1/(k*r) + n_test/n_train) * s^2, with s^2 the sample variance (ddof=1) of all k*r differences;
    t distribution with k*r - 1 degrees of freedom.
    """
    d = np.asarray(diffs, float)
    assert d.ndim == 1 and len(d) == k * r, f"expected {k * r} paired differences, got {len(d)}"
    mean = float(d.mean())
    s2 = float(d.var(ddof=1))
    se = float(np.sqrt((1.0 / (k * r) + n_test / n_train) * s2))
    df = k * r - 1
    q = float(stats.t.ppf(1 - alpha / 2, df))
    t = mean / se if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t), df)) if se > 0 else float("nan")
    return {"mean_diff": mean, "sd_diff": float(np.sqrt(s2)), "se_corrected": se, "lo": mean - q * se, "hi": mean + q * se,
            "t": t, "df": df, "p_value": p, "k": k, "r": r}


def cal_intercept_slope(y, p) -> tuple[float, float]:
    """Calibration-in-the-large (intercept with the slope fixed at 1) and calibration slope (logistic recalibration)."""
    import statsmodels.api as sm
    y = np.asarray(y).astype(int)
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(q / (1 - q))
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    citl = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return float(citl), float(slope)


def cal_intercept_slope_checked(y, p, max_abs: float = 1e3) -> dict:
    """cal_intercept_slope with a convergence check (stage 8c): a value is set to NaN when its GLM fit raises, does not
    converge, emits a perfect-separation/convergence warning, or is non-finite or |value| > max_abs.
    Returns {"cal_intercept", "cal_slope", "cal_intercept_ok", "cal_slope_ok"}."""
    import warnings
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning
    y = np.asarray(y).astype(int)
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(q / (1 - q))

    def fit(exog, offset, j):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                r = sm.GLM(y, exog, family=sm.families.Binomial(), offset=offset).fit()
            bad = any(issubclass(x.category, (ConvergenceWarning, PerfectSeparationWarning)) for x in w)
            v = float(r.params[j])
            ok = bool(r.converged) and not bad and np.isfinite(v) and abs(v) <= max_abs
        except Exception:
            v, ok = np.nan, False
        return (v if ok else np.nan), ok

    slope, slope_ok = fit(sm.add_constant(lp), None, 1)
    citl, citl_ok = fit(np.ones_like(lp), lp, 0)
    return {"cal_intercept": citl, "cal_slope": slope, "cal_intercept_ok": citl_ok, "cal_slope_ok": slope_ok}


def decision_curve(y, p, thresholds=None) -> pd.DataFrame:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    thresholds = np.arange(0.01, 1.0, 0.01) if thresholds is None else thresholds
    n, prev = len(y), y.mean()
    rows = []
    for t in thresholds:
        pred = p >= t
        tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
        w = t / (1 - t)
        rows.append({"threshold": t, "net_benefit_model": tp / n - fp / n * w,
                     "net_benefit_all": prev - (1 - prev) * w, "net_benefit_none": 0.0})
    return pd.DataFrame(rows)


def useful_range(dca: pd.DataFrame) -> tuple[float, float]:
    """Threshold range where the model is useful: from the first threshold at which its net benefit exceeds both
    treat-all and treat-none, up to the last threshold before its net benefit first drops to <= 0. Small dips
    below treat-all near the lower end, where the model and treat-all curves coincide, are ignored."""
    nb, nb_all, t = dca.net_benefit_model.values, dca.net_benefit_all.values, dca.threshold.values
    ok = (nb > np.maximum(nb_all, 0) + 1e-9)
    if not ok.any():
        return float("nan"), float("nan")
    i = int(np.argmax(ok))
    pos = nb[i:] > 1e-9
    j = i + (int(np.argmin(pos)) if not pos.all() else len(pos))
    return float(t[i]), float(t[j - 1])


def clinical_impact(y, p, thresholds=None, population: int = 1000) -> pd.DataFrame:
    """Number classified high risk and number of true events among them, per `population` (rmda-style)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    thresholds = np.arange(0.01, 1.0, 0.01) if thresholds is None else thresholds
    n = len(y)
    return pd.DataFrame([{"threshold": t,
                          "n_high_risk": population * np.mean(p >= t),
                          "n_high_risk_with_event": population * np.sum((p >= t) & (y == 1)) / n}
                         for t in thresholds])
