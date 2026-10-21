"""glmnet-style L1 logistic regression with K-fold CV on binomial deviance (lambda.min / lambda.1se).

glmnet minimises  -loglik/n + lambda * ||beta||_1  (intercept unpenalised). scikit-learn's L1
LogisticRegression minimises  C * sum(logloss) + ||beta||_1, so C = 1 / (n * lambda).
Solver: liblinear with a large intercept_scaling (intercept penalty negligible); folds run in parallel.
Lambda grid as glmnet: 100 log-spaced values from lambda_max down to lambda_max * 1e-4 (n > p).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


@dataclass
class LassoCVResult:
    lambdas: np.ndarray        # descending
    cvm: np.ndarray            # mean held-out binomial deviance
    cvsd: np.ndarray           # standard error across folds
    nzero: np.ndarray          # non-zero coefficients on the full training data per lambda
    coefs: np.ndarray          # (n_lambda, p) path on the full training data
    lambda_min: float
    lambda_1se: float
    features: list[str]

    def selected(self, which: str = "1se") -> list[str]:
        lam = self.lambda_1se if which == "1se" else self.lambda_min
        i = int(np.argmin(np.abs(self.lambdas - lam)))
        return [f for f, c in zip(self.features, self.coefs[i]) if c != 0]


def lambda_max(X: np.ndarray, y: np.ndarray) -> float:
    return float(np.max(np.abs(X.T @ (y - y.mean()))) / len(y))


def _path(X, y, lambdas, seed):
    """L1 path with liblinear (newGLMNET coordinate descent, as glmnet). liblinear penalises the intercept
    as a feature of value `intercept_scaling`; with a large scaling that penalty (|b| / scaling) is negligible."""
    n = len(y)
    coefs, intercepts = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for lam in lambdas:
            m = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=1.0 / (n * lam), tol=1e-8,
                                   intercept_scaling=1e3, max_iter=10000, random_state=seed)
            m.fit(X, y)
            coefs.append(m.coef_.ravel().copy())
            intercepts.append(float(m.intercept_[0]))
    return np.array(coefs), np.array(intercepts)


def _deviance(X, y, coef, intercept):
    eta = X @ coef + intercept
    p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    return -2 * np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def cv_lasso(X: np.ndarray, y: np.ndarray, features: list[str], n_folds: int = 10, n_lambda: int = 100,
             min_ratio: float = 1e-4, seed: int = 42, n_jobs: int = 6) -> LassoCVResult:
    X, y = np.asarray(X, float), np.asarray(y, int)
    lmax = lambda_max(X, y)
    lambdas = np.exp(np.linspace(np.log(lmax), np.log(lmax * min_ratio), n_lambda))
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(folds.split(X, y))
    paths = Parallel(n_jobs=n_jobs)(delayed(_path)(X[tr], y[tr], lambdas, seed) for tr, _ in splits)
    dev = np.array([[_deviance(X[te], y[te], c, b) for c, b in zip(*paths[k])] for k, (_, te) in enumerate(splits)])
    cvm = dev.mean(axis=0)
    cvsd = dev.std(axis=0, ddof=1) / np.sqrt(n_folds)
    i_min = int(np.argmin(cvm))
    ok = np.where(cvm <= cvm[i_min] + cvsd[i_min])[0]
    i_1se = int(ok.min())                      # largest lambda (lambdas descending) within 1 SE
    coefs, _ = _path(X, y, lambdas, seed)
    return LassoCVResult(lambdas, cvm, cvsd, (coefs != 0).sum(axis=1), coefs,
                         float(lambdas[i_min]), float(lambdas[i_1se]), list(features))
