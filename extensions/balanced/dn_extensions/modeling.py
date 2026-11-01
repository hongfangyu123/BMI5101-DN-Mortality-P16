"""Simplified computation: fixed parameters, within-fold representation learning and cross-fitted stacking."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .settings import BURDEN, DOCUMENT, MODE_VARS, STATIC, WBC_EXEMPT
from src.lasso import _deviance, _path, lambda_max

EXPERIMENTS = {
    'static_lr': ('static', 'LR'),
    'static_rf': ('static', 'RF'),
    'static_xgb': ('static', 'XGBoost'),
    'static_stacking': ('static', 'Stacking'),
    'temporal_lr': ('temporal', 'LR'),
    'burden_lr': ('burden', 'LR'),
    'embedding_lr': ('embedding', 'LR'),
    'combined_stacking': ('combined', 'Stacking'),
    # The same feature-construction rules are used for different learners, so model effects are not confounded with feature effects.
    'temporal_rf': ('temporal', 'RF'),
    'temporal_xgb': ('temporal', 'XGBoost'),
    'burden_rf': ('burden', 'RF'),
    'burden_xgb': ('burden', 'XGBoost'),
    'embedding_rf': ('embedding', 'RF'),
    'embedding_xgb': ('embedding', 'XGBoost'),
}


def fit_numeric(frame, threshold):
    """Compute missing rates, means/modes and standard deviations only on the training rows passed in."""
    frame = frame.replace([np.inf, -np.inf], np.nan).astype(float)
    keep = [c for c in frame if frame[c].notna().any() and
            (frame[c].isna().mean() <= threshold or c in WBC_EXEMPT)]
    fill = pd.Series({c: frame[c].mode().iloc[0] if c in MODE_VARS else frame[c].mean() for c in keep})
    xi = frame[keep].fillna(fill)
    mu, sd = xi.mean(), xi.std(ddof=0)
    keep = [c for c in keep if np.isfinite(sd[c]) and sd[c] > 1e-12]
    return keep, fill[keep], mu[keep], sd[keep]


def scaled(frame, keep, fill, mu, sd):
    frame = frame.replace([np.inf, -np.inf], np.nan).astype(float)
    return ((frame[keep].fillna(fill) - mu) / sd).to_numpy(float)


def choose_lasso(raw, y, options, seed):
    """Reuse the V3 glmnet-style path and λ.1se; preprocessing is redone even inside the LASSO inner folds."""
    keep, fill, mu, sd = fit_numeric(raw, options['missing_threshold'])
    if not keep:
        return [], (keep, fill, mu, sd), None
    xs = scaled(raw, keep, fill, mu, sd)
    lmax = lambda_max(xs, y)
    if not np.isfinite(lmax) or lmax <= 1e-12:
        return [], (keep, fill, mu, sd), None
    k = min(options['inner_lasso_folds'], int(np.bincount(y).min()))
    if k < 2:
        raise ValueError('the LASSO training set needs at least 2 cases per class')
    lambdas = np.geomspace(lmax, lmax * 1e-4, options['lasso_lambdas'])
    deviations = []
    for tr, va in StratifiedKFold(k, shuffle=True, random_state=seed).split(raw, y):
        ik, iff, im, iss = fit_numeric(raw.iloc[tr], options['missing_threshold'])
        if ik:
            inner_x = scaled(raw.iloc[tr], ik, iff, im, iss)
            valid_x = scaled(raw.iloc[va], ik, iff, im, iss)
            coefficients, intercepts = _path(inner_x, y[tr], lambdas, seed)
            dev = [_deviance(valid_x, y[va], c, b) for c, b in zip(coefficients, intercepts)]
        else:
            p = np.clip(y[tr].mean(), 1e-12, 1 - 1e-12)
            dev = [-2 * np.mean(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p))] * len(lambdas)
        deviations.append(dev)
    deviations = np.asarray(deviations)
    means, se = deviations.mean(axis=0), deviations.std(axis=0, ddof=1) / np.sqrt(k)
    imin = int(np.argmin(means))
    i1se = int(np.flatnonzero(means <= means[imin] + se[imin])[0])
    coefficients, _ = _path(xs, y, lambdas, seed)
    selected = [c for c, coef in zip(keep, coefficients[i1se]) if coef != 0]
    return selected, (keep, fill, mu, sd), float(lambdas[i1se])


class FoldRepresentation(TransformerMixin, BaseEstimator):
    def __init__(self, variant='static', options=None, seed=42, scale_output=True):
        self.variant = variant
        self.options = options
        self.seed = seed
        self.scale_output = scale_output

    def fit(self, X, y):
        options = self.options
        numeric = list(STATIC)
        if self.variant in ('temporal', 'combined'):
            numeric += [c for c in X if c.startswith('trajectory_')]
        self.numeric_ = numeric
        self.selected_, values, self.lambda_1se_ = choose_lasso(X[numeric], np.asarray(y, int), options, self.seed)
        self.kept_, self.fill_, self.mu_, self.sd_ = values
        self.vectorizer_, self.svd_, self.embedding_width_ = None, None, 0
        self.use_burden_ = self.variant in ('burden', 'embedding', 'combined')
        self.use_embedding_ = self.variant in ('embedding', 'combined')
        if self.use_burden_:
            self.burden_scaler_ = StandardScaler().fit(self._burden(X))
        if self.use_embedding_:
            docs = X[DOCUMENT].fillna('').astype(str)
            vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b',
                                         min_df=options['embedding_min_df'], sublinear_tf=True)
            try:
                tfidf = vectorizer.fit_transform(docs)
            except ValueError as e:
                if not any(s in str(e).lower() for s in ['empty vocabulary', 'no terms remain', 'max_df corresponds']):
                    raise
            else:
                self.vectorizer_ = vectorizer
                if tfidf.shape[1] >= 2 and tfidf.shape[0] >= 2:
                    width = min(options['embedding_components'], tfidf.shape[1] - 1, tfidf.shape[0] - 1)
                    self.svd_ = TruncatedSVD(n_components=width, n_iter=7, random_state=self.seed)
                    embedded = self.svd_.fit_transform(tfidf)
                else:
                    # With only one term, keep its one-dimensional TF-IDF to avoid an SVD error.
                    embedded = tfidf.toarray()
                self.embedding_width_ = embedded.shape[1]
                self.embedding_scaler_ = StandardScaler().fit(embedded)
        names = self.selected_ + (BURDEN if self.use_burden_ else [])
        names += [f'icd_embedding_{i:02d}' for i in range(self.embedding_width_)]
        self.output_names_ = names or ['constant_no_selected_feature']
        self.n_features_in_ = X.shape[1]
        return self

    @staticmethod
    def _burden(X):
        a = X[BURDEN].fillna(0).to_numpy(float).copy()
        a[:, :3] = np.log1p(a[:, :3])
        return a

    def transform(self, X):
        pieces = []
        if self.selected_:
            raw = X[self.selected_].replace([np.inf, -np.inf], np.nan).astype(float).fillna(self.fill_)
            if self.scale_output:
                raw = (raw - self.mu_[self.selected_]) / self.sd_[self.selected_]
            pieces.append(raw.to_numpy(float))
        if self.use_burden_:
            a = self._burden(X)
            pieces.append(self.burden_scaler_.transform(a) if self.scale_output else a)
        if self.embedding_width_:
            tfidf = self.vectorizer_.transform(X[DOCUMENT].fillna('').astype(str))
            a = self.svd_.transform(tfidf) if self.svd_ is not None else tfidf.toarray()
            pieces.append(self.embedding_scaler_.transform(a) if self.scale_output else a)
        return np.column_stack(pieces) if pieces else np.zeros((len(X), 1))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.output_names_, object)


def base_pipeline(name, variant, options, seed, memory):
    parameters = dict(options['fixed_parameters'][name])
    if options['profile'] == 'smoke':
        if name in ('RF', 'XGBoost'):
            parameters['n_estimators'] = 8
    if name == 'LR':
        estimator = LogisticRegression(l1_ratio=0.0, max_iter=10000, random_state=seed, **parameters)
    elif name == 'RF':
        estimator = RandomForestClassifier(random_state=seed, n_jobs=options['jobs'], **parameters)
    elif name == 'XGBoost':
        estimator = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                  tree_method='hist', random_state=seed, n_jobs=options['jobs'], **parameters)
    else:
        raise ValueError(name)
    # Representation learning is identical, so LR/RF/XGB can share the LASSO/embedding cache for the same training rows.
    # Scaling for LR is moved to a separate lightweight step; tree models still use the imputed original scale.
    steps = [('representation', FoldRepresentation(variant, options, seed, False))]
    if name == 'LR':
        steps.append(('scale', StandardScaler()))
    steps.append(('model', estimator))
    return Pipeline(steps, memory=Memory(str(memory), verbose=0))



def make_estimator(experiment, options, seed, memory):
    variant, name = EXPERIMENTS[experiment]
    if name != 'Stacking':
        return base_pipeline(name, variant, options, seed, memory)
    bases = [(name, base_pipeline(name, variant, options, seed, memory))
             for name in ['LR', 'RF', 'XGBoost']]
    return StackingClassifier(
        estimators=bases,
        final_estimator=LogisticRegression(C=1.0, max_iter=10000, random_state=seed),
        cv=StratifiedKFold(options['stack_folds'], shuffle=True, random_state=seed),
        stack_method='predict_proba', passthrough=False, n_jobs=1,
    )


def fitted_metadata(estimator):
    if isinstance(estimator, Pipeline):
        representation = estimator.named_steps['representation']
        model = estimator.named_steps['model']
        keys = ['C', 'n_estimators', 'max_depth', 'min_samples_leaf', 'max_features',
                'learning_rate', 'subsample', 'colsample_bytree']
        actual = model.get_params(deep=False)
        return {'selected_numeric': representation.selected_,
                'lambda_1se': representation.lambda_1se_,
                'embedding_width': representation.embedding_width_,
                'kept_numeric': representation.kept_,
                'fixed_params': {key: actual[key] for key in keys if key in actual},
                'hyperparameter_tuning': False}
    return {'base_models': {name: fitted_metadata(estimator.named_estimators_[name])
                            for name in ['LR', 'RF', 'XGBoost']}}
