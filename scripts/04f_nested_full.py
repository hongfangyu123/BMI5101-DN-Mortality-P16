"""Stage 8c — full nested cross-validation (decisions.md D8.4–D8.8).

Usage:
  python scripts/04f_nested_full.py --track replication --cohort C              # full run (5 repeats x 10 folds)
  python scripts/04f_nested_full.py --track replication --cohort C --selfcheck  # LASSO-only nesting, seed 42,
                                                                               # stage-4 hyper-parameters (vs 04d)
  python scripts/04f_nested_full.py --track replication --cohort C --timing     # repeat 0 / fold 0 only, timed
  add --fallback to use the D8.5 fallback (tuned RF / tuned XGBoost on the first 3 repeats only)
  add --variant comorb_time for the stage-8d comorbidity-timing sensitivity analysis (D8.2): cci and sapsii are
  replaced by cci_sens / sapsii_sens (data/processed/features_comorb_time.parquet, scripts/02b_comorb_time.py), all
  other features unchanged; only the recommended unpenalised LR and tuned RF; separate cache and outputs
  (*_C_comorb_time.csv), plus a paired comparison with the main analysis on the same outer splits (vs_main).

  python scripts/04f_nested_full.py --track clinical                           # CLIN (stage 8e-2, D8.1-D8.4)
  add --variant no_wbc_diff | prior_only --models paper:LR,tuned:RF for the CLIN sensitivity analyses (D8.3/D8.4
  revisions of 2026-10-03; --models = the D8.8-selected model and tuned RF):
    no_wbc_diff : the 8 WBC-differential variables are subject to the 20% rule (method 1); no wbc_diff_missing
    prior_only  : the 6 comorbidities, CCI (no age score) and SAPS-II chronic item from prior admissions only
                  (data/processed/comorb_clin.parquet *_prior); has_prior_admission added as a LASSO candidate

Replication: --track replication --cohort C; input = the cohort-C training set (1,568 patients).
Clinical: --track clinical; input = the CLIN training set of split_clin (2,045 patients), candidates = the 55 predictors
(CLIN comorbidities, CCI without age score, SAPS-II with the recomputed chronic item; data/processed/features_clin.parquet)
+ wbc_diff_missing. The 8 WBC-differential variables (lymphocytes / neutrophils / monocytes, NLR, PLR, LMR, PNR, SII) are
exempt from the 20% rule (D8.4 method 2) and mean-imputed inside the fold like every other variable; all other
variables go through the in-fold 20% rule. Outputs results/clinical/nested_cv_full_*_clin[_variant].csv.
The held-out set is never loaded (parquet row filters; for CLIN only its stay_ids are read, to assert disjointness).

Inside every outer training fold (seed = the repeat's seed everywhere: outer split, LASSO inner 10 folds, tuning
inner 5 folds, model random_state, SVC Platt CV):
  20% missing filter -> imputation -> standardisation -> LASSO lambda.1se -> per model: GridSearchCV (tuned models,
  inner stratified 5-fold, AUROC, src/models.tuning_grids) or fixed paper parameters -> fit on the whole outer
  training fold -> predict the outer validation fold. Every fitting step asserts that its rows are disjoint from the
  outer validation fold. If LASSO selects no variable, every model predicts the training-fold mortality (constant).

Per-fold results are cached in data/interim/nested_cache/<track>_<cohort>[_selfcheck]/ (git-ignored; written to a
temporary file and atomically renamed; existing files are skipped, so runs can resume). Only aggregate results are
written to results/replication/ and printed.

Calibration intercept/slope: a fit that raises, does not converge, warns of perfect separation, or gives |value| > 1e3 is
recorded as NaN (src/metrics.cal_intercept_slope_checked); the summary counts such folds/repeats. RF (R defaults) gets
fold-level calibration NaN in the summary (pooled OOF only). The summary also lists each repeat's pooled-OOF calibration
slope and |ln slope| and their mean over repeats (the D8.8 rule 4 quantity).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from src.common import ROOT, load_config, path
from src.lasso import cv_lasso
from src.metrics import (auprc, cal_intercept_slope_checked, decision_curve, nadeau_bengio, scaled_brier,
                         youden_threshold)
from src.models import SCALED_MODELS, paper_models, tuned_model, tuning_grids
from src.preprocess import impute_scale, missing_filter
from src.variables import FEATURES

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ap = argparse.ArgumentParser()
ap.add_argument("--track", choices=["replication", "clinical"], required=True)
ap.add_argument("--cohort", choices=["A", "C"], default=None)
ap.add_argument("--selfcheck", action="store_true")
ap.add_argument("--timing", action="store_true")
ap.add_argument("--fallback", action="store_true")
ap.add_argument("--variant", choices=["main", "comorb_time", "no_wbc_diff", "prior_only"], default="main")
ap.add_argument("--models", default=None, help="sensitivity variants: comma-separated params:model keys, e.g. paper:LR,tuned:RF")
args = ap.parse_args()
if args.variant != "main" and args.selfcheck:
    raise SystemExit("--selfcheck applies to the main analysis only")
VARIANT = args.variant
SUF = "" if VARIANT == "main" else f"_{VARIANT}"       # output / cache suffix; main-analysis names unchanged
CLIN = args.track == "clinical"
if CLIN:
    if args.cohort is not None or args.selfcheck or VARIANT == "comorb_time":
        raise SystemExit("--track clinical takes no --cohort / --selfcheck; variants: no_wbc_diff, prior_only")
    COH = "clin"
else:
    if args.cohort != "C" or VARIANT in ("no_wbc_diff", "prior_only"):
        raise SystemExit("only --track replication --cohort C (variants: comorb_time) is implemented")
    COH = args.cohort

cfg = load_config()
NC = cfg["nested_cv"]
SEEDS = [42] if args.selfcheck else list(NC["seeds"])
K, K_LASSO, K_TUNE = NC["outer_folds"], NC["inner_lasso_folds"], NC["inner_tune_folds"]
N_JOBS = int(cfg["duckdb"]["threads"])
P = path("processed")
TAG = f"{args.track}_{COH}"
CACHE = path("interim") / "nested_cache" / (TAG + ("_selfcheck" if args.selfcheck else "") + SUF)
OUT = ROOT / "results" / args.track
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# model keys: (params group, model name); the paper-parameter LR is the recommended unpenalised LR
MODELS = [("paper", m) for m in ["RF", "RF (R defaults)", "XGBoost", "SVM", "LR"]] + \
         [("tuned", m) for m in ["RF", "XGBoost", "SVM", "LR"]]
if VARIANT == "comorb_time":                          # D8.2: recommended LR and tuned RF only
    MODELS = [("paper", "LR"), ("tuned", "RF")]
if VARIANT in ("no_wbc_diff", "prior_only"):          # D8.3 / D8.4 revisions: D8.8-selected model + tuned RF
    if not args.models:
        raise SystemExit("--models is required for the CLIN sensitivity variants (the D8.8-selected model and tuned RF)")
    keys = [tuple(k.split(":", 1)) for k in args.models.split(",")]
    assert all(k in MODELS for k in keys), f"unknown model key in {keys}"
    MODELS = [k for k in MODELS if k in keys]
LABEL = {("paper", "LR"): "LR (unpenalised)", ("tuned", "LR"): "LR (L2)"}
def label(key): return LABEL.get(key, f"{key[1]} ({key[0]})")
def slug(key): return f"{key[0]}_{key[1].replace(' ', '').replace('(', '').replace(')', '')}"
REPEATS = {key: list(range(len(SEEDS))) for key in MODELS}
if args.fallback:                                    # D8.5 fallback: tuned RF / XGBoost on seeds 42, 43, 44 only
    for key in [("tuned", "RF"), ("tuned", "XGBoost")]:
        REPEATS[key] = [0, 1, 2]

# ---------------------------------------------------------------- data: training set only, holdout never loaded
WBCD = ["lymphocytes_mean", "neutrophils_mean", "monocytes_mean", "nlr", "plr", "lmr", "pnr", "sii"]
if CLIN:
    hold_ids = pd.read_parquet(P / "split_clin.parquet", columns=["stay_id"], filters=[("split", "==", "valid")]).stay_id
    split = pd.read_parquet(P / "split_clin.parquet", filters=[("split", "==", "train")])
    assert (split.split == "train").all() and len(split) == 2045, "expected the 2,045-patient CLIN training set"
    feat = pd.read_parquet(P / "features_clin.parquet", filters=[("stay_id", "in", split.stay_id.tolist())])
    assert set(feat.stay_id).isdisjoint(set(hold_ids)) and len(hold_ids) == 512, "held-out CLIN rows were loaded"
    CAND = {"main": FEATURES + ["wbc_diff_missing"], "no_wbc_diff": list(FEATURES),
            "prior_only": FEATURES + ["wbc_diff_missing", "has_prior_admission"]}[VARIANT]
    EXEMPT = [] if VARIANT == "no_wbc_diff" else WBCD          # D8.4 method 2: not subject to the 20% rule
else:
    split = pd.read_parquet(P / "split.parquet", filters=[("split", "==", "train")])
    assert (split.split == "train").all() and len(split) == 1568, "expected the 1,568-patient cohort-C training set"
    feat = pd.read_parquet(P / "features.parquet", filters=[("stay_id", "in", split.stay_id.tolist())])
    CAND, EXEMPT = list(FEATURES), []
assert len(feat) == len(split) and (feat.stay_id.values == split.stay_id.values).all(), "row order differs from split"
assert (feat.y.values == split.y.values).all()
if VARIANT == "prior_only":                           # D8.3 sensitivity: prior-admission comorbidities only
    pr = pd.read_parquet(P / "comorb_clin.parquet")
    PRIOR = ["myocardial_infarct", "congestive_heart_failure", "chronic_pulmonary_disease", "renal_disease",
             "hypertension", "obesity", "cci", "sapsii"]
    feat = feat.drop(columns=PRIOR + ["has_prior_admission"]).merge(
        pr[["stay_id", "has_prior_admission"] + [f"{c}_prior" for c in PRIOR]], on="stay_id", how="left",
        validate="one_to_one").rename(columns={f"{c}_prior": c for c in PRIOR})
    assert (feat.stay_id.values == split.stay_id.values).all()
if VARIANT == "comorb_time":                          # replace cci / sapsii only (D8.2)
    sens = pd.read_parquet(P / "features_comorb_time.parquet")
    feat = feat.merge(sens, on="stay_id", how="left", validate="one_to_one")
    assert feat[["cci_sens", "sapsii_sens"]].notna().sum().tolist() == feat[["cci", "sapsii"]].notna().sum().tolist()
    feat["cci"], feat["sapsii"] = feat.pop("cci_sens"), feat.pop("sapsii_sens")
X_raw = feat[CAND].reset_index(drop=True)
y = feat.y.values.astype(int)

# ---------------------------------------------------------------- cache identity
def _src(p): return (ROOT / p).read_text()
meta = {"mode": "selfcheck" if args.selfcheck else "full", "seeds": SEEDS, "K": K, "K_lasso": K_LASSO, "K_tune": K_TUNE,
        "features": CAND, "train_ids_sha": hashlib.sha256(np.sort(split.stay_id.values).tobytes()).hexdigest(),
        "code_sha": hashlib.sha256("".join(_src(p) for p in ["src/models.py", "src/lasso.py", "src/preprocess.py"])
                                   .encode()).hexdigest(),
        "stage4_params": json.load(open(path("tables") / "tuned_hyperparameters.json")) if args.selfcheck else None}
if VARIANT != "main":                                 # main-analysis meta unchanged so its cache stays valid
    meta["variant"] = VARIANT
    meta["models"] = [list(k) for k in MODELS]
    meta["data_sha"] = hashlib.sha256(np.ascontiguousarray(X_raw[["cci", "sapsii"]].values)).hexdigest()
if CLIN:                                              # replication meta unchanged so its caches stay valid
    meta["track"] = "clinical"
    meta["exempt_missing_filter"] = EXEMPT
    meta["data_sha"] = hashlib.sha256(np.ascontiguousarray(X_raw.values.astype(float))).hexdigest()
meta_file = CACHE / "meta.json"
if meta_file.exists():
    old = json.loads(meta_file.read_text())
    if old != json.loads(json.dumps(meta)):
        raise SystemExit(f"cache {CACHE} was built with different settings/code/data; refusing to reuse it")
else:
    meta_file.write_text(json.dumps(meta, indent=1))


def save_atomic(obj, f: Path):
    tmp = f.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, f)


def disjoint(fit_rows, val_rows, step):
    assert len(np.intersect1d(fit_rows, val_rows)) == 0, f"{step}: fit rows overlap the outer validation fold"


# ---------------------------------------------------------------- one outer fold
def fold_lasso(rep, k, itr, ite):
    f = CACHE / f"r{rep}_f{k}_lasso.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    seed = SEEDS[rep]
    t0 = time.time()
    disjoint(itr, ite, "missing filter")
    passed = set(missing_filter(X_raw, itr))           # in-fold 20% rule; EXEMPT (CLIN WBC differential) kept always
    kept = [c for c in CAND if c in passed or c in EXEMPT]
    disjoint(itr, ite, "imputation/standardisation"); Xi, Xs = impute_scale(X_raw[kept], itr)
    disjoint(itr, ite, "LASSO")
    res = cv_lasso(Xs.loc[itr].values, y[itr], kept, n_folds=K_LASSO, seed=seed, n_jobs=N_JOBS)
    out = {"kept": kept, "n_dropped_missing": len(CAND) - len(kept), "selected": res.selected("1se"),
           "lambda_1se": res.lambda_1se, "seconds": time.time() - t0, "test_idx": ite}
    save_atomic(out, f)
    return out


def fold_model(rep, k, itr, ite, las, key):
    f = CACHE / f"r{rep}_f{k}_{slug(key)}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    seed = SEEDS[rep]
    sel, (grp, m) = las["selected"], key
    t0, params = time.time(), None
    if not sel:                                       # D8.5: no variable selected -> constant prediction
        pred, constant = np.full(len(ite), y[itr].mean()), True
    else:
        disjoint(itr, ite, "imputation/standardisation (model)")
        Xi, Xs = impute_scale(X_raw[las["kept"]], itr)
        Xm = Xs if m in SCALED_MODELS else Xi
        a, b = Xm.loc[itr, sel], Xm.loc[ite, sel]
        disjoint(itr, ite, f"fit {label(key)}")
        if grp == "paper":
            est = paper_models(seed)[m].fit(a, y[itr])
        elif args.selfcheck:                          # LASSO-only nesting: stage-4 hyper-parameters, no inner tuning
            est = tuned_model(m, meta["stage4_params"][f"noleak/{m}"], len(sel), seed).fit(a, y[itr])
        else:
            base, grid = tuning_grids(len(sel), seed)[m]
            gs = GridSearchCV(clone(base), grid, cv=StratifiedKFold(K_TUNE, shuffle=True, random_state=seed),
                              scoring="roc_auc", n_jobs=N_JOBS, refit=True).fit(a, y[itr])
            params = {kk: (vv if not isinstance(vv, float) or np.isfinite(vv) else "inf") for kk, vv in gs.best_params_.items()}
            est = (SVC(kernel="rbf", probability=True, random_state=seed, **gs.best_params_).fit(a, y[itr])
                   if m == "SVM" else gs.best_estimator_)      # SVC Platt CV seeded with the repeat's seed
        pred, constant = est.predict_proba(b)[:, 1], False
    out = {"test_idx": ite, "pred": pred, "constant": constant, "params": params, "seconds": time.time() - t0}
    save_atomic(out, f)
    return out


def fold_metrics(yt, p, constant):
    auc = 0.5 if constant else roc_auc_score(yt, p)
    cal = cal_intercept_slope_checked(yt, p)          # non-converged / |value| > 1e3 -> NaN
    if constant:
        cal["cal_slope"], cal["cal_slope_ok"] = np.nan, True   # slope not estimable for a constant prediction
    return {"auroc": auc, "auprc": auprc(yt, p), "prevalence": float(yt.mean()), "brier": brier_score_loss(yt, p),
            "scaled_brier": scaled_brier(yt, p), **cal}


# ---------------------------------------------------------------- run
reps = [0] if args.timing else list(range(len(SEEDS)))
fold_rows, cache_by = [], {}
for rep in reps:
    splits = list(StratifiedKFold(K, shuffle=True, random_state=SEEDS[rep]).split(X_raw, y))
    for k, (itr, ite) in enumerate(splits):
        if args.timing and k > 0:
            break
        las = fold_lasso(rep, k, itr, ite)
        for key in MODELS:
            if rep not in REPEATS[key]:
                continue
            res = fold_model(rep, k, itr, ite, las, key)
            assert np.array_equal(res["test_idx"], ite)
            cache_by[(key, rep, k)] = res
            fold_rows.append({"repeat": rep, "seed": SEEDS[rep], "fold": k, "params": key[0], "model": label(key),
                              "model_key": key[1], "n_train": len(itr), "n_test": len(ite), "n_selected": len(las["selected"]),
                              "n_dropped_missing": las["n_dropped_missing"], "lambda_1se": las["lambda_1se"],
                              "constant_prediction": res["constant"], "seconds": res["seconds"],
                              "lasso_seconds": las["seconds"], **fold_metrics(y[ite], res["pred"], res["constant"])})
            print(f"rep {rep} fold {k} {label(key):22s} AUROC {fold_rows[-1]['auroc']:.3f}  ({res['seconds']:.1f}s)",
                  flush=True)
folds = pd.DataFrame(fold_rows)

# ---------------------------------------------------------------- self-check: LASSO-only nesting vs 04d
if args.selfcheck:
    ref = pd.read_csv(path("tables") / "nested_cv_auc.csv").set_index(["params", "model"]).cv_auc_nested
    m = folds.groupby(["params", "model", "model_key"], sort=False).auroc.mean().reset_index()
    m["nested_cv_auc_04d"] = [ref.loc[(p, km)] for p, km in zip(m.params, m.model_key)]
    m["abs_diff"] = (m.auroc - m.nested_cv_auc_04d).abs()
    m.drop(columns="model_key").round(6).to_csv(OUT / "nested_selfcheck_C.csv", index=False)
    print(m.drop(columns="model_key").round(4).to_string(index=False))
    print("SELF-CHECK", "PASSED" if (m.abs_diff < 1e-3).all() else "FAILED", f"(max |diff| {m.abs_diff.max():.2e})")
    sys.exit(0)

# ---------------------------------------------------------------- timing: repeat 0 / fold 0, extrapolated
if args.timing:
    t = folds[["params", "model", "seconds"]].copy()
    t["folds_planned"] = [K * len(REPEATS[k]) for k in MODELS]
    t["extrapolated_hours"] = t.seconds * t.folds_planned / 3600
    lasso_h = folds.lasso_seconds.iloc[0] * K * len(SEEDS) / 3600
    t = pd.concat([t, pd.DataFrame([{"params": "-", "model": "LASSO (per fold, shared)", "seconds": folds.lasso_seconds.iloc[0],
                                     "folds_planned": K * len(SEEDS), "extrapolated_hours": lasso_h}])], ignore_index=True)
    t.round(3).to_csv(OUT / f"nested_timing_{COH}{SUF}.csv", index=False)
    print(t.round(2).to_string(index=False))
    print(f"TOTAL extrapolated: {t.extrapolated_hours.sum():.1f} h")
    las0 = pickle.load(open(CACHE / "r0_f0_lasso.pkl", "rb"))
    print(f"fold 0: {len(CAND)} candidates, {las0['n_dropped_missing']} dropped by the 20% rule "
          f"({[c for c in CAND if c not in las0['kept']]}), LASSO selected {len(las0['selected'])}: {las0['selected']}")
    sys.exit(0)

# ---------------------------------------------------------------- full run: aggregate outputs
folds.drop(columns=["seconds", "lasso_seconds", "model_key"]).round(6).to_csv(OUT / f"nested_cv_full_folds_{COH}{SUF}.csv", index=False)
MET = ["auroc", "auprc", "prevalence", "brier", "scaled_brier", "cal_intercept", "cal_slope"]
FOLD_CAL_POOLED_ONLY = {("paper", "RF (R defaults)")}  # fold-level calibration not reported (diverges); pooled OOF only

pooled_rows, dca_rows, thr_rows = [], [], []
for key in MODELS:
    for rep in REPEATS[key]:
        idx = np.concatenate([cache_by[(key, rep, k)]["test_idx"] for k in range(K)])
        pr = np.concatenate([cache_by[(key, rep, k)]["pred"] for k in range(K)])
        assert len(np.unique(idx)) == len(y)          # every training patient predicted exactly once per repeat
        yt = y[idx]
        const = all(cache_by[(key, rep, k)]["constant"] for k in range(K))
        pooled_rows.append({"params": key[0], "model": label(key), "repeat": rep, **fold_metrics(yt, pr, const)})
        d = decision_curve(yt, pr, thresholds=np.round(np.arange(0.05, 0.6001, 0.01), 2))
        dca_rows.append(d.assign(params=key[0], model=label(key), repeat=rep))
        thr_rows.append({"params": key[0], "model": label(key), "repeat": rep, "youden_threshold": youden_threshold(yt, pr)[0]})
pooled = pd.DataFrame(pooled_rows)

summ = []
for key in MODELS:
    f = folds[(folds.params == key[0]) & (folds.model == label(key))]
    pp = pooled[(pooled.params == key[0]) & (pooled.model == label(key))]
    row = {"params": key[0], "model": label(key), "n_repeats": len(REPEATS[key]), "n_folds": len(f),
           "n_constant_folds": int(f.constant_prediction.sum()), "n_nan_cal_slope_folds": int(f.cal_slope.isna().sum()),
           "n_nonconv_cal_intercept_folds": int((~f.cal_intercept_ok).sum()),
           "n_nonconv_cal_slope_folds": int((~f.cal_slope_ok).sum()),
           "n_nonconv_cal_pooled_repeats": int((~(pp.cal_intercept_ok & pp.cal_slope_ok)).sum()),
           "cal_fold_level_reported": key not in FOLD_CAL_POOLED_ONLY}
    for c in MET:
        row[f"{c}_fold_mean"] = np.nanmean(f[c]); row[f"{c}_fold_sd"] = np.nanstd(f[c], ddof=1)
        row[f"{c}_pooled_mean"] = np.nanmean(pp[c]); row[f"{c}_pooled_sd_over_repeats"] = np.nanstd(pp[c], ddof=1)
    if key in FOLD_CAL_POOLED_ONLY:
        for c in ["cal_intercept", "cal_slope"]:
            row[f"{c}_fold_mean"] = row[f"{c}_fold_sd"] = np.nan
    for rep in range(len(SEEDS)):                     # D8.8 rule 4: per-repeat pooled-OOF slope and |ln slope|
        v = pp.loc[pp.repeat == rep, "cal_slope"]
        sl = float(v.iloc[0]) if len(v) else np.nan
        row[f"cal_slope_pooled_rep{SEEDS[rep]}"] = sl
        row[f"abs_ln_cal_slope_pooled_rep{SEEDS[rep]}"] = abs(np.log(sl)) if sl > 0 else np.nan
    row["abs_ln_cal_slope_pooled_mean_over_repeats"] = np.nanmean(
        [row[f"abs_ln_cal_slope_pooled_rep{SEEDS[r]}"] for r in REPEATS[key]])
    summ.append(row)
pd.DataFrame(summ).round(6).to_csv(OUT / f"nested_cv_full_summary_{COH}{SUF}.csv", index=False)

dca = pd.concat(dca_rows).groupby(["params", "model", "threshold"], sort=False)[
    ["net_benefit_model", "net_benefit_all", "net_benefit_none"]].mean().reset_index()
dca.round(6).to_csv(OUT / f"nested_cv_full_dca_{COH}{SUF}.csv", index=False)

pw = []
for i, a in enumerate(MODELS):
    for b in MODELS[i + 1:]:
        common = sorted(set(REPEATS[a]) & set(REPEATS[b]))
        fa = folds[(folds.params == a[0]) & (folds.model == label(a)) & folds.repeat.isin(common)].sort_values(["repeat", "fold"])
        fb = folds[(folds.params == b[0]) & (folds.model == label(b)) & folds.repeat.isin(common)].sort_values(["repeat", "fold"])
        assert (fa[["repeat", "fold"]].values == fb[["repeat", "fold"]].values).all()
        nb = nadeau_bengio(fa.auroc.values - fb.auroc.values, fa.n_train.mean(), fa.n_test.mean(), K, len(common))
        pw.append({"model_1": label(a), "model_2": label(b), "auroc_1": fa.auroc.mean(), "auroc_2": fb.auroc.mean(),
                   "common_repeats": ",".join(str(SEEDS[r]) for r in common), **nb})
pd.DataFrame(pw).round(6).to_csv(OUT / f"nested_cv_full_pairwise_{COH}{SUF}.csv", index=False)

sel_counts = pd.Series(0, index=CAND)
n_lasso_folds = 0
for rep in range(len(SEEDS)):
    for k in range(K):
        sel_counts[pickle.load(open(CACHE / f"r{rep}_f{k}_lasso.pkl", "rb"))["selected"]] += 1
        n_lasso_folds += 1
pd.DataFrame({"feature": CAND, "n_folds_selected": sel_counts.values, "n_folds": n_lasso_folds}) \
    .sort_values("n_folds_selected", ascending=False).to_csv(OUT / f"nested_cv_full_selection_{COH}{SUF}.csv", index=False)

prm = []
for key in [k for k in MODELS if k[0] == "tuned"]:
    vals = [json.dumps(cache_by[(key, rep, k)]["params"], sort_keys=True) for rep in REPEATS[key] for k in range(K)
            if cache_by[(key, rep, k)]["params"] is not None]
    for v, n in pd.Series(vals).value_counts().items():
        prm.append({"model": label(key), "params": v, "n_folds": int(n)})
pd.DataFrame(prm).to_csv(OUT / f"nested_cv_full_params_{COH}{SUF}.csv", index=False)

thr = pd.DataFrame(thr_rows)
thr_out = thr.groupby(["params", "model"], sort=False).youden_threshold.agg(
    median="median", min="min", max="max", n_repeats="count").reset_index()
thr_out.round(4).to_csv(OUT / f"nested_cv_full_oof_threshold_{COH}{SUF}.csv", index=False)

if VARIANT != "main":                                 # paired comparison with the main analysis, same outer splits
    main = pd.read_csv(OUT / f"nested_cv_full_folds_{COH}.csv")
    vm = []
    for key in MODELS:
        common = sorted(set(REPEATS[key]) & set(main[main.model == label(key)].repeat))
        fv = folds[(folds.model == label(key)) & folds.repeat.isin(common)].sort_values(["repeat", "fold"])
        fm = main[(main.model == label(key)) & main.repeat.isin(common)].sort_values(["repeat", "fold"])
        assert (fv[["repeat", "fold", "n_test"]].values == fm[["repeat", "fold", "n_test"]].values).all()
        nb = nadeau_bengio(fv.auroc.values - fm.auroc.values, fv.n_train.mean(), fv.n_test.mean(), K, len(common))
        vm.append({"model": label(key), "auroc_variant": fv.auroc.mean(), "auroc_main": fm.auroc.mean(),
                   "common_repeats": ",".join(str(SEEDS[r]) for r in common), **nb})
    vm = pd.DataFrame(vm)
    vm.round(6).to_csv(OUT / f"nested_cv_full_vs_main_{COH}{SUF}.csv", index=False)

pd.set_option("display.width", 220)
print(pd.DataFrame(summ)[["params", "model", "n_repeats", "n_constant_folds", "n_nonconv_cal_intercept_folds",
                          "n_nonconv_cal_slope_folds", "n_nonconv_cal_pooled_repeats", "auroc_fold_mean", "auroc_fold_sd",
                          "auroc_pooled_mean", "cal_slope_pooled_mean", "abs_ln_cal_slope_pooled_mean_over_repeats"]]
      .round(4).to_string(index=False))
print(pd.DataFrame(summ)[["model"] + [c for c in summ[0] if c.startswith(("cal_slope_pooled_rep", "abs_ln_cal_slope_pooled_rep"))]]
      .round(4).to_string(index=False))
