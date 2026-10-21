"""Stage 8e-4 step 2 — CLIN held-out set: confirmatory evaluation of the frozen model, run ONCE (decisions.md D8.6,
D8.7, D8.15).

Guards (checked before the held-out set is read):
  1. results/clinical/frozen_model_clin.json and the fit record final_model_clin.json are tracked by git and have no
     staged or unstaged changes relative to HEAD;
  2. the model bundle (data/interim/final_clin/final_clin.pkl, from 04h) matches the frozen file and the fit record:
     file sha256, spec sha256, training stay_id hash, src/ code hashes, selected variables, hyper-parameters and the
     estimator configuration; refitting LR (L2) from the bundle's preprocessing reproduces its coefficients;
  3. none of the outputs exists (results/clinical/holdout_clin.csv etc.) — no overwriting.

Evaluation on the 512 held-out patients, LR (L2) (recommended, D8.14) and tuned RF (comparator): AUROC (DeLong 95% CI);
paired DeLong LR − RF; AUPRC (stratified bootstrap, 2,000 resamples, seed 42, 95% percentile CI) with the prevalence
baseline; Brier, scaled Brier; calibration intercept / slope; threshold metrics at the frozen thresholds (nested-CV
pooled-OOF Youden medians); decision curve 0.05–0.60.
Interpretation (as D4.16): LR (L2) penalised OR per SD; unpenalised LR on the same variables (statsmodels GLM fitted on
the training set) OR, Wald 95% CI, p value, VIF; SHAP of LR (L2) on the held-out set (LinearExplainer, log-odds scale,
training set as background), ranked by mean |SHAP|.
Outputs (aggregate only): results/clinical/holdout_clin.csv, holdout_dca_clin.csv, coefs_clin.csv, shap_clin.csv.

--smoke --out DIR: same code path, but the "evaluation set" is a stratified 512-patient subsample of the TRAINING set
(the held-out set is not read) and outputs go to DIR; guard 3 is skipped and no metric is printed. Used only to test
the code before the single real run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.common import ROOT, path
from src.metrics import (auprc, cal_intercept_slope_checked, decision_curve, delong_auc_ci, delong_roc_test,
                         scaled_brier, threshold_metrics)
from src.models import tuning_grids
from src.variables import FEATURES

warnings.filterwarnings("ignore", category=FutureWarning)

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--out", default=None)
args = ap.parse_args()

CL = ROOT / "results" / "clinical"
SPEC_FILE, REC_FILE = CL / "frozen_model_clin.json", CL / "final_model_clin.json"
if args.smoke:
    if not args.out:
        raise SystemExit("--smoke requires --out DIR")
    OUT = Path(args.out).resolve()
    if OUT == CL.resolve():
        raise SystemExit("--smoke must not write to results/clinical")
    OUT.mkdir(parents=True, exist_ok=True)
else:
    if args.out:
        raise SystemExit("--out is only for --smoke")
    OUT = CL
OUTPUTS = {k: OUT / f for k, f in [("res", "holdout_clin.csv"), ("dca", "holdout_dca_clin.csv"),
                                    ("coef", "coefs_clin.csv"), ("shap", "shap_clin.csv")]}

# ---------------------------------------------------------------- guard 3: never overwrite (checked first)
if not args.smoke:
    existing = [str(f.relative_to(ROOT)) for f in OUTPUTS.values() if f.exists()]
    if existing:
        raise SystemExit(f"refusing to run: output(s) already exist {existing}; the held-out evaluation is run once only")

# ---------------------------------------------------------------- guard 1: frozen files committed and unchanged
def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)

for f in (SPEC_FILE, REC_FILE):
    rel = str(f.relative_to(ROOT))
    if git("ls-files", "--error-unmatch", rel).returncode != 0 or git("cat-file", "-e", f"HEAD:{rel}").returncode != 0:
        raise SystemExit(f"refusing to run: {rel} is not committed")
    if git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"refusing to run: {rel} differs from HEAD (staged or unstaged changes)")

# ---------------------------------------------------------------- guard 2: model bundle matches the frozen spec
spec, rec = json.loads(SPEC_FILE.read_text()), json.loads(REC_FILE.read_text())
spec_sha = hashlib.sha256(SPEC_FILE.read_bytes()).hexdigest()
BUNDLE = ROOT / rec["bundle"]
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == rec["bundle_sha256"], "model file differs from the fit record"
b = pickle.load(open(BUNDLE, "rb"))
assert b["spec_sha256"] == spec_sha == rec["spec_sha256"], "model was not fitted from the committed frozen spec"
assert b["train_stay_id_sha256"] == spec["training_set"]["stay_id_sha256"]
for f, h in spec["code_sha256"].items():
    assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h, f"{f} changed since the model was frozen"
SEED = spec["seed"]
CAND = FEATURES + ["wbc_diff_missing"]
sel = b["lasso"]["selected"]
assert b["candidates"] == CAND and sel == rec["selected"] and b["params"] == rec["params"]
assert set(sel) <= set(b["kept"]) and all(c in b["kept"] for c in spec["candidates"]["missing_filter_exempt"])
lr, rf = b["models"]["LR (L2)"], b["models"]["RF (tuned)"]
lr_base, lr_grid = tuning_grids(len(sel), SEED)["LR"]
rf_base, rf_grid = tuning_grids(len(sel), SEED)["RF"]
C = np.inf if rec["params"]["LR (L2)"]["C"] == "inf" else rec["params"]["LR (L2)"]["C"]
assert isinstance(lr, LogisticRegression) and lr.C == C and C in lr_grid["C"]
assert {k: v for k, v in lr.get_params().items() if k != "C"} == {k: v for k, v in lr_base.get_params().items() if k != "C"}
assert isinstance(rf, RandomForestClassifier) and rf.n_estimators == 500 and rf.random_state == SEED
for k, v in rec["params"]["RF (tuned)"].items():
    assert rf.get_params()[k] == v and v in rf_grid[k]
assert list(lr.feature_names_in_) == sel and list(rf.feature_names_in_) == sel
THR = {m: float(spec["thresholds"][m]) for m in ("LR (L2)", "RF (tuned)")}

# ---------------------------------------------------------------- data
P = path("processed")
split = pd.read_parquet(P / "split_clin.parquet")
tr_ids = split.loc[split.split == "train", "stay_id"]
assert hashlib.sha256(np.sort(tr_ids.values).tobytes()).hexdigest() == spec["training_set"]["stay_id_sha256"]


def prep(ids):
    f = pd.read_parquet(P / "features_clin.parquet", filters=[("stay_id", "in", list(ids))])
    f = f.set_index("stay_id").loc[list(ids)].reset_index()
    Xi = f[b["kept"]].fillna(b["fill"])                # training-set fill values, means and SDs
    return Xi, (Xi - b["mu"]) / b["sd"], f.y.values.astype(int)


Xtr_i, Xtr_s, ytr = prep(tr_ids)
assert len(ytr) == spec["training_set"]["n"] and int(ytr.sum()) == spec["training_set"]["events"]
refit = clone(lr).fit(Xtr_s[sel], ytr)                 # bundle preprocessing reproduces the frozen LR
assert np.allclose(refit.coef_, lr.coef_, atol=1e-8) and np.allclose(refit.intercept_, lr.intercept_, atol=1e-8)

if args.smoke:                                        # training subsample stands in for the held-out set
    tr_pos = np.flatnonzero((split.split == "train").values)
    _, idx = next(StratifiedShuffleSplit(1, test_size=512, random_state=SEED).split(tr_pos, split.y.values[tr_pos]))
    ev_ids = split.stay_id.values[tr_pos[idx]]
else:
    ev_ids = split.loc[split.split == "valid", "stay_id"].values
    assert len(ev_ids) == 512 and set(ev_ids).isdisjoint(set(tr_ids))
Xev_i, Xev_s, yev = prep(ev_ids)

# ---------------------------------------------------------------- performance
N_BOOT = 2000


def boot_auprc_ci(y, p, n_boot=N_BOOT, seed=SEED):
    """Stratified bootstrap: resample events and non-events separately (fixed counts), percentile 95% CI."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        vals[i] = auprc(y[idx], p[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


pred = {"LR (L2)": lr.predict_proba(Xev_s[sel])[:, 1], "RF (tuned)": rf.predict_proba(Xev_i[sel])[:, 1]}
dl = delong_roc_test(yev, pred["LR (L2)"], pred["RF (tuned)"])
rows, dca_rows = [], []
for m, p in pred.items():
    auc, lo, hi = delong_auc_ci(yev, p)
    assert abs(auc - roc_auc_score(yev, p)) < 1e-9
    ap_lo, ap_hi = boot_auprc_ci(yev, p)
    cal = cal_intercept_slope_checked(yev, p)
    t = THR[m]
    rows.append({
        "model": m, "role": "recommended (D8.14)" if m == "LR (L2)" else "comparator (tuned RF)",
        "n_eval": len(yev), "events_eval": int(yev.sum()), "prevalence_eval": float(yev.mean()),
        "auroc": auc, "auroc_lo": lo, "auroc_hi": hi,
        "auprc": auprc(yev, p), "auprc_lo": ap_lo, "auprc_hi": ap_hi,
        "brier": brier_score_loss(yev, p), "scaled_brier": scaled_brier(yev, p),
        "cal_intercept": cal["cal_intercept"], "cal_slope": cal["cal_slope"],
        "cal_converged": cal["cal_intercept_ok"] and cal["cal_slope_ok"],
        "threshold_frozen": t, **threshold_metrics(yev, p, t), "flagged_pct": 100 * float(np.mean(p >= t)),
        "delong_lr_minus_rf_diff": dl["diff"], "delong_lr_minus_rf_lo": dl["diff_lo"],
        "delong_lr_minus_rf_hi": dl["diff_hi"], "delong_lr_minus_rf_p": dl["p_value"],
        "n_train": len(ytr), "events_train": int(ytr.sum()), "n_candidates": len(CAND), "n_selected": len(sel),
        "epv_selected": int(ytr.sum()) / len(sel), "events_per_candidate": int(ytr.sum()) / len(CAND),
    })
    d = decision_curve(yev, p, thresholds=np.round(np.arange(0.05, 0.6001, 0.01), 2))
    dca_rows.append(d.assign(model=m))
res = pd.DataFrame(rows)
dca = pd.concat(dca_rows)[["model", "threshold", "net_benefit_model", "net_benefit_all", "net_benefit_none"]]

# ---------------------------------------------------------------- interpretation (training-set fits)
Xg = sm.add_constant(Xtr_s[sel])
glm = sm.GLM(ytr, Xg, family=sm.families.Binomial()).fit()
assert glm.converged, "unpenalised GLM did not converge"
ci = glm.conf_int()
coef = pd.DataFrame({
    "feature": sel,
    "sd_original_units": Xtr_i[sel].std(ddof=0).values,     # 1 SD of the standardised predictor, in original units
    "coef_l2_per_sd": lr.coef_[0], "or_l2_per_sd": np.exp(lr.coef_[0]),
    "coef_glm_per_sd": glm.params[sel].values, "or_glm_per_sd": np.exp(glm.params[sel].values),
    "or_glm_lo": np.exp(ci.loc[sel, 0].values), "or_glm_hi": np.exp(ci.loc[sel, 1].values),
    "p_glm": glm.pvalues[sel].values,
    "vif": [variance_inflation_factor(Xg.values, i + 1) for i in range(len(sel))],
}).sort_values("coef_l2_per_sd", key=abs, ascending=False)
intercepts = {"intercept_l2": float(lr.intercept_[0]), "intercept_glm": float(glm.params["const"])}

sv = shap.LinearExplainer(lr, Xtr_s[sel])(Xev_s[sel])
imp = pd.Series(np.abs(sv.values).mean(axis=0), index=sel).sort_values(ascending=False)
shp = pd.DataFrame([{"model": "LR (L2)", "rank": r, "feature": f_, "mean_abs_shap_logodds": v}
                    for r, (f_, v) in enumerate(imp.items(), 1)])

# ---------------------------------------------------------------- write (all outputs together), then print
if not args.smoke and any(f.exists() for f in OUTPUTS.values()):
    raise SystemExit("an output appeared during the run; not overwriting")
res.round(6).to_csv(OUTPUTS["res"], index=False)
dca.round(6).to_csv(OUTPUTS["dca"], index=False)
coef.assign(**intercepts).round(6).to_csv(OUTPUTS["coef"], index=False)
shp.round(6).to_csv(OUTPUTS["shap"], index=False)

if args.smoke:
    print(f"SMOKE OK: eval n={len(yev)} (training subsample), {len(sel)} variables; outputs in {OUT}: "
          f"{[f.name for f in OUTPUTS.values()]}; rows {len(res)}/{len(dca)}/{len(coef)}/{len(shp)}")
    sys.exit(0)

pd.set_option("display.width", 220)
print(res.round(4).T.to_string())
for m_, d in dca.groupby("model", sort=False):
    better = d[(d.net_benefit_model > d.net_benefit_all) & (d.net_benefit_model > 0)].threshold
    print(f"[DCA] {m_}: net benefit > treat-all and > 0 at {len(better)}/{len(d)} thresholds "
          f"({better.min():.2f}–{better.max():.2f})" if len(better) else f"[DCA] {m_}: 0/{len(d)} thresholds")
print(coef.round(3).to_string(index=False))
print(intercepts)
print(shp.head(10).round(4).to_string(index=False))
