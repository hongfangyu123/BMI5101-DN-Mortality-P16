"""Post-hoc descriptive analysis — 95% CIs for the calibration intercept and slope of the frozen CLIN model on the
held-out set (requested at review). Nothing is refitted: the frozen model, holdout_clin.csv and every other existing
result are left unchanged, and 04i is not re-run.

The frozen model is loaded and the held-out predictions rebuilt with the code of 04i (guards 1 and 2, prep(), sel and
the LR (L2) refit-consistency assert, copied verbatim; guard 3 "outputs must not exist" is intentionally not copied).
For LR (L2) and tuned RF:
  - point estimates from src.metrics.cal_intercept_slope_checked, asserted equal to cal_intercept / cal_slope in
    results/clinical/holdout_clin.csv (|difference| < 1e-6), otherwise the script stops;
  - Wald 95% CI: slope from the GLM of y on logit(p); calibration-in-the-large from the intercept-only GLM with
    offset logit(p) (same fits as cal_intercept_slope_checked, conf_int());
  - stratified bootstrap 95% percentile CI: 2,000 resamples, seed 42, events and non-events resampled separately as in
    04i boot_auprc_ci; predictions only are resampled (no refitting); resamples in which either fit does not converge
    (cal_intercept_slope_checked returns NaN) are skipped and counted (n_boot_ok = resamples used);
  - Wald tests: slope = 1 (z = (slope − 1) / SE) and intercept = 0 (GLM Wald p value), two-sided.
Output (aggregate only): results/clinical/holdout_calibration_ci_clin.csv.

--smoke --out DIR: as in 04i, a stratified 512-patient subsample of the TRAINING set stands in for the held-out set,
outputs go to DIR, the comparison with holdout_clin.csv is skipped and no metric is printed.
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
import statsmodels.api as sm
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from src.common import ROOT, path
from src.metrics import cal_intercept_slope_checked
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
OUT_FILE = OUT / "holdout_calibration_ci_clin.csv"

# ---------------------------------------------------------------- guard 1 (as 04i): frozen files committed and unchanged
def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)

for f in (SPEC_FILE, REC_FILE):
    rel = str(f.relative_to(ROOT))
    if git("ls-files", "--error-unmatch", rel).returncode != 0 or git("cat-file", "-e", f"HEAD:{rel}").returncode != 0:
        raise SystemExit(f"refusing to run: {rel} is not committed")
    if git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"refusing to run: {rel} differs from HEAD (staged or unstaged changes)")

# ---------------------------------------------------------------- guard 2 (as 04i): model bundle matches the frozen spec
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

# ---------------------------------------------------------------- data (as 04i)
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

pred = {"LR (L2)": lr.predict_proba(Xev_s[sel])[:, 1], "RF (tuned)": rf.predict_proba(Xev_i[sel])[:, 1]}

# ---------------------------------------------------------------- calibration CIs
N_BOOT = 2000
saved = None if args.smoke else pd.read_csv(CL / "holdout_clin.csv").set_index("model")


def wald(y, p):
    """Same GLMs as cal_intercept_slope_checked; returns estimates, Wald 95% CIs and Wald p values."""
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(q / (1 - q))
    fs = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    fi = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit()
    assert fs.converged and fi.converged
    s, s_se, (s_lo, s_hi) = float(fs.params[1]), float(fs.bse[1]), fs.conf_int()[1]
    i, (i_lo, i_hi) = float(fi.params[0]), fi.conf_int()[0]
    return {"cal_intercept": i, "int_wald_lo": float(i_lo), "int_wald_hi": float(i_hi),
            "cal_slope": s, "slope_wald_lo": float(s_lo), "slope_wald_hi": float(s_hi),
            "p_slope_eq_1": float(2 * stats.norm.sf(abs((s - 1) / s_se))), "p_int_eq_0": float(fi.pvalues[0])}


def boot_cal_ci(y, p, n_boot=N_BOOT, seed=SEED):
    """Stratified bootstrap as 04i boot_auprc_ci (events / non-events resampled separately), percentile 95% CI;
    resamples where either calibration fit fails are skipped."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    ints, slopes = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        c = cal_intercept_slope_checked(y[idx], p[idx])
        if c["cal_intercept_ok"] and c["cal_slope_ok"]:
            ints.append(c["cal_intercept"]); slopes.append(c["cal_slope"])
    ints, slopes = np.array(ints), np.array(slopes)
    return {"int_boot_lo": float(np.percentile(ints, 2.5)), "int_boot_hi": float(np.percentile(ints, 97.5)),
            "slope_boot_lo": float(np.percentile(slopes, 2.5)), "slope_boot_hi": float(np.percentile(slopes, 97.5)),
            "n_boot_ok": len(ints)}


rows = []
for m, p in pred.items():
    cal = cal_intercept_slope_checked(yev, p)
    assert cal["cal_intercept_ok"] and cal["cal_slope_ok"], f"{m}: calibration fit did not converge"
    if saved is not None:
        for k in ("cal_intercept", "cal_slope"):
            d = abs(cal[k] - saved.loc[m, k])
            if not d < 1e-6:
                raise SystemExit(f"STOP: {m} {k} = {cal[k]:.8f} differs from holdout_clin.csv "
                                 f"({saved.loc[m, k]:.8f}) by {d:.2e}")
    w = wald(yev, p)
    assert abs(w["cal_intercept"] - cal["cal_intercept"]) < 1e-12 and abs(w["cal_slope"] - cal["cal_slope"]) < 1e-12
    rows.append({"model": m, **w, **boot_cal_ci(yev, p)})
cols = ["model", "cal_intercept", "int_wald_lo", "int_wald_hi", "int_boot_lo", "int_boot_hi", "cal_slope",
        "slope_wald_lo", "slope_wald_hi", "slope_boot_lo", "slope_boot_hi", "p_slope_eq_1", "p_int_eq_0", "n_boot_ok"]
res = pd.DataFrame(rows)[cols]
res.round(6).to_csv(OUT_FILE, index=False)

if args.smoke:
    print(f"SMOKE OK: eval n={len(yev)} (training subsample); {OUT_FILE} rows {len(res)}, "
          f"n_boot_ok {res.n_boot_ok.tolist()}/{N_BOOT}")
    sys.exit(0)
pd.set_option("display.width", 220)
print(res.round(4).to_string(index=False))
