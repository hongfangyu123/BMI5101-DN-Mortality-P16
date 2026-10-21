"""SHAP figures of the frozen CLIN model (LR (L2)) on the held-out set, for paper Figure 4a. Plots only: nothing is refitted.

The frozen model bundle (data/interim/final_clin/final_clin.pkl) is checked against the committed fit record and frozen
spec exactly as in 04j. SHAP values are computed with the same algorithm as 04i (LinearExplainer, standardised training
set as background, the 512-patient held-out set explained, log-odds scale). Every variable's mean |SHAP| is compared with
results/clinical/shap_clin.csv after the same 6-decimal rounding used when that file was written (|difference| < 1e-9),
and the ranking must match; any mismatch aborts before plotting. No patient-level values are written anywhere; no force
plots are drawn.

Figures (results/figures/, PNG at 300 dpi and PDF, white background, no figure title):
  figC5_shap_bar_clin_holdout       mean |SHAP|, all 22 selected variables
  figC6_shap_beeswarm_clin_holdout  beeswarm, all 22 variables; colour = imputed value in original units (as 04_model)
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.common import ROOT, path
from src.variables import FEATURES

warnings.filterwarnings("ignore", category=FutureWarning)
CL, F, P = ROOT / "results" / "clinical", path("figures"), path("processed")
SPEC_FILE, REC_FILE = CL / "frozen_model_clin.json", CL / "final_model_clin.json"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 10})

# paper names (docs/paper_v2_en.md)
LABELS = {
    "rdw_mean": "RDW", "cci": "CCI (without age)", "age": "Age", "apsiii": "APS-III", "oasis": "OASIS",
    "sapsii": "SAPS-II", "lods": "LODS", "chloride_max": "Chloride (max)", "aniongap_min": "Anion gap (min)",
    "bun_min": "Urea nitrogen (min)", "calcium_min": "Calcium (min)", "pt_min": "PT (min)", "ptt_min": "PTT (min)",
    "resp_rate_mean": "Respiratory rate", "monocytes_mean": "Monocytes", "nlr": "NLR", "plr": "PLR", "lmr": "LMR",
    "sii": "SII", "obesity": "Obesity (coded)", "wbc_diff_missing": "No white-cell differential",
    "gender_male": "Male sex",
}

# ---------------------------------------------------------------- frozen model (same checks as 04j)
spec, rec = json.loads(SPEC_FILE.read_text()), json.loads(REC_FILE.read_text())
BUNDLE = ROOT / rec["bundle"]
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == rec["bundle_sha256"], "model file differs from the fit record"
b = pickle.load(open(BUNDLE, "rb"))
assert b["spec_sha256"] == hashlib.sha256(SPEC_FILE.read_bytes()).hexdigest() == rec["spec_sha256"]
assert b["candidates"] == FEATURES + ["wbc_diff_missing"] and b["lasso"]["selected"] == rec["selected"]
sel = b["lasso"]["selected"]
lr = b["models"]["LR (L2)"]
assert list(lr.feature_names_in_) == sel and set(sel) <= set(LABELS), "unexpected selected variables"

split = pd.read_parquet(P / "split_clin.parquet")
tr_ids = split.loc[split.split == "train", "stay_id"].values
ev_ids = split.loc[split.split == "valid", "stay_id"].values
assert hashlib.sha256(np.sort(tr_ids).tobytes()).hexdigest() == spec["training_set"]["stay_id_sha256"]
assert len(ev_ids) == 512 and set(ev_ids).isdisjoint(set(tr_ids))


def prep(ids):                                          # training-set fill values, means and SDs (as 04i / 04j)
    f = pd.read_parquet(P / "features_clin.parquet", filters=[("stay_id", "in", list(ids))]).set_index("stay_id").loc[list(ids)]
    Xi = f[b["kept"]].fillna(b["fill"])
    return Xi, (Xi - b["mu"]) / b["sd"]


Xtr_i, Xtr_s = prep(tr_ids)
Xev_i, Xev_s = prep(ev_ids)

# ---------------------------------------------------------------- SHAP (04i line 211) and consistency check
sv = shap.LinearExplainer(lr, Xtr_s[sel])(Xev_s[sel])
imp = pd.Series(np.abs(sv.values).mean(axis=0), index=sel).sort_values(ascending=False)
ref = pd.read_csv(CL / "shap_clin.csv")
assert (ref.model == "LR (L2)").all() and len(ref) == len(sel) == 22
ref = ref.sort_values("rank")
diff = (imp.round(6).reindex(ref.feature).values - ref.mean_abs_shap_logodds.values)
max_diff = float(np.max(np.abs(diff)))
if not (max_diff < 1e-9 and list(imp.index) == list(ref.feature)):
    raise SystemExit(f"SHAP consistency check FAILED (max |diff| {max_diff:.2e}; ranking equal: "
                     f"{list(imp.index) == list(ref.feature)}); no figures written")
print(f"SHAP consistency check PASSED: 22 variables, max |diff| after 6-dp rounding = {max_diff:.1e}, ranking identical")

# ---------------------------------------------------------------- figures
sv_disp = shap.Explanation(sv.values, base_values=sv.base_values, data=Xev_i[sel].values,
                           feature_names=[LABELS[c] for c in sel])
for stem, plot in [("figC5_shap_bar_clin_holdout", shap.plots.bar),
                   ("figC6_shap_beeswarm_clin_holdout", shap.plots.beeswarm)]:
    plt.figure()
    plot(sv_disp, max_display=len(sel), show=False)
    ax = plt.gca()
    ax.set_xlabel("mean |SHAP value| (log-odds)" if plot is shap.plots.bar else "SHAP value (log-odds)")
    ax.set_title("")
    for ext in ("png", "pdf"):
        plt.savefig(F / f"{stem}.{ext}")
    plt.close("all")
    print(f"written: results/figures/{stem}.png / .pdf")
print("top 10:", ", ".join(imp.index[:10]))
