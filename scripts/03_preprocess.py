"""Stage 3 — preprocessing (two versions) and Table 1.

- Fixed stratified 80/20 split (random_state=42), shared by every later step.
- Variables with > 20% missing are dropped (decided on the full cohort, as in the paper).
- Imputation: continuous -> mean; binary / ordinal count scores -> mode.
  faithful : imputer + scaler fitted on the full dataset (paper, SPSS "Replace Missing Values")
  noleak   : imputer + scaler fitted on the training set only, applied to both sets
- Table 1 on observed (pre-imputation) data; a post-imputation version is also written for comparison.
Outputs: data/processed/{split,faithful_imputed,faithful_scaled,noleak_imputed,noleak_scaled}.parquet,
         results/tables/table1.csv, table1_after_imputation.csv, missingness.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from src.common import load_config, path
from src.preprocess import impute_scale
from src.variables import BINARY, FEATURES

cfg = load_config()
SEED = cfg["seed"]
P, T = path("processed"), path("tables")

df = pd.read_parquet(P / "features.parquet")
assert list(df.columns[5:60]) == FEATURES

# ---------------------------------------------------------------- split
tr_idx, te_idx = train_test_split(df.index, test_size=cfg["model"]["test_size"], stratify=df["y"], random_state=SEED)
df["split"] = "train"
df.loc[te_idx, "split"] = "valid"
df[["stay_id", "split", "y"]].to_parquet(P / "split.parquet", index=False)

# ---------------------------------------------------------------- missingness filter
miss = df[FEATURES].isna().mean().rename("missing_frac").to_frame()
miss["dropped"] = miss["missing_frac"] > 0.20
miss.round(4).to_csv(T / "missingness.csv")
kept = [f for f in FEATURES if not miss.loc[f, "dropped"]]

ids = df[["stay_id", "split", "y"]]
versions = {}
for name, fit_rows in [("faithful", df.index), ("noleak", df.index[df.split == "train"])]:
    Xi, Xs = impute_scale(df[kept], fit_rows)
    pd.concat([ids, Xi], axis=1).to_parquet(P / f"{name}_imputed.parquet", index=False)
    pd.concat([ids, Xs], axis=1).to_parquet(P / f"{name}_scaled.parquet", index=False)
    versions[name] = Xi
    assert not Xi.isna().any().any()

# ---------------------------------------------------------------- Table 1
def fmt_mean(s):   return f"{s.mean():.2f} ± {s.std():.2f}"
def fmt_med(s):    return f"{s.median():.2f} ({s.quantile(.25):.2f}, {s.quantile(.75):.2f})"

def table1(data: pd.DataFrame) -> pd.DataFrame:
    g0, g1 = data[data.y == 0], data[data.y == 1]
    rows = [{"variable": "n", "total": len(data), "survival": len(g0), "non_survival": len(g1)}]
    for f in kept:
        a, b, t = g0[f].dropna(), g1[f].dropna(), data[f].dropna()
        row = {"variable": f, "missing_pct": round(100 * data[f].isna().mean(), 2)}
        if f in BINARY:
            ct = pd.crosstab(data[f], data.y)
            chi2, p, _, _ = stats.chi2_contingency(ct, correction=False)
            row.update(type="categorical", test="chi-square",
                       total=f"{int(t.sum())} ({100 * t.mean():.2f}%)", survival=f"{int(a.sum())} ({100 * a.mean():.2f}%)",
                       non_survival=f"{int(b.sum())} ({100 * b.mean():.2f}%)", p_value=p)
        else:
            p_sw0, p_sw1 = stats.shapiro(a).pvalue, stats.shapiro(b).pvalue
            normal = p_sw0 > 0.05 and p_sw1 > 0.05
            if normal:
                p = stats.ttest_ind(a, b, equal_var=False).pvalue
                row.update(type="continuous-normal", test="Welch t-test", total=fmt_mean(t), survival=fmt_mean(a),
                           non_survival=fmt_mean(b), p_value=p)
            else:
                p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
                row.update(type="continuous-nonnormal", test="Mann-Whitney U", total=fmt_med(t), survival=fmt_med(a),
                           non_survival=fmt_med(b), p_value=p)
            row.update(shapiro_p_survival=p_sw0, shapiro_p_nonsurvival=p_sw1,
                       mean_sd_total=fmt_mean(t), median_iqr_total=fmt_med(t))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_display"] = out["p_value"].map(lambda p: "" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}"))
    return out

t1 = table1(df)
t1.to_csv(T / "table1.csv", index=False)
t1_imp = table1(pd.concat([df[["y"]], versions["faithful"]], axis=1))
t1_imp.to_csv(T / "table1_after_imputation.csv", index=False)

pd.set_option("display.width", 220)
print("split:", df.groupby("split").y.agg(["size", "sum", "mean"]).round(4).to_dict("index"))
print("dropped for >20% missing:", [f for f in FEATURES if miss.loc[f, "dropped"]] or "none")
print("features kept:", len(kept))
print(t1[["variable", "missing_pct", "total", "survival", "non_survival", "test", "p_display"]].to_string(index=False))
sig = t1.loc[t1.p_value < 0.05, "variable"].tolist()
print(f"\nsignificant (p<0.05): {len(sig)} ->", sig)
print("normal by Shapiro-Wilk (both groups):", t1.loc[t1.type == "continuous-normal", "variable"].tolist())
