"""Stage 8f — CLIN Table 1 (descriptive; decisions.md D8.17).

Whole CLIN cohort (2,557 patients, training + held-out) by 1-year outcome, on observed (pre-imputation) values, same
rules as stage 3 (D3.5, D3.6): binary -> n (%) and Pearson chi-square without continuity correction; continuous ->
Shapiro-Wilk per group, median (IQR) and Mann-Whitney U unless both groups are normal. Variables: the 55 CLIN
predictors (CLIN comorbidities, CCI without age score, SAPS-II with the recomputed chronic item), wbc_diff_missing and
has_prior_admission. Also the 1-year mortality by wbc_diff_missing (descriptive).
Also obesity by source (prior admissions vs index admission only), descriptive.
Outputs (aggregate only): results/clinical/table1_clin.csv, wbc_diff_missing_mortality_clin.csv, obesity_source_clin.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy import stats

from src.common import ROOT, path
from src.variables import BINARY, FEATURES

OUT = ROOT / "results" / "clinical"
df = pd.read_parquet(path("processed") / "features_clin.parquet")
assert len(df) == 2557 and df.stay_id.is_unique
VARS = FEATURES + ["wbc_diff_missing", "has_prior_admission"]
BIN = set(BINARY) | {"wbc_diff_missing", "has_prior_admission"}


def fmt_mean(s): return f"{s.mean():.2f} ± {s.std():.2f}"
def fmt_med(s):  return f"{s.median():.2f} ({s.quantile(.25):.2f}, {s.quantile(.75):.2f})"


def table1(data: pd.DataFrame) -> pd.DataFrame:      # same rules as scripts/03_preprocess.py
    g0, g1 = data[data.y == 0], data[data.y == 1]
    rows = [{"variable": "n", "total": len(data), "survival": len(g0), "non_survival": len(g1)}]
    for f in VARS:
        a, b, t = g0[f].dropna(), g1[f].dropna(), data[f].dropna()
        row = {"variable": f, "missing_pct": round(100 * data[f].isna().mean(), 2)}
        if f in BIN:
            ct = pd.crosstab(data[f], data.y)
            chi2, p, _, _ = stats.chi2_contingency(ct, correction=False)
            row.update(type="categorical", test="chi-square",
                       total=f"{int(t.sum())} ({100 * t.mean():.2f}%)", survival=f"{int(a.sum())} ({100 * a.mean():.2f}%)",
                       non_survival=f"{int(b.sum())} ({100 * b.mean():.2f}%)", p_value=p)
        else:
            p_sw0, p_sw1 = stats.shapiro(a).pvalue, stats.shapiro(b).pvalue
            if p_sw0 > 0.05 and p_sw1 > 0.05:
                row.update(type="continuous-normal", test="Welch t-test", total=fmt_mean(t), survival=fmt_mean(a),
                           non_survival=fmt_mean(b), p_value=stats.ttest_ind(a, b, equal_var=False).pvalue)
            else:
                row.update(type="continuous-nonnormal", test="Mann-Whitney U", total=fmt_med(t), survival=fmt_med(a),
                           non_survival=fmt_med(b), p_value=stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
            row.update(shapiro_p_survival=p_sw0, shapiro_p_nonsurvival=p_sw1, mean_sd_total=fmt_mean(t),
                       median_iqr_total=fmt_med(t))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_display"] = out["p_value"].map(lambda p: "" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}"))
    return out


t1 = table1(df)
t1.to_csv(OUT / "table1_clin.csv", index=False)

g = df.groupby("wbc_diff_missing").y.agg(n="size", deaths="sum")
g["mortality"] = g.deaths / g.n
chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df.wbc_diff_missing, df.y), correction=False)
g = g.reset_index().assign(chi2_p=p)
assert g.n.min() >= 10
g.round(6).to_csv(OUT / "wbc_diff_missing_mortality_clin.csv", index=False)

# obesity by source (descriptive, for the D8.16 discussion): CLIN rule (prior ∪ index) vs prior admissions only
cm = pd.read_parquet(path("processed") / "comorb_clin.parquet",
                     columns=["stay_id", "obesity", "obesity_prior", "has_prior_admission"]).merge(df[["stay_id", "y"]], on="stay_id")
assert len(cm) == len(df) and (cm.obesity_prior <= cm.obesity).all()
src = {"obesity = 1 (CLIN rule: prior ∪ index)": cm.obesity == 1,
       "obesity = 1 in prior admissions": cm.obesity_prior == 1,
       "obesity = 1 only from the index admission": (cm.obesity == 1) & (cm.obesity_prior == 0),
       "obesity = 0 (CLIN rule)": cm.obesity == 0,
       "  among patients with a prior admission: obesity = 1 only from the index admission":
           (cm.has_prior_admission == 1) & (cm.obesity == 1) & (cm.obesity_prior == 0),
       "  among patients with a prior admission: obesity = 1 in prior admissions":
           (cm.has_prior_admission == 1) & (cm.obesity_prior == 1)}
ob = pd.DataFrame([{"group": k, "n": int(m.sum()), "deaths": int(cm.y[m].sum()), "mortality": float(cm.y[m].mean())}
                   for k, m in src.items()])
assert ob.n.min() >= 10
ob.round(6).to_csv(OUT / "obesity_source_clin.csv", index=False)
print(ob.round(4).to_string(index=False))

print(f"table1_clin.csv: {len(t1) - 1} variables; survivors {t1.survival.iloc[0]}, deaths {t1.non_survival.iloc[0]}; "
      f"p < 0.05 for {int((t1.p_value < 0.05).sum())}")
print(g.round(4).to_string(index=False))
