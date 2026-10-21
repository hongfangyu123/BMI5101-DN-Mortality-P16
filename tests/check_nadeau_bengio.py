"""Unit check for src.metrics.nadeau_bengio with simulated data (no MIMIC data).

1. Formula: sigma^2 is the sample variance (ddof=1) of all k*r paired differences, the variance is multiplied by
   1/(k*r) + n_test/n_train, and the t distribution has k*r - 1 degrees of freedom (hand computation).
2. Length check: a wrong number of differences is rejected.
3. Null simulation: two logistic regressions on two equally informative, independent feature sets are compared by
   10 x 5 repeated stratified CV on many simulated data sets (true AUROC difference = 0). The corrected test should
   reject at roughly the nominal 5% level, while the naive paired t-test (no correction) over-rejects.
Run: conda run -n dn_repro python tests/check_nadeau_bengio.py
"""
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import nadeau_bengio  # noqa: E402

rng = np.random.default_rng(2026)
k, r = 10, 5

# ---- 1. formula
d = rng.normal(0.01, 0.03, k * r)
res = nadeau_bengio(d, n_train=900, n_test=100, k=k, r=r)
s2 = statistics.variance(d.tolist())                       # ddof = 1
se = np.sqrt((1 / (k * r) + 100 / 900) * s2)
q = stats.t.ppf(0.975, k * r - 1)
assert res["df"] == k * r - 1 == 49
assert np.isclose(res["se_corrected"], se) and np.isclose(res["lo"], d.mean() - q * se) and np.isclose(res["hi"], d.mean() + q * se)
assert np.isclose(res["p_value"], 2 * stats.t.sf(abs(d.mean() / se), k * r - 1))
se_ddof0 = np.sqrt((1 / (k * r) + 100 / 900) * np.var(d))    # population variance would give a different (wrong) SE
assert not np.isclose(res["se_corrected"], se_ddof0)
print("1. formula (ddof=1, df=k*r-1, correction factor): OK")

# ---- 2. length check
try:
    nadeau_bengio(d[:-1], 900, 100, k, r)
    raise SystemExit("length check failed")
except AssertionError:
    print("2. wrong number of differences rejected: OK")

# ---- 3. null simulation
n_sim, n = 200, 300
rej_nb = rej_naive = 0
for s in range(n_sim):
    x1, x2 = rng.normal(size=(n, 2)), rng.normal(size=(n, 2))
    lp = 0.8 * x1.sum(1) + 0.8 * x2.sum(1) - 1.0             # both feature sets equally informative
    y = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    diffs, ntr, nte = [], [], []
    for rep in range(r):
        for itr, ite in StratifiedKFold(k, shuffle=True, random_state=1000 * s + rep).split(x1, y):
            a = LogisticRegression().fit(x1[itr], y[itr]).predict_proba(x1[ite])[:, 1]
            b = LogisticRegression().fit(x2[itr], y[itr]).predict_proba(x2[ite])[:, 1]
            diffs.append(roc_auc_score(y[ite], a) - roc_auc_score(y[ite], b))
            ntr.append(len(itr)); nte.append(len(ite))
    res = nadeau_bengio(diffs, np.mean(ntr), np.mean(nte), k, r)
    rej_nb += res["p_value"] < 0.05
    naive_p = 2 * stats.t.sf(abs(np.mean(diffs) / (np.std(diffs, ddof=1) / np.sqrt(k * r))), k * r - 1)
    rej_naive += naive_p < 0.05
print(f"3. null simulation ({n_sim} data sets, 10x5 CV): rejection rate corrected {rej_nb / n_sim:.3f}, "
      f"naive {rej_naive / n_sim:.3f}")
assert rej_nb / n_sim <= 0.10, "corrected test rejects far too often under the null"
assert rej_naive / n_sim > rej_nb / n_sim, "naive test should reject more often than the corrected test"
print("all checks passed")
