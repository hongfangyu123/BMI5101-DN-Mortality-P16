"""Approximate corrected paired intervals for completed repeated holdouts; summaries only, no training or slide edits."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
ROOT = Path(__file__).resolve().parents[1]
PAIRS = [('static_stacking', 'static_lr'), ('combined_stacking', 'static_stacking'), ('combined_stacking', 'static_lr')]
METRICS = ('AUROC', 'AUPRC', 'Brier')
SOURCE = 'https://doi.org/10.1023/A:1024068626366'

def corrected_interval(differences, test_train_ratio):
    """Nadeau–Bengio corrected resampled t approximation, n-1 degrees of freedom."""
    d = np.asarray(differences, float)
    if d.ndim != 1 or len(d) < 2 or (not np.isfinite(d).all()) or (test_train_ratio <= 0):
        raise ValueError('At least two finite paired differences and a positive test/train ratio are required.')
    n = len(d)
    mean, sd = (float(d.mean()), float(d.std(ddof=1)))
    se = float(np.sqrt((1 / n + test_train_ratio) * sd ** 2))
    q = float(stats.t.ppf(0.975, n - 1))
    t = mean / se if se > 0 else float('nan')
    p = float(2 * stats.t.sf(abs(t), n - 1)) if se > 0 else float('nan')
    return {'n_splits': n, 'difference_mean': mean, 'difference_sd': sd, 'test_train_ratio': float(test_train_ratio), 'se_corrected': se, 'ci_lo': mean - q * se, 'ci_hi': mean + q * se, 'df': n - 1, 'p_value': p, 'ci_level': 0.95, 'ci_multiplicity_adjusted': False}


def result_metadata(output):
    """Read local run metadata or verify the aggregate-only published archive."""
    if (output / 'run_identity.json').exists() and (output / 'protocol.json').exists():
        return (json.loads((output / 'run_identity.json').read_text()),
                json.loads((output / 'protocol.json').read_text()))
    archived = json.loads((output / 'source_provenance.json').read_text())
    for name in ('split_metrics.csv', 'paired_summary.csv'):
        expected = archived['aggregate_file_sha256'][name]
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Published aggregate input changed: ' + name)
    return archived, archived

def calculate(output):
    identity, protocol = result_metadata(output)
    if not identity['complete'] or identity['n_completed_splits'] != 5:
        raise ValueError('This supplement requires the five prespecified completed splits.')
    frame = pd.read_csv(output / 'split_metrics.csv')
    if frame.duplicated(['repeat', 'split_seed', 'experiment']).any() or len(frame) != 15:
        raise ValueError('Exactly three recipe results are required for every planned split.')
    if set(frame.split_seed) != set(protocol['split_seeds']):
        raise ValueError('Split seeds differ from the frozen protocol.')
    rows = []
    for new, ref in PAIRS:
        a = frame[frame.experiment.eq(new)].set_index(['repeat', 'split_seed']).sort_index()
        b = frame[frame.experiment.eq(ref)].set_index(['repeat', 'split_seed']).sort_index()
        if not a.index.equals(b.index) or len(a) != 5:
            raise ValueError('Models are not paired on the same five patient splits.')
        for column in ['n_train', 'n_test', 'events_train', 'events_test']:
            if not np.array_equal(a[column], b[column]):
                raise ValueError('Paired models have different sample or event counts.')
        ratios = a.n_test.to_numpy() / a.n_train.to_numpy()
        if not np.allclose(ratios, ratios[0]):
            raise ValueError('The five splits must have the same test/train ratio.')
        for metric in METRICS:
            result = corrected_interval(a[metric].to_numpy() - b[metric].to_numpy(), float(ratios[0]))
            rows.append({'experiment': new, 'reference': ref, 'metric': metric, **result})
    result = pd.DataFrame(rows)
    result['p_holm_three_comparisons_per_metric'] = np.nan
    for metric in METRICS:
        ix = result.index[result.metric.eq(metric) & result.p_value.notna()]
        result.loc[ix, 'p_holm_three_comparisons_per_metric'] = multipletests(result.loc[ix, 'p_value'], method='holm')[1]
    old = pd.read_csv(output / 'paired_summary.csv').set_index(['experiment', 'reference', 'metric'])
    for row in result.itertuples():
        expected = old.loc[(row.experiment, row.reference, row.metric), 'difference_mean']
        if not np.isclose(row.difference_mean, expected, atol=1e-12, rtol=0):
            raise ValueError('Recomputed paired differences disagree with the original summary.')
    return result

def write_supplement(output, result):
    result.to_csv(output / 'paired_corrected_ci.csv', index=False)
    names = {'static_lr': 'Static LR', 'static_stacking': 'Static stacking', 'combined_stacking': 'Combined stacking'}
    table = ['| Comparison (first minus reference) | Metric | Mean difference | Approx. corrected 95% CI | Raw p |', '|---|---|---:|---:|---:|']
    for row in result.itertuples():
        table.append(f'| {names[row.experiment]} − {names[row.reference]} | {row.metric} | {row.difference_mean:+.5f} | [{row.ci_lo:+.5f}, {row.ci_hi:+.5f}] | {row.p_value:.4f} |')
    text = '\n\n'.join(['## Approximate corrected intervals for five-split model differences (2026-10-27 supplement)', 'The initial split-sensitivity analysis reported means, SDs, ranges and win counts only. Overlapping splits cannot be treated as independent ordinary paired-t observations. This adds an approximate correlation-corrected analysis after seeing the results; it is not a prespecified confirmatory test.', 'Use the Nadeau-Bengio corrected resampled-t approximation: each d is extension minus reference on the same test patients. SE = sqrt((1/R + n_test/n_train)*s_d^2); CI = mean(d) +/- t(0.975,R-1)*SE. Here R=5, n_test=512, n_train=2045, df=4. Applying this to non-additive AUROC/AUPRC is approximate and does not guarantee exact 95% coverage. Five splits give little ability to assess distributional assumptions.', '\n'.join(table), 'All AUROC/AUPRC/Brier difference intervals include zero, so a reliable improvement is not established. This does not prove equivalence or absence of benefit. Positive AUROC/AUPRC and negative Brier differences are favorable.', 'Reported intervals and raw p-values are not multiplicity adjusted. The CSV additionally gives Holm-adjusted p-values across the three comparisons for each metric. Resampling-correlation correction and multiple-comparison adjustment address different issues.', 'These intervals concern mean differences across retrained recipes, not the original single-holdout patient-bootstrap intervals. They do not correct adaptive model-selection bias, exclude leakage or provide new blind/external validation. Only saved aggregate results are used; no training or slide changes.', f'Method: Nadeau C, Bengio Y. Inference for the Generalization Error. Machine Learning (2003). [Paper]({SOURCE}).']) + '\n'
    (output / 'paired_corrected_ci.md').write_text(text, encoding='utf-8')
    metadata = {'method': 'approximate_corrected_resampled_t_interval', 'post_hoc': True, 'source': SOURCE, 'entry_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'split_metrics_sha256': hashlib.sha256((output / 'split_metrics.csv').read_bytes()).hexdigest(), 'retraining': False, 'ppt_modified': False, 'limitations': ['five overlapping splits', 'approximate coverage for AUROC/AUPRC', 'does not correct adaptive model selection', 'marginal CIs not multiplicity adjusted']}
    (output / 'paired_corrected_ci_method.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    begin, end = ('<!-- PAIRED_CORRECTED_CI_BEGIN -->', '<!-- PAIRED_CORRECTED_CI_END -->')
    report = output / 'report.md'
    existing = report.read_text()
    if begin in existing:
        head, tail = existing.split(begin, 1)
        _, rest = tail.split(end, 1)
        updated = head.rstrip() + '\n\n' + begin + '\n' + text + end + rest
    else:
        updated = existing.rstrip() + '\n\n' + begin + '\n' + text + end + '\n'
    report.write_text(updated, encoding='utf-8')
    for path in [ROOT / 'notebooks/08_repeated_holdout.ipynb', output / 'repeated_holdout_results.ipynb']:
        nb = (json.loads(path.read_text()) if path.exists() else
              {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': []})
        nb['cells'] = [c for c in nb['cells'] if c.get('id') != 'paired-corrected-confidence-intervals']
        nb['cells'].append({'cell_type': 'markdown', 'id': 'paired-corrected-confidence-intervals', 'metadata': {}, 'source': text.splitlines(keepends=True)})
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Append the supplement to result files and notebooks; print only by default')
    args = parser.parse_args()
    base = ROOT / 'results/repeated_holdout'
    marker = base / 'latest.txt'
    output = base / (marker.read_text().strip() if marker.exists() else 'd6f09e8dedcfbfc1')
    result = calculate(output)
    print(result[['experiment', 'reference', 'metric', 'difference_mean', 'ci_lo', 'ci_hi', 'p_value']].round(6).to_string(index=False))
    if args.write:
        write_supplement(output, result)
        print(f"Supplement: {output / 'paired_corrected_ci.csv'}; no retraining or slide edits.")
if __name__ == '__main__':
    main()
