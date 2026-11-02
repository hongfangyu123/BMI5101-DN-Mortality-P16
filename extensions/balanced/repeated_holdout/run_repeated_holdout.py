"""Five prespecified 80/20 split-sensitivity analyses; check only by default, --run to train."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'holdout'))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import run_holdout as fixed
from dn_extensions.features import fingerprint
from dn_extensions.settings import DATABASE, V3_PROCESSED
from src.metrics import decision_curve, delong_auc_ci
SPLIT_SEEDS = (101, 102, 103, 104, 105)
MODEL_SEED = 42
TEST_SIZE = 0.2
EXPERIMENTS = fixed.EXPERIMENTS
PAIRS = fixed.PAIRS
METRICS = ('AUROC', 'AUPRC', 'Brier')

def entry_hash():
    h = hashlib.sha256()
    for path in (Path(__file__), Path(fixed.__file__)):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()

def validate_full_frame(frame, subjects):
    if len(frame) != len(subjects) or not frame.stay_id.is_unique:
        raise ValueError('Feature rows and patient alignment differ.')
    if not pd.Series(subjects).is_unique or pd.Series(subjects).isna().any():
        raise ValueError('One ICU record per patient is required; never split repeated patient rows across datasets.')
    if not np.isin(frame.y, [0, 1]).all() or frame.y.nunique() != 2:
        raise ValueError('Outcomes must be complete binary labels.')

def planned_splits(frame, subjects):
    """The three recipes share the same patients per split; patients cannot cross sides within a split."""
    validate_full_frame(frame, subjects)
    positions = np.arange(len(frame))
    for repeat, seed in enumerate(SPLIT_SEEDS, start=1):
        tr, te = train_test_split(positions, test_size=TEST_SIZE, random_state=seed, stratify=frame.y.to_numpy(int))
        if set(tr) & set(te) or len(tr) + len(te) != len(frame) or (not set(subjects[tr]).isdisjoint(subjects[te])):
            raise ValueError('Training/test patients overlap or the split is incomplete.')
        yield (repeat, seed, tr, te)

def build_full_frame(development, cohort, static, reference):
    """Cache unfitted raw features only. Refit imputation, selection and SVD inside each training split."""
    stat = DATABASE.stat()
    feature_identity = {'development_sha256': fingerprint(development), 'holdout_static_sha256': fingerprint(static), 'holdout_cohort_sha256': fingerprint(cohort), 'source_code_sha256': reference['code_sha256'], 'database_bytes': stat.st_size, 'database_mtime_ns': stat.st_mtime_ns}
    feature_signature = fixed.sha_json(feature_identity)[:16]
    folder = ROOT / 'data/repeated_holdout_features' / feature_signature
    target, manifest = (folder / 'holdout_raw_features.parquet', folder / 'manifest.json')
    if target.exists() and manifest.exists():
        observed = json.loads(manifest.read_text())
        holdout = pd.read_parquet(target)
        if observed['input_identity'] != feature_identity or observed['raw_features_sha256'] != fingerprint(holdout):
            raise ValueError('Raw feature cache failed verification; refusing reuse.')
        print('Reuse verified raw held-out features; no imputation, feature selection or embedding has been fitted.', flush=True)
    else:
        print('Extract raw trajectories and prior diagnoses for the original 512 holdout patients once for all five splits.', flush=True)
        holdout = fixed.build_holdout_features(cohort, static, development)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix('.tmp.parquet')
        holdout.to_parquet(tmp, index=False)
        os.replace(tmp, target)
        fixed.atomic_json({'input_identity': feature_identity, 'raw_features_sha256': fingerprint(holdout)}, manifest)
    expected = [c for c in development if c != 'y']
    if list(holdout.columns) != expected or holdout.stay_id.tolist() != static.stay_id.tolist():
        raise ValueError('Raw held-out feature schema or order differs.')
    holdout = holdout.assign(y=fixed.load_holdout_labels(holdout.stay_id))
    frame = pd.concat([development, holdout[development.columns]], ignore_index=True)
    frame = frame.sort_values('stay_id', kind='stable').reset_index(drop=True)
    all_subjects = pd.read_parquet(V3_PROCESSED / 'cohort_clin.parquet', columns=['stay_id', 'subject_id'])
    if not all_subjects.stay_id.is_unique:
        raise ValueError('The original cohort contains duplicate ICU stays.')
    subjects = all_subjects.set_index('stay_id').loc[frame.stay_id, 'subject_id'].to_numpy()
    validate_full_frame(frame, subjects)
    if len(frame) != 2557 or int(frame.y.sum()) != 680:
        raise ValueError('The prespecified CLIN cohort must contain 2,557 patients and 680 deaths; do not redefine the cohort.')
    return (frame, subjects)

def make_protocol(frame, subjects, options, reference):
    return {'analysis': 'exploratory_repeated_holdout_split_sensitivity', 'source_cv_signature': fixed.SOURCE_SIGNATURE, 'source_code_sha256': reference['code_sha256'], 'entry_sha256': entry_hash(), 'options': options, 'software_versions': reference['software_versions'], 'full_frame_sha256': fingerprint(frame), 'subject_alignment_sha256': hashlib.sha256(pd.util.hash_array(subjects).tobytes()).hexdigest(), 'n_patients': len(frame), 'n_events': int(frame.y.sum()), 'split_seeds': list(SPLIT_SEEDS), 'model_seed': MODEL_SEED, 'test_size': TEST_SIZE, 'experiments': list(EXPERIMENTS), 'all_labels_loaded_for_stratification': True, 'original_holdout_patients_reincluded': True, 'hyperparameter_search': False, 'notes': ['Use the complete original CLIN cohort. The previous holdout was inspected; merging it supports exploratory split sensitivity only.', 'Change split seeds to change patient membership; hold model seed at 42. All recipes share each split.', 'All labels are used for stratification and scoring only; fitting receives training patients and training labels only.', 'Refit imputation, missingness filtering, LASSO and TF-IDF/SVD within each training split.', 'Retain inner LASSO selection and stacking cross-fitting; no base-model grid search.', 'Report every prespecified split without selecting the best seed. Splits overlap, so SD is not a 95% CI.']}

@contextlib.contextmanager
def run_lock(folder):
    """Prevent concurrent writes to one run signature; normal interruption releases the lock."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / '.running.lock'
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 384)
    except FileExistsError as error:
        raise RuntimeError(f'This run is active, or an abnormal exit left a lock: {path}. Confirm that no identical task is running before deleting the lock.') from error
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)

def aggregate(rows):
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby('experiment', sort=False).agg(n_splits=('repeat', 'nunique'), AUROC_mean=('AUROC', 'mean'), AUROC_sd=('AUROC', 'std'), AUROC_min=('AUROC', 'min'), AUROC_max=('AUROC', 'max'), AUPRC_mean=('AUPRC', 'mean'), AUPRC_sd=('AUPRC', 'std'), Brier_mean=('Brier', 'mean'), Brier_sd=('Brier', 'std'), cal_intercept_mean=('cal_intercept', 'mean'), cal_slope_mean=('cal_slope', 'mean')).reset_index()
    differences = []
    for new, ref in PAIRS:
        a = metrics[metrics.experiment.eq(new)].set_index(['repeat', 'split_seed'])
        b = metrics[metrics.experiment.eq(ref)].set_index(['repeat', 'split_seed'])
        paired = a.join(b, lsuffix='_new', rsuffix='_reference', how='inner')
        for (repeat, seed), row in paired.iterrows():
            for metric in METRICS:
                differences.append({'repeat': repeat, 'split_seed': seed, 'experiment': new, 'reference': ref, 'metric': metric, 'difference': row[metric + '_new'] - row[metric + '_reference']})
    paired = pd.DataFrame(differences)
    comparison_rows = []
    for (new, ref, metric), group in paired.groupby(['experiment', 'reference', 'metric'], sort=False):
        delta = group['difference']
        direction = -1 if metric == 'Brier' else 1
        comparison_rows.append({'experiment': new, 'reference': ref, 'metric': metric, 'n_splits': len(delta), 'difference_mean': delta.mean(), 'difference_sd': delta.std(ddof=1), 'difference_min': delta.min(), 'difference_max': delta.max(), 'n_better': int((direction * delta > 1e-12).sum()), 'n_equal': int((delta.abs() <= 1e-12).sum()), 'n_worse': int((direction * delta < -1e-12).sum())})
    return (metrics, summary, paired, pd.DataFrame(comparison_rows))

def notebook(source):
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python 3 (ml)', 'language': 'python', 'name': 'python3'}}, 'cells': [{'cell_type': 'markdown', 'id': 'repeated-holdout-results', 'metadata': {}, 'source': source.splitlines(keepends=True)}]}

def write_progress(output, rows, dcas, fits, protocol):
    metrics, summary, paired, comparisons = aggregate(rows)
    n_done = metrics.repeat.nunique()
    complete = n_done == len(SPLIT_SEEDS)
    metrics.to_csv(output / 'split_metrics.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    paired.to_csv(output / 'paired_differences_by_split.csv', index=False)
    comparisons.to_csv(output / 'paired_summary.csv', index=False)
    pd.concat(dcas, ignore_index=True).to_csv(output / 'dca_by_split.csv', index=False)
    fixed.atomic_json({'complete': complete, 'n_completed_splits': int(n_done), 'n_planned_splits': len(SPLIT_SEEDS), 'model_seed': MODEL_SEED, 'all_labels_loaded_for_stratification': True, 'fits': fits}, output / 'run_identity.json')
    statements = []
    for row in comparisons[comparisons.metric.eq('AUROC')].itertuples():
        statements.append(f'- {row.experiment} versus {row.reference}: mean AUROC difference {row.difference_mean:+.4f}, {row.n_better}/{row.n_splits} splits higher, {row.n_worse}/{row.n_splits} splits lower; descriptive stability only.')
    full = summary.set_index('experiment')
    if 'combined_stacking' in full.index:
        s = full.loc['combined_stacking']
        statements.append(f'- Current complete-extension AUROC mean: {s.AUROC_mean:.4f}, range {s.AUROC_min:.4f}–{s.AUROC_max:.4f}; do not select the highest split as the final score.')
    text = '\n\n'.join(['# Repeated holdout: split-sensitivity analysis', f'Completed {n_done}/{len(SPLIT_SEEDS)} splits; ' + ('complete.' if complete else 'partial results only; all five splits are not complete.'), f"Original CLIN cohort: {protocol['n_patients']} patients, {protocol['n_events']} deaths; split seeds {list(SPLIT_SEEDS)}, stratified 80/20 each time; model seed {MODEL_SEED}. Refit the same frozen recipes each time, without base-model hyperparameter search.", 'Refit imputation, the 20% variable missingness rule, LASSO and TF-IDF/SVD inside each training split. Refit pipelines inside the three stacking subfolds; train the meta-model on out-of-fold predictions. All labels are used only for stratification and matching scores, never to fit the corresponding test patients.', '## All split-level results', '```text\n' + metrics.round(6).to_string(index=False) + '\n```', '## Recipe means and split variation', '```text\n' + summary.round(6).to_string(index=False) + '\n```', 'SD and range describe variation across overlapping splits, not a 95% CI or five independent cohorts. Per-split AUROC_lo/hi are DeLong CIs for each fitted model, not a CI for the five-split average. Do not pool repeated predictions as if patients were independent.', '## Paired differences within the same split', '```text\n' + comparisons.round(6).to_string(index=False) + '\n```', 'Positive AUROC/AUPRC differences and negative Brier differences are favorable. Patients overlap between repetitions: do not use an ordinary paired t-test or mean +/- 1.96*SD/sqrt(5) on the five differences as independent observations.', '## Descriptive findings', '\n'.join(statements), 'This is exploratory internal validation after inspecting the original holdout. A five-split mean above 0.780 neither proves superiority to the paper nor excludes leakage. Keep reporting the original single-holdout AUROC near 0.756. No new untouched test patients remain.', 'Static feature definitions are unchanged. Availability of index-admission chronic diagnosis codes still relies on the original assumption; resplitting does not resolve this timing limitation. DCA uses prespecified 0.05-0.60 thresholds and is descriptive without CIs.']) + '\n'
    (output / 'report.md').write_text(text, encoding='utf-8')
    (output / 'repeated_holdout_results.ipynb').write_text(json.dumps(notebook(text), ensure_ascii=False, indent=2), encoding='utf-8')
    return (summary, comparisons, complete)

def run_analysis(frame, subjects, options, reference):
    validate_full_frame(frame, subjects)
    if fixed.SEED != MODEL_SEED or tuple(fixed.EXPERIMENTS) != tuple(EXPERIMENTS):
        raise ValueError('The shared fitting entry changed its fixed model seed or recipe list.')
    protocol = make_protocol(frame, subjects, options, reference)
    signature = fixed.sha_json(protocol)[:16]
    cache = ROOT / 'data/repeated_holdout' / signature
    output = ROOT / 'results/repeated_holdout' / signature
    rows, dcas, fits = ([], [], [])
    with run_lock(cache):
        output.mkdir(parents=True, exist_ok=True)
        fixed.atomic_json(protocol, output / 'protocol.json')
        (output.parent / 'active.txt').write_text(signature + '\n')
        for repeat, split_seed, tr, te in planned_splits(frame, subjects):
            train = frame.iloc[tr].reset_index(drop=True)
            test = frame.iloc[te].drop(columns=['y']).reset_index(drop=True)
            y_test = frame.y.iloc[te].to_numpy(int)
            split_identity = {'protocol_sha256': fixed.sha_json(protocol), 'repeat': repeat, 'split_seed': split_seed, 'train_rows_sha256': fingerprint(frame.iloc[tr][['stay_id', 'y']]), 'test_rows_sha256': fingerprint(frame.iloc[te][['stay_id', 'y']])}
            split_signature = fixed.sha_json(split_identity)[:16]
            split_cache = cache / f'repeat_{repeat}_seed_{split_seed}'
            print(f'\nSplit {repeat}/{len(SPLIT_SEEDS)}, seed={split_seed}: training {len(tr)} patients, test {len(te)} patients; model seed={MODEL_SEED}.', flush=True)
            predictions, fit_rows = fixed.fit_and_predict(train, test, options, split_cache, split_signature)
            fixed.atomic_json(split_identity, split_cache / 'split_identity.json')
            for name, indices in [('train', tr), ('test', te)]:
                pd.DataFrame({'stay_id': frame.stay_id.iloc[indices].to_numpy(), 'subject_id': subjects[indices]}).to_parquet(split_cache / f'{name}_ids.parquet', index=False)
            pd.DataFrame({'stay_id': test.stay_id, 'y': y_test, **predictions}).to_parquet(split_cache / 'predictions.parquet', index=False)
            for name, p in predictions.items():
                _, lo, hi = delong_auc_ci(y_test, p)
                rows.append({'repeat': repeat, 'split_seed': split_seed, 'model_seed': MODEL_SEED, 'experiment': name, 'n_train': len(tr), 'n_test': len(te), 'events_train': int(train.y.sum()), 'events_test': int(y_test.sum()), **fixed.fold_metrics(y_test, p), 'AUROC_lo': lo, 'AUROC_hi': hi})
                dca = decision_curve(y_test, p, thresholds=np.round(np.arange(0.05, 0.601, 0.01), 2))
                dcas.append(dca.assign(repeat=repeat, split_seed=split_seed, experiment=name))
            fits.extend(({'repeat': repeat, 'split_seed': split_seed, **row} for row in fit_rows))
            summary, comparisons, complete = write_progress(output, rows, dcas, fits, protocol)
            print(pd.DataFrame(rows)[lambda d: d['repeat'].eq(repeat)][['experiment', 'AUROC', 'AUPRC', 'Brier']].round(6).to_string(index=False), flush=True)
            print(f'Progress saved: {output}', flush=True)
        if complete:
            (output.parent / 'latest.txt').write_text(signature + '\n')
            print('\nAll five splits completed:', flush=True)
            print(summary.round(6).to_string(index=False), flush=True)
    return (output, summary, comparisons)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check-only', action='store_true', help='Check environment, original data and protocol; do not train or load original holdout labels')
    group.add_argument('--run', action='store_true', help='Explicitly train five splits, totaling fifteen outer recipe fits')
    args = parser.parse_args()
    options, reference, development, cohort, static = fixed.load_inputs()
    print(f'Checks passed: original development {len(development)} patients + original holdout {len(static)} patients.', flush=True)
    print(f'Prespecified split seeds: {list(SPLIT_SEEDS)}; fixed model seed {MODEL_SEED}.', flush=True)
    print('Three recipes: ' + ', '.join(EXPERIMENTS) + '; original holdout labels not loaded yet.', flush=True)
    if not args.run:
        print('Check only. To train: bash extensions/balanced/run_extensions.sh repeated-holdout --run', flush=True)
        return
    print('Start exploratory split sensitivity: load all labels for stratification; no new untouched test set remains.', flush=True)
    frame, subjects = build_full_frame(development, cohort, static, reference)
    output, _, _ = run_analysis(frame, subjects, options, reference)
    print(f'Results: {output}', flush=True)
if __name__ == '__main__':
    main()
