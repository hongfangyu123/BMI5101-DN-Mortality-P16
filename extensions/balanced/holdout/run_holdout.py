"""Supplementary evaluation of three frozen recipes on the original V3 holdout; check only by default."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from importlib import metadata
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from dn_extensions.evaluation import atomic_pickle, code_fingerprint, fold_metrics, load_extension_frame
from dn_extensions.features import CONCEPTS, PRIOR_QUERY, extract_temporal, fingerprint, history_documents, load_development
from dn_extensions.modeling import fitted_metadata, make_estimator
from dn_extensions.settings import DATABASE, STATIC, V3_PROCESSED, profile_options
from src.metrics import decision_curve, delong_auc_ci
SOURCE_SIGNATURE = '27464ba98305edbc'
EXPERIMENTS = ('static_lr', 'static_stacking', 'combined_stacking')
PAIRS = (('static_stacking', 'static_lr'), ('combined_stacking', 'static_stacking'), ('combined_stacking', 'static_lr'))
SEED = 42
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261023

def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def atomic_json(value, target):
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, target)

def validate_subjects(development, holdout):
    """Check patient-level separation so different ICU stays of one patient cannot cross datasets."""
    if not development.subject_id.is_unique or not holdout.subject_id.is_unique:
        raise ValueError('The original CLIN cohort must contain one record per patient.')
    if not set(development.subject_id).isdisjoint(holdout.subject_id):
        raise ValueError('Development and holdout datasets share patients.')
    if not set(development.stay_id).isdisjoint(holdout.stay_id):
        raise ValueError('Development and holdout datasets share ICU stays.')


def compatible_source(reference):
    """Accept exact original bytes or this explicitly reviewed English/relocated source."""
    current = code_fingerprint()
    if current == reference['code_sha256']:
        return True
    manifest = json.loads((ROOT / 'holdout/source_compatibility.json').read_text())
    if (reference['code_sha256'] != manifest['original_code_sha256']
            or current != manifest['reviewed_repository_code_sha256']):
        return False
    for relative, expected in manifest['required_file_sha256'].items():
        if hashlib.sha256((ROOT.parents[1] / relative).read_bytes()).hexdigest() != expected:
            return False
    return True

def load_inputs():
    """Load development labels only; explicitly exclude y from the held-out feature columns."""
    reference_dir = ROOT / 'results/full' / SOURCE_SIGNATURE
    reference = json.loads((reference_dir / 'run_identity.json').read_text())
    options = profile_options('full')
    versions = {name: metadata.version(name) for name in reference['software_versions']}
    development = load_extension_frame()
    if options != reference['options'] or not compatible_source(reference) or fingerprint(development) != reference['data_sha256'] or (versions != reference['software_versions']):
        raise ValueError('Parameters, source, development data or software differ from the reviewed balanced experiment. Use the pinned environment.')
    completed = pd.read_csv(reference_dir / 'summary.csv').set_index('experiment')
    if any((completed.loc[name, 'n_outer_folds'] != 50 for name in EXPERIMENTS)):
        raise ValueError('Development CV has not completed for all three prespecified recipes.')
    _, _, development_cohort = load_development()
    holdout_ids = pd.read_parquet(V3_PROCESSED / 'split_clin.parquet', columns=['stay_id'], filters=[('split', '==', 'valid')])
    if not holdout_ids.stay_id.is_unique or len(holdout_ids) != 512:
        raise ValueError('The original holdout must contain 512 distinct stays; do not resplit it.')
    ids = holdout_ids.stay_id.tolist()
    cohort = pd.read_parquet(V3_PROCESSED / 'cohort_clin.parquet', columns=['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'outtime'], filters=[('stay_id', 'in', ids)])
    static = pd.read_parquet(V3_PROCESSED / 'features_clin.parquet', columns=['stay_id'] + STATIC, filters=[('stay_id', 'in', ids)])
    if not cohort.stay_id.is_unique or not static.stay_id.is_unique:
        raise ValueError('Duplicate stays in the held-out cohort or static features.')
    cohort = cohort.set_index('stay_id').loc[ids].reset_index()
    static = static.set_index('stay_id').loc[ids].reset_index()
    validate_subjects(development_cohort, cohort)
    if not (cohort.outtime > cohort.intime + pd.Timedelta(hours=24)).all():
        raise ValueError('Held-out patients do not satisfy the original 24-hour landmark.')
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        for concept, variables in CONCEPTS.items():
            names = set(con.execute(f'DESCRIBE mimiciv_derived.{concept}').df().column_name)
            if not set(variables) <= names:
                raise ValueError(f'{concept} is missing trajectory columns.')
        con.execute('SELECT subject_id, hadm_id, dischtime FROM mimiciv_hosp.admissions LIMIT 0')
        con.execute('SELECT subject_id, hadm_id, icd_version, icd_code FROM mimiciv_hosp.diagnoses_icd LIMIT 0')
    return (options, reference, development, cohort, static)

def build_holdout_features(cohort, static, development):
    """Reuse identical time windows and diagnosis history; fit no statistics on held-out patients."""
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        con.execute('SET threads=2')
        temporal = extract_temporal(con, cohort)
        history = history_documents(con.execute(PRIOR_QUERY).df(), cohort.stay_id)
    frame = static.merge(temporal, on='stay_id', validate='one_to_one')
    frame = frame.merge(history, on='stay_id', validate='one_to_one')
    expected = [c for c in development if c != 'y']
    if set(frame.columns) != set(expected) or frame.stay_id.tolist() != static.stay_id.tolist():
        raise ValueError('Held-out and development raw feature schemas or row order differ.')
    return frame[expected]

def make_protocol(options, reference, development, holdout):
    return {'analysis': 'supplementary_evaluation_on_previously_used_V3_holdout', 'source_cv_signature': SOURCE_SIGNATURE, 'source_code_sha256': reference['code_sha256'], 'holdout_entry_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'development_sha256': fingerprint(development), 'holdout_features_sha256': fingerprint(holdout), 'software_versions': reference['software_versions'], 'options': options, 'experiments': list(EXPERIMENTS), 'seed': SEED, 'n_development': len(development), 'events_development': int(development.y.sum()), 'n_holdout': len(holdout), 'bootstrap_n': BOOTSTRAP_N, 'bootstrap_seed': BOOTSTRAP_SEED, 'classification_threshold': None, 'note': 'Refit on the entire development set with seed=42; this is not an average of five fitted predictions. Do not select models, hyperparameters or thresholds, or recalibrate, using holdout outcomes. The original V3 holdout was already inspected; this is supplementary internal evaluation, not new external validation.'}

def fit_and_predict(development, holdout, options, cache, signature):
    """Fit all models and predict before receiving holdout labels. Resume completed model predictions."""
    import pickle
    X = development.drop(columns=['stay_id', 'y'])
    Xt = holdout.drop(columns=['stay_id'])
    if list(X.columns) != list(Xt.columns):
        raise ValueError('Development and holdout input column orders differ.')
    y = development.y.to_numpy(int)
    predictions, fit_rows = ({}, [])
    cache.mkdir(parents=True, exist_ok=True)
    for name in EXPERIMENTS:
        target = cache / f'{name}.pkl'
        if target.exists():
            with target.open('rb') as f:
                record = pickle.load(f)
            if record['signature'] != signature:
                raise ValueError('Held-out prediction cache does not match the frozen protocol.')
            print(f'Reuse completed model: {name}', flush=True)
        else:
            print(f'Start: {name}, training on all {len(X)} development patients.', flush=True)
            start = time.time()
            estimator = make_estimator(name, options, SEED, cache / 'pipeline_cache')
            estimator.fit(X, y)
            probability = estimator.predict_proba(Xt)[:, 1]
            record = {'signature': signature, 'experiment': name, 'probability': probability, 'seconds': time.time() - start, 'fit_metadata': fitted_metadata(estimator)}
            validate_probability(record['probability'], len(Xt))
            atomic_pickle(record, target)
            print(f"Finished: {name} ({record['seconds']:.1f}s)", flush=True)
            del estimator
        validate_probability(record['probability'], len(Xt))
        predictions[name] = record['probability']
        fit_rows.append({k: record[k] for k in ('experiment', 'seconds', 'fit_metadata')})
    return (predictions, fit_rows)

def validate_probability(p, n):
    p = np.asarray(p)
    if p.shape != (n,) or not np.isfinite(p).all() or (not ((p >= 0) & (p <= 1)).all()):
        raise ValueError('Invalid held-out probability shape or values.')

def load_holdout_labels(ids):
    """Read labels in the fixed stay order only after every model has finished predicting."""
    labels = pd.read_parquet(V3_PROCESSED / 'split_clin.parquet', columns=['stay_id', 'y'], filters=[('split', '==', 'valid')])
    if not labels.stay_id.is_unique or set(labels.stay_id) != set(ids):
        raise ValueError('The set of stays associated with held-out outcomes has changed.')
    y = labels.set_index('stay_id').loc[list(ids), 'y'].to_numpy()
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Held-out labels must be binary and contain survivors and deaths.')
    return y.astype(int)

def assess_predictions(y, predictions, bootstrap_n=BOOTSTRAP_N):
    """Resample the same patients together across models to preserve pairing."""
    metric_functions = {'AUROC': roc_auc_score, 'AUPRC': average_precision_score, 'Brier': brier_score_loss}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pos, neg = (np.flatnonzero(y == 1), np.flatnonzero(y == 0))
    boot = {name: {m: [] for m in metric_functions} for name in predictions}
    for _ in range(bootstrap_n):
        ix = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        for name, p in predictions.items():
            for metric, fun in metric_functions.items():
                boot[name][metric].append(fun(y[ix], p[ix]))
    rows, differences, dcas = ([], [], [])
    for name, p in predictions.items():
        point = fold_metrics(y, p)
        _, lo, hi = delong_auc_ci(y, p)
        row = {'experiment': name, 'n_holdout': len(y), 'events_holdout': int(y.sum()), 'seed': SEED, **point, 'AUROC_lo': lo, 'AUROC_hi': hi}
        for metric in ('AUPRC', 'Brier'):
            row[metric + '_lo'], row[metric + '_hi'] = np.quantile(boot[name][metric], [0.025, 0.975])
        rows.append(row)
        dcas.append(decision_curve(y, p, np.round(np.arange(0.05, 0.601, 0.01), 2)).assign(experiment=name))
    summary = pd.DataFrame(rows)
    indexed = summary.set_index('experiment')
    for new, ref in PAIRS:
        if new not in predictions or ref not in predictions:
            continue
        for metric in metric_functions:
            delta = np.asarray(boot[new][metric]) - np.asarray(boot[ref][metric])
            lo, hi = np.quantile(delta, [0.025, 0.975])
            differences.append({'experiment': new, 'reference': ref, 'metric': metric, 'difference': indexed.loc[new, metric] - indexed.loc[ref, metric], 'lo': lo, 'hi': hi, 'multiplicity_adjusted': False})
    return (summary, pd.DataFrame(differences), pd.concat(dcas, ignore_index=True))

def write_outputs(output, summary, paired, dca, protocol):
    summary.to_csv(output / 'holdout_summary.csv', index=False)
    paired.to_csv(output / 'holdout_paired_differences.csv', index=False)
    dca.to_csv(output / 'holdout_dca.csv', index=False)
    report = '\n\n'.join(['# Supplementary evaluation on the original V3 holdout', 'Train on all 2,045 development patients and evaluate on the original 512 held-out patients, with seed=42 for each recipe. These are neither 5x10-fold CV means nor external validation. This holdout was previously reported by V3 and recipes arose from development exploration, so the evaluation is supplementary rather than a new untouched confirmation.', 'Freeze recipes before loading holdout labels. Missingness filtering, imputation, scaling, LASSO, TF-IDF and SVD fit on development data only, including inside stacking subfolds. The holdout is transformed and predicted only; no holdout-based tuning, threshold selection or recalibration.', '## Held-out results', '```text\n' + summary.round(6).to_string(index=False) + '\n```', f"AUROC uses DeLong 95% CIs. AUPRC/Brier and paired differences use {protocol['bootstrap_n']} stratified patient bootstrap resamples with fixed event prevalence. Intervals quantify sampling uncertainty for the fitted models, not variation from refitting the training pipeline.", '## Paired differences: extension minus reference', '```text\n' + paired.round(6).to_string(index=False) + '\n```', 'Difference intervals are not multiplicity adjusted. Positive AUROC/AUPRC differences and negative Brier differences are favorable. DCA at the prespecified 0.05-0.60 thresholds is descriptive, without CIs. One holdout does not establish stable gains; also report development CV.']) + '\n'
    (output / 'report.md').write_text(report, encoding='utf-8')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python 3 (ml)', 'language': 'python', 'name': 'python3'}}, 'cells': [{'cell_type': 'markdown', 'id': 'holdout-results', 'metadata': {}, 'source': report.splitlines(keepends=True)}]}
    (output / 'holdout_results.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Check splits, settings, environment and schemas without reading holdout labels')
    parser.add_argument('--run', action='store_true', help='Train the three frozen recipes and perform supplementary holdout evaluation')
    args = parser.parse_args()
    if args.run and args.check_only:
        parser.error('--run and --check-only are mutually exclusive')
    options, reference, development, cohort, static = load_inputs()
    print(f'Checks passed: development {len(development)} patients; original holdout {len(static)} patients; holdout labels not loaded.', flush=True)
    print('Frozen recipes: ' + ', '.join(EXPERIMENTS) + '; seed=42; original balanced settings.', flush=True)
    if not args.run:
        print('Check only. To train: bash extensions/balanced/run_extensions.sh holdout --run', flush=True)
        return
    print('Extract 0-24h trajectories and diagnoses from admissions completed before index admission; fit no statistics.', flush=True)
    holdout = build_holdout_features(cohort, static, development)
    protocol = make_protocol(options, reference, development, holdout)
    signature = sha_json(protocol)[:16]
    cache = ROOT / 'data/holdout' / signature
    output = ROOT / 'results/holdout' / signature
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(protocol, output / 'protocol.json')
    predictions, fits = fit_and_predict(development, holdout, options, cache, signature)
    atomic_json({'protocol_sha256': sha_json(protocol), 'all_predictions_ready': True, 'holdout_labels_loaded': False}, output / 'predictions_ready.json')
    print('All three recipes have finished predicting; now load holdout labels and calculate metrics.', flush=True)
    y = load_holdout_labels(holdout.stay_id)
    pd.DataFrame({'stay_id': holdout.stay_id, 'y': y, **predictions}).to_parquet(cache / 'predictions.parquet', index=False)
    summary, paired, dca = assess_predictions(y, predictions)
    atomic_json({'protocol_sha256': sha_json(protocol), 'holdout_labels_loaded': True, 'events_holdout': int(y.sum()), 'fits': fits}, output / 'run_identity.json')
    write_outputs(output, summary, paired, dca, protocol)
    (output.parent / 'latest.txt').write_text(signature + '\n')
    print(summary.round(6).to_string(index=False), flush=True)
    print(f'Results: {output}', flush=True)
if __name__ == '__main__':
    main()
