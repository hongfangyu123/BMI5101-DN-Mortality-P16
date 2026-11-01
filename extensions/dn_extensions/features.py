"""Extract development-set temporal trajectories and prior ICD documents; no patient deletion or whole-sample imputation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .settings import BURDEN, DATABASE, DOCUMENT, PROCESSED, ROOT, STATIC, V3_PROCESSED

# Reuse the official V3 concept tables; the white-cell differential uses absolute counts and temperature uses the converted Celsius values.
CONCEPTS = {
    'chemistry': ['glucose', 'aniongap', 'bicarbonate', 'creatinine', 'bun',
                  'calcium', 'chloride', 'sodium', 'potassium'],
    'complete_blood_count': ['hemoglobin', 'platelet', 'wbc', 'rbc', 'rdw'],
    'blood_differential': ['lymphocytes_abs', 'neutrophils_abs', 'monocytes_abs'],
    'coagulation': ['ptt', 'pt', 'inr'],
    'vitalsign': ['heart_rate', 'mbp', 'resp_rate', 'temperature', 'spo2'],
}
STATISTICS = ['delta', 'slope_per_hour', 'std', 'late_minus_early', 'count']


def fingerprint(frame):
    """Data and column names enter the cache identity; no patient-level content is output."""
    h = hashlib.sha256('|'.join(frame.columns).encode())
    h.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return h.hexdigest()


def load_development():
    """Read only the features and outcomes of train; for the held-out set only stay_id is read, to check disjointness."""
    split_path = V3_PROCESSED / 'split_clin.parquet'
    train = pd.read_parquet(split_path, filters=[('split', '==', 'train')])
    valid_ids = pd.read_parquet(split_path, columns=['stay_id'], filters=[('split', '==', 'valid')])
    ids = train.stay_id.tolist()
    feat = pd.read_parquet(V3_PROCESSED / 'features_clin.parquet', filters=[('stay_id', 'in', ids)])
    cohort = pd.read_parquet(V3_PROCESSED / 'cohort_clin.parquet',
                             columns=['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'outtime'],
                             filters=[('stay_id', 'in', ids)])
    assert train.stay_id.is_unique and feat.stay_id.is_unique and cohort.subject_id.is_unique
    assert set(ids).isdisjoint(valid_ids.stay_id)
    feat = feat.set_index('stay_id').loc[ids].reset_index()
    cohort = cohort.set_index('stay_id').loc[ids].reset_index()
    assert np.array_equal(feat.y.to_numpy(), train.y.to_numpy())
    assert (cohort.outtime > cohort.intime + pd.Timedelta(hours=24)).all()
    missing = set(STATIC) - set(feat.columns)
    if missing:
        raise ValueError(f'V3 is missing static columns: {sorted(missing)}')
    return train.reset_index(drop=True), feat, cohort


def event_query(concept, columns):
    if concept not in CONCEPTS or not set(columns) <= set(CONCEPTS[concept]):
        raise ValueError('unauthorised concept/variable')
    # Labs must belong to the same hospitalization and vital signs to the same ICU stay; the right end of the window includes 24 h.
    key = 'x.stay_id = c.stay_id' if concept == 'vitalsign' else 'x.hadm_id = c.hadm_id'
    return f'''
      SELECT c.stay_id, x.charttime,
             date_diff('second', c.intime, x.charttime) / 3600.0 AS hour,
             {', '.join('x.' + col for col in columns)}
      FROM mimiciv_derived.{concept} x JOIN extension_cohort c
        ON x.subject_id = c.subject_id AND {key}
       AND x.charttime >= c.intime
       AND x.charttime <= c.intime + INTERVAL 24 HOUR
      ORDER BY c.stay_id, x.charttime
    '''


def trajectory_statistics(hours, values):
    """First merge measurements at the same time point; delta, SD and slope are defined only with at least two distinct time points."""
    d = pd.DataFrame({'hour': hours, 'value': values}).replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d.hour >= 0) & (d.hour <= 24)]
    d = d.groupby('hour', sort=True).value.mean().reset_index()
    n = len(d)
    out = dict.fromkeys(STATISTICS, np.nan)
    out['count'] = n
    if n >= 2:
        t, v = d.hour.to_numpy(), d.value.to_numpy()
        out['delta'] = float(v[-1] - v[0])
        out['std'] = float(v.std(ddof=1))
        tc = t - t.mean()
        out['slope_per_hour'] = float(np.dot(tc, v - v.mean()) / np.dot(tc, tc))
    early, late = d.loc[d.hour < 12, 'value'], d.loc[d.hour >= 12, 'value']
    if len(early) and len(late):
        out['late_minus_early'] = float(late.mean() - early.mean())
    return out


def extract_temporal(con, cohort):
    con.register('extension_cohort', cohort)
    out = pd.DataFrame(index=pd.Index(cohort.stay_id, name='stay_id'))
    for concept, columns in CONCEPTS.items():
        events = con.execute(event_query(concept, columns)).df()
        groups = list(events.groupby('stay_id', sort=False))
        for variable in columns:
            names = [f'trajectory_{variable}_{stat}' for stat in STATISTICS]
            block = pd.DataFrame(np.nan, index=out.index, columns=names)
            block[names[-1]] = 0.0
            for stay, group in groups:
                stats = trajectory_statistics(group.hour, group[variable])
                block.loc[stay, names] = [stats[s] for s in STATISTICS]
            out = out.join(block)
    return out.reset_index()


PRIOR_QUERY = '''
  SELECT c.stay_id, a.hadm_id AS prior_hadm_id, d.icd_version,
         upper(replace(trim(d.icd_code), '.', '')) AS code
  FROM extension_cohort c JOIN mimiciv_hosp.admissions a
    ON a.subject_id = c.subject_id AND a.hadm_id != c.hadm_id
   AND a.dischtime < c.admittime
  LEFT JOIN mimiciv_hosp.diagnoses_icd d
    ON d.subject_id = a.subject_id AND d.hadm_id = a.hadm_id
'''


def history_documents(rows, stays):
    """Exact code + three-character category, without mixing ICD9/10; the document is empty when there are no prior diagnoses."""
    out = pd.DataFrame({'stay_id': list(stays)})
    out[DOCUMENT] = ''
    for col in BURDEN:
        out[col] = 0
    by_id = out.set_index('stay_id')
    for stay, group in rows.groupby('stay_id', sort=False):
        dx = group.dropna(subset=['code', 'icd_version']).copy()
        dx = dx[dx.code.ne('')].drop_duplicates(['prior_hadm_id', 'icd_version', 'code'])
        tokens = set()
        exact = set()
        for version, code in zip(dx.icd_version, dx.code):
            if int(version) not in (9, 10):
                raise ValueError('ICD version must be 9 or 10')
            token = f'v{int(version)}_{code}'
            exact.add(token)
            tokens.update([token, f'c{int(version)}_{code[:3]}'])
        by_id.loc[stay, DOCUMENT] = ' '.join(sorted(tokens))
        by_id.loc[stay, BURDEN] = [group.prior_hadm_id.nunique(), len(exact), len(dx), int(bool(exact))]
    return by_id.reset_index()


def build_features():
    train, static, cohort = load_development()
    if not DATABASE.exists():
        raise FileNotFoundError(f'complete the V3 cohort/feature pipeline first: {DATABASE}')
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        con.execute('SET threads=2')
        temporal = extract_temporal(con, cohort)
        history = history_documents(con.execute(PRIOR_QUERY).df(), cohort.stay_id)
    frame = static[['stay_id', 'y'] + STATIC].merge(temporal, on='stay_id', validate='one_to_one')
    frame = frame.merge(history, on='stay_id', validate='one_to_one')
    assert frame.stay_id.tolist() == train.stay_id.tolist()
    assert frame.y.tolist() == train.y.tolist()
    manifest = {
        'n_development': len(frame), 'events_development': int(frame.y.sum()),
        'static': STATIC, 'temporal': [c for c in frame if c.startswith('trajectory_')],
        'burden': BURDEN, 'document': DOCUMENT, 'data_sha256': fingerprint(frame),
        'source': str(V3_PROCESSED), 'holdout_labels_loaded': False,
        'temporal_window': 'ICU intime <= charttime <= intime + 24h; same hadm_id/stay_id',
        'history_window': 'previous admission dischtime < index admittime',
        'note': 'trajectories only 0–24h; V3 static columns keep the original -6–24h window; no vocabulary/imputation/selection fitted.',
    }
    tmp = PROCESSED / 'development_features.tmp.parquet'
    frame.to_parquet(tmp, index=False)
    tmp.replace(PROCESSED / 'development_features.parquet')
    (PROCESSED / 'feature_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    table_dir = ROOT / 'results/features'
    table_dir.mkdir(parents=True, exist_ok=True)
    frame[STATIC + manifest['temporal'] + BURDEN].isna().mean().rename('missing_fraction').to_csv(table_dir / 'missingness_descriptive.csv')
    (table_dir / 'feature_summary.json').write_text(json.dumps({k: manifest[k] for k in
        ['n_development', 'events_development', 'holdout_labels_loaded', 'temporal_window', 'history_window']},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'development-set features prepared: {len(frame)} patients, {int(frame.y.sum())} deaths; {len(manifest["temporal"])} trajectory candidates.')
    return manifest
