"""不运行患者数据模型；合成测试时间边界、折内变换和八组模型的接线。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import duckdb
import numpy as np
import pandas as pd

from dn_extensions.evaluation import run_evaluation, synthetic_frame
from dn_extensions.features import PRIOR_QUERY, event_query, history_documents, trajectory_statistics
from dn_extensions.modeling import EXPERIMENTS, FoldRepresentation, make_estimator
from dn_extensions.settings import DOCUMENT, profile_options


def temporal_checks():
    s = trajectory_statistics([0, 0, 12, 24, 25], [1, 3, 4, 6, 999])
    assert s['count'] == 3 and s['delta'] == 4
    assert np.isclose(s['slope_per_hour'], 1/6)
    assert s['late_minus_early'] == 3
    assert np.isnan(trajectory_statistics([0], [1])['slope_per_hour'])
    cohort = pd.DataFrame({'subject_id': [1], 'hadm_id': [10], 'stay_id': [100],
                           'intime': [pd.Timestamp('2020-01-01')],
                           'admittime': [pd.Timestamp('2019-12-31')]})
    with duckdb.connect() as con:
        con.register('extension_cohort', cohort)
        con.execute('CREATE SCHEMA mimiciv_derived')
        con.execute('CREATE TABLE mimiciv_derived.chemistry(subject_id INTEGER,hadm_id INTEGER,charttime TIMESTAMP,glucose DOUBLE)')
        con.execute("INSERT INTO mimiciv_derived.chemistry VALUES (1,10,'2020-01-01',1),(1,10,'2020-01-02',2),(1,10,'2020-01-02 01:00',999),(1,11,'2020-01-01 12:00',999),(1,10,'2019-12-31 23:00',999)")
        events = con.execute(event_query('chemistry', ['glucose'])).df()
        assert len(events) == 2 and events.hour.tolist() == [0, 24]
        con.execute('CREATE SCHEMA mimiciv_hosp')
        con.execute('CREATE TABLE mimiciv_hosp.admissions(subject_id INTEGER,hadm_id INTEGER,dischtime TIMESTAMP)')
        con.execute("INSERT INTO mimiciv_hosp.admissions VALUES (1,9,'2019-12-30'),(1,10,'2020-01-04'),(1,11,'2020-02-01')")
        con.execute('CREATE TABLE mimiciv_hosp.diagnoses_icd(subject_id INTEGER,hadm_id INTEGER,icd_version INTEGER,icd_code VARCHAR)')
        con.execute("INSERT INTO mimiciv_hosp.diagnoses_icd VALUES (1,9,10,'E11.9'),(1,10,10,'I10'),(1,11,10,'Z999')")
        history = history_documents(con.execute(PRIOR_QUERY).df(), [100, 101])
        assert history.loc[0, DOCUMENT] == 'c10_E11 v10_E119'
        assert history.loc[1, DOCUMENT] == ''
        assert 'Z999' not in history.loc[0, DOCUMENT] and 'I10' not in history.loc[0, DOCUMENT]


def transformer_checks(options):
    frame = synthetic_frame(120).drop(columns=['stay_id', 'y'])
    y = np.tile([0, 1], 60)
    rep = FoldRepresentation('embedding', options, seed=42).fit(frame.iloc[:80], y[:80])
    saved_fill = rep.fill_.copy()
    saved_vocabulary = dict(rep.vectorizer_.vocabulary_)
    altered = frame.iloc[80:].copy()
    altered['age'] = 1e10
    altered[DOCUMENT] = 'VALIDATION_ONLY_TOKEN'
    transformed = rep.transform(altered)
    pd.testing.assert_series_equal(saved_fill, rep.fill_)
    assert saved_vocabulary == rep.vectorizer_.vocabulary_
    assert 'VALIDATION_ONLY_TOKEN' not in rep.vectorizer_.vocabulary_
    assert np.isfinite(transformed).all()
    for document in ['', 'single_token']:
        edge = frame.iloc[:40].copy()
        edge[DOCUMENT] = document
        edge_rep = FoldRepresentation('embedding', options, seed=42).fit(edge, y[:40])
        assert np.isfinite(edge_rep.transform(edge)).all()


def main():
    options = profile_options('smoke')
    temporal_checks()
    transformer_checks(options)
    pipelines = [make_estimator(name, options, 42, Path('/tmp/bmi-balanced-structure-cache'))
                 for name in ['static_lr', 'static_rf', 'static_xgb']]
    assert all(type(p).__name__ == 'Pipeline' for p in pipelines)
    representations = [p.named_steps['representation'].get_params() for p in pipelines]
    assert representations[0] == representations[1] == representations[2]
    stack = make_estimator('static_stacking', options, 42, Path('/tmp/bmi-balanced-structure-cache'))
    assert all(type(p).__name__ == 'Pipeline' for _, p in stack.estimators)
    output, summary, _ = run_evaluation(options, list(EXPERIMENTS))
    assert len(summary) == len(EXPERIMENTS)
    assert (summary.n_outer_folds == 2).all()
    assert summary.AUROC_fold_mean.between(0, 1).all()
    assert output.parent.name == 'smoke'
    print('所有轻量检查通过；模型评估只用了120行合成数据，没有训练真实患者模型。')


if __name__ == '__main__':
    main()
