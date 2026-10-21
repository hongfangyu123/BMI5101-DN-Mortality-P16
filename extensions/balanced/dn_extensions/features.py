"""提取开发集的时间轨迹与既往 ICD 文档；没有患者删除或全样本插补。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .settings import BURDEN, DATABASE, DOCUMENT, PROCESSED, ROOT, STATIC, V3_PROCESSED

# 复用 V3 官方 concept 表；白细胞分类使用绝对计数，体温使用已换算的摄氏度。
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
    """数据与列名进入缓存身份；不输出患者级内容。"""
    h = hashlib.sha256('|'.join(frame.columns).encode())
    h.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return h.hexdigest()


def load_development():
    """只读 train 的特征和结局；留出集只读 stay_id 用于互斥检查。"""
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
        raise ValueError(f'V3 缺少静态列：{sorted(missing)}')
    return train.reset_index(drop=True), feat, cohort


def event_query(concept, columns):
    if concept not in CONCEPTS or not set(columns) <= set(CONCEPTS[concept]):
        raise ValueError('未授权的 concept/变量')
    # 化验必须属于同次住院，生命体征必须属于同次 ICU；窗口右端包含 24 h。
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
    """先合并同一时刻的测量；至少两个不同时刻才定义 delta、SD 和斜率。"""
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
    """确切码 + 三级类目，不混淆 ICD9/10；没有既往诊断时文档为空。"""
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
                raise ValueError('ICD version 应为 9 或 10')
            token = f'v{int(version)}_{code}'
            exact.add(token)
            tokens.update([token, f'c{int(version)}_{code[:3]}'])
        by_id.loc[stay, DOCUMENT] = ' '.join(sorted(tokens))
        by_id.loc[stay, BURDEN] = [group.prior_hadm_id.nunique(), len(exact), len(dx), int(bool(exact))]
    return by_id.reset_index()


def build_features():
    raise RuntimeError('balanced 只读共享特征。请在上级原扩展目录执行 bash run_extensions.sh prepare。')
