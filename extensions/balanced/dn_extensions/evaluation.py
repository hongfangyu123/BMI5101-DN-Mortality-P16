"""同一开发集、同一外层折的配对比较；真实留出集不参与扩展选择。"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests

from .features import fingerprint, load_development
from .modeling import EXPERIMENTS, fitted_metadata, make_estimator
from .settings import BURDEN, DOCUMENT, PROCESSED, ROOT, STATIC, V3_ROOT
from src.metrics import cal_intercept_slope_checked, decision_curve, nadeau_bengio


def code_fingerprint():
    h = hashlib.sha256()
    sources = sorted((ROOT / 'dn_extensions').glob('*.py')) + sorted((ROOT / 'scripts').glob('*.py'))
    sources += [ROOT / 'config.yaml', V3_ROOT / 'src/lasso.py', V3_ROOT / 'src/variables.py', V3_ROOT / 'src/metrics.py']
    for p in sources:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def load_extension_frame():
    path = PROCESSED / 'development_features.parquet'
    if not path.exists():
        raise FileNotFoundError('请先运行 bash run_extensions.sh prepare。')
    frame = pd.read_parquet(path)
    _, original, _ = load_development()
    columns = ['stay_id', 'y'] + STATIC
    if fingerprint(frame[columns]) != fingerprint(original[columns]):
        raise ValueError('V3 开发集/静态特征已改变；请重新运行 prepare，不复用旧特征。')
    return frame


def synthetic_frame(n=120):
    """冒烟测试只用合成数据；其中的数字不是任何 DN 模型性能。"""
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(n, len(STATIC))), columns=STATIC)
    X['wbc_diff_missing'] = 0.0
    X['age'] = rng.normal(60, 10, n)
    X['trajectory_rdw_delta'] = rng.normal(size=n)
    X['trajectory_rdw_slope_per_hour'] = X.trajectory_rdw_delta / 24
    for c in BURDEN:
        X[c] = rng.integers(0, 4, n)
    X[DOCUMENT] = [f'v10_E119 c10_E11 v10_{["I10", "N18", "I50"][i % 3]}' if i % 4 else '' for i in range(n)]
    X['stay_id'] = np.arange(n)
    X['y'] = np.tile([0, 1], n // 2)
    X.loc[X.index[::7], 'calcium_min'] = np.nan
    return X


def atomic_pickle(obj, target):
    tmp = target.with_suffix('.tmp')
    with tmp.open('wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, target)


def fold_metrics(y, probability):
    cal = cal_intercept_slope_checked(y, probability)
    return {'AUROC': roc_auc_score(y, probability), 'AUPRC': average_precision_score(y, probability),
            'Brier': brier_score_loss(y, probability), 'cal_intercept': cal['cal_intercept'],
            'cal_slope': cal['cal_slope'], 'cal_intercept_ok': cal['cal_intercept_ok'],
            'cal_slope_ok': cal['cal_slope_ok']}


def summarize(records, y, options, output):
    metric_rows, pooled, dcas = [], [], []
    for record in records:
        metric_rows.append({k: record[k] for k in ['experiment', 'repeat', 'seed', 'fold', 'n_train', 'n_test', 'seconds']} |
                           fold_metrics(y[record['test_indices']], record['probability']))
    folds = pd.DataFrame(metric_rows)
    folds.to_csv(output / 'fold_metrics.csv', index=False)
    for experiment in sorted(folds.experiment.unique()):
        for repeat, seed in enumerate(options['seeds']):
            subset = [r for r in records if r['experiment'] == experiment and r['repeat'] == repeat]
            if len(subset) != options['outer_folds']:
                continue
            p = np.full(len(y), np.nan)
            counts = np.zeros(len(y), int)
            for rec in subset:
                p[rec['test_indices']] = rec['probability']
                counts[rec['test_indices']] += 1
            assert (counts == 1).all() and np.isfinite(p).all()
            pooled.append({'experiment': experiment, 'repeat': repeat, 'seed': seed, **fold_metrics(y, p)})
            dca = decision_curve(y, p, thresholds=np.round(np.arange(0.05, 0.601, 0.01), 2))
            dcas.append(dca.assign(experiment=experiment, repeat=repeat, seed=seed))
    pooled_frame = pd.DataFrame(pooled)
    pooled_frame.to_csv(output / 'pooled_oof_metrics.csv', index=False)
    if dcas:
        pd.concat(dcas, ignore_index=True).to_csv(output / 'pooled_dca.csv', index=False)
    summary = folds.groupby('experiment').agg(
        n_outer_folds=('AUROC', 'size'), AUROC_fold_mean=('AUROC', 'mean'), AUROC_fold_sd=('AUROC', 'std'),
        AUPRC_fold_mean=('AUPRC', 'mean'), Brier_fold_mean=('Brier', 'mean'),
        calibration_slope_fold_mean=('cal_slope', 'mean'), seconds_total=('seconds', 'sum'),
    ).reset_index()
    if len(pooled_frame):
        pooled_summary = pooled_frame.groupby('experiment')[['AUROC', 'AUPRC', 'Brier', 'cal_intercept', 'cal_slope']].mean()
        pooled_summary.columns = [c + '_pooled_mean' for c in pooled_summary.columns]
        summary = summary.merge(pooled_summary, on='experiment', how='left')
    summary.to_csv(output / 'summary.csv', index=False)
    references = {'static_rf': 'static_lr', 'static_xgb': 'static_lr', 'static_stacking': 'static_lr',
                  'temporal_lr': 'static_lr', 'burden_lr': 'static_lr', 'embedding_lr': 'burden_lr',
                  'combined_stacking': 'static_stacking',
                  'temporal_rf': 'static_rf', 'temporal_xgb': 'static_xgb',
                  'burden_rf': 'static_rf', 'burden_xgb': 'static_xgb',
                  'embedding_rf': 'burden_rf', 'embedding_xgb': 'burden_xgb'}
    comparisons = []
    # embedding 的主比较控制住诊断负担；另报相对静态模型的总增益。
    reference_pairs = list(references.items()) + [
        ('embedding_rf', 'static_rf'), ('embedding_xgb', 'static_xgb')]
    for exp, ref in reference_pairs:
        if exp not in set(folds.experiment) or ref not in set(folds.experiment):
            continue
        a = folds[folds.experiment == exp].set_index(['repeat', 'fold'])
        b = folds[folds.experiment == ref].set_index(['repeat', 'fold'])
        pair = a.join(b, lsuffix='_extension', rsuffix='_reference', how='inner')
        if len(pair) != len(options['seeds']) * options['outer_folds']:
            continue
        for metric in ['AUROC', 'AUPRC', 'Brier']:
            difference = pair[metric + '_extension'] - pair[metric + '_reference']
            test = nadeau_bengio(difference.to_numpy(), pair.n_train_extension.mean(), pair.n_test_extension.mean(),
                                k=options['outer_folds'], r=len(options['seeds']))
            comparisons.append({'experiment': exp, 'reference': ref, 'metric': metric, **test})
    paired = pd.DataFrame(comparisons)
    if len(paired):
        paired['p_holm_current_family'] = np.nan
        for metric in paired.metric.unique():
            ix = paired.index[(paired.metric == metric) & paired.p_value.notna()]
            if len(ix):
                paired.loc[ix, 'p_holm_current_family'] = multipletests(paired.loc[ix, 'p_value'], method='holm')[1]
    paired.to_csv(output / 'paired_comparisons.csv', index=False)
    # 每折的变量选择/超参数为汇总信息；概率和患者标识只保存在 data/。
    metadata = [{k: rec[k] for k in ['experiment', 'repeat', 'seed', 'fold', 'fit_metadata']} for rec in records]
    (output / 'fit_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary, paired


def run_evaluation(options, experiments, frame=None, *, reuse_from=None, check_only=False):
    synthetic = options['profile'] == 'smoke'
    frame = synthetic_frame() if synthetic else load_extension_frame() if frame is None else frame
    y = frame.y.to_numpy(int)
    X = frame.drop(columns=['stay_id', 'y'])
    unknown = set(experiments) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f'未知实验：{sorted(unknown)}')
    experiments = list(dict.fromkeys(['static_lr'] + experiments))
    # 自动补齐同类型静态模型，以及 embedding 的诊断负担对照；不会添加 stacking。
    for suffix in ('lr', 'rf', 'xgb'):
        if any(f'{variant}_{suffix}' in experiments for variant in ('temporal', 'burden', 'embedding')):
            experiments.append(f'static_{suffix}')
        if f'embedding_{suffix}' in experiments:
            experiments.append(f'burden_{suffix}')
    if 'combined_stacking' in experiments:
        experiments.append('static_stacking')
    experiments = list(dict.fromkeys(experiments))
    software = {name: metadata.version(name) for name in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'duckdb', 'joblib']}
    identity = {'options': options, 'data_sha256': fingerprint(frame), 'code_sha256': code_fingerprint(),
                'platform': sys.platform, 'python': sys.version.split()[0], 'software_versions': software,
                'n_development': len(frame), 'events_development': int(y.sum()),
                'synthetic': synthetic, 'holdout_labels_loaded': False}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cache = ROOT / 'data/cache' / options['profile'] / signature
    output = ROOT / 'results' / options['profile'] / signature
    pipeline_cache = cache / 'pipeline_cache'
    if reuse_from:
        if synthetic:
            raise ValueError('合成测试不能导入真实患者预测')
        from .cache_reuse import reuse_approved_cache
        pipeline_cache = reuse_approved_cache(reuse_from, identity, frame, cache, output, check_only=check_only)
    if check_only:
        print(f'仅检查，不训练：{options["profile"]}；实验={experiments}；新签名={signature}', flush=True)
        return None, None, None
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_identity.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding='utf-8')
    folds_list = {}
    for repeat, seed in enumerate(options['seeds']):
        for fold, (tr, va) in enumerate(StratifiedKFold(options['outer_folds'], shuffle=True, random_state=seed).split(X, y)):
            folds_list[(repeat, fold)] = va
            assert set(frame.stay_id.iloc[tr]).isdisjoint(frame.stay_id.iloc[va])
            for experiment in experiments:
                target = cache / f'r{repeat}_f{fold}_{experiment}.pkl'
                if target.exists():
                    with target.open('rb') as f:
                        rec = pickle.load(f)
                    assert np.array_equal(rec['test_indices'], va), '缓存外层折不同'
                    print(f'缓存：rep {repeat + 1} fold {fold + 1} {experiment}', flush=True)
                    continue
                start = time.time()
                est = make_estimator(experiment, options, seed, pipeline_cache)
                est.fit(X.iloc[tr], y[tr])
                probability = est.predict_proba(X.iloc[va])[:, 1]
                if not np.isfinite(probability).all():
                    raise ValueError('预测概率存在非有限值')
                rec = {'experiment': experiment, 'repeat': repeat, 'seed': seed, 'fold': fold,
                       'n_train': len(tr), 'n_test': len(va), 'test_indices': va, 'probability': probability,
                       'seconds': time.time() - start, 'fit_metadata': fitted_metadata(est)}
                atomic_pickle(rec, target)
                print(f'rep {repeat + 1}/{len(options["seeds"])} fold {fold + 1}/{options["outer_folds"]} '
                      f'{experiment}: 完成 ({rec["seconds"]:.1f}s)', flush=True)
    # 同一签名目录可分次执行不同扩展，最后一起做配对比较。
    records = []
    for p in sorted(cache.glob('r*_f*_*.pkl')):
        with p.open('rb') as f:
            rec = pickle.load(f)
        assert np.array_equal(rec['test_indices'], folds_list[(rec['repeat'], rec['fold'])])
        records.append(rec)
    summary, paired = summarize(records, y, options, output)
    (output.parent / 'latest.txt').write_text(signature + '\n')
    if not synthetic:
        from .reporting import update_notebooks
        update_notebooks(output, options)
    report = ['# V3 扩展开发集实验', f'profile={options["profile"]}；合成数据={synthetic}；留出集标签未读取。',
              'balanced：固定基模型参数；保留折内 Pipeline 与 λ.1se，LASSO 5折/30λ、stacking 3折。与严格版分开记录。',
              '这些属于开发集探索性比较，不能声称已由独立留出集确认。',
              '树模型扩展保持原固定参数；embedding vs burden 检验潜在表示增益，embedding vs static 检验总增益。',
              '## 同折模型比较', '```text', summary.round(6).to_string(index=False), '```',
              '## 配对差异（扩展减去参考模型）', '```text', paired.round(6).to_string(index=False), '```',
              'AUROC/AUPRC 差值为正更好，Brier 差值为负更好。CI 为 Nadeau–Bengio 校正区间；'
              'p_holm_current_family 对当前已有实验按每个指标作 Holm 多重比较校正。',
              'pilot 只用于检查流程和耗时；smoke 仅使用合成数据，不能作为研究结果。']
    (output / 'report.md').write_text('\n\n'.join(report) + '\n', encoding='utf-8')
    print(f'结果写入：{output}', flush=True)
    return output, summary, paired
