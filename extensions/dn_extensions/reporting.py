"""把每次真实开发集实验的状态与结论同步写回对应 Notebook。"""
from __future__ import annotations

import json

import nbformat
import pandas as pd

from .settings import ROOT

NOTEBOOK_EXPERIMENTS = {
    '02_stacking.ipynb': ['static_lr', 'static_rf', 'static_xgb', 'static_stacking'],
    '03_temporal_trajectories.ipynb': ['static_lr', 'temporal_lr'],
    '04_comorbidity_embedding.ipynb': ['static_lr', 'burden_lr', 'embedding_lr'],
    '05_combined_extensions.ipynb': ['static_stacking', 'combined_stacking'],
}


def update_notebooks(output, options):
    summary = pd.read_csv(output / 'summary.csv')
    paired_path = output / 'paired_comparisons.csv'
    try:
        paired = pd.read_csv(paired_path)
    except pd.errors.EmptyDataError:
        paired = pd.DataFrame()
    for name, experiments in NOTEBOOK_EXPERIMENTS.items():
        path = ROOT / 'notebooks' / name
        if not path.exists():
            continue
        selected = summary[summary.experiment.isin(experiments)]
        if selected.empty or not any(e != 'static_lr' and e in set(selected.experiment) for e in experiments):
            continue
        nb = nbformat.read(path, as_version=4)
        text = ['## 实验状态与结论（自动更新）', f'运行模式：`{options["profile"]}`；结果目录：`{output.relative_to(ROOT)}`。',
                '固定留出集未参与本实验。', '```text', selected.round(5).to_string(index=False), '```']
        if len(paired):
            comparisons = paired[paired.experiment.isin(experiments)]
            text += ['配对差异：', '```text', comparisons.round(5).to_string(index=False), '```']
            for row in comparisons[comparisons.metric == 'AUROC'].itertuples():
                interpretation = ('有内部验证提升证据' if row.lo > 0 else
                                  '有内部验证下降证据' if row.hi < 0 else '尚无明确的提升或下降证据')
                text.append(f'- `{row.experiment}` 相对 `{row.reference}`：ΔAUROC={row.mean_diff:.4f}，'
                            f'校正95% CI [{row.lo:.4f}, {row.hi:.4f}]；{interpretation}。'
                            '还需查看 Holm 校正 p 值、AUPRC、校准和 DCA。')
        if options['profile'] != 'full':
            text.append('**本次只是 pilot 小规模试跑，不能写成正式研究结论。**')
        cell = next((c for c in nb.cells if c.get('metadata', {}).get('extension_results')), None)
        if cell is None:
            cell = nbformat.v4.new_markdown_cell()
            cell.metadata['extension_results'] = True
            nb.cells.append(cell)
        cell.source = '\n\n'.join(text)
        nbformat.write(nb, path)


def update_preparation_notebook(manifest):
    path = ROOT / 'notebooks/01_prepare_features.ipynb'
    if not path.exists():
        return
    nb = nbformat.read(path, as_version=4)
    cell = next(c for c in nb.cells if c.get('metadata', {}).get('extension_results'))
    cell.source = ('## 实验状态与结论（自动更新）\n\n'
                   f'特征准备完成：开发集 {manifest["n_development"]} 人，'
                   f'{manifest["events_development"]} 例死亡，{len(manifest["temporal"])} 个时间轨迹候选。\n\n'
                   '此步没有拟合插补、词表、SVD 或模型，也没有读取留出集标签。'
                   '描述性缺失率不能直接用作全样本特征筛选；筛选发生在后续训练折内。')
    nbformat.write(nb, path)
