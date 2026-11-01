"""Write the status and conclusions of each real development-set experiment back into the corresponding notebook."""
from __future__ import annotations

import json

import nbformat
import pandas as pd

from .settings import ROOT

NOTEBOOK_EXPERIMENTS = {
    '02_stacking.ipynb': ['static_lr', 'static_rf', 'static_xgb', 'static_stacking'],
    '03_temporal_trajectories.ipynb': ['static_lr', 'temporal_lr', 'static_rf', 'temporal_rf', 'static_xgb', 'temporal_xgb'],
    '04_comorbidity_embedding.ipynb': ['static_lr', 'burden_lr', 'embedding_lr', 'static_rf', 'burden_rf', 'embedding_rf', 'static_xgb', 'burden_xgb', 'embedding_xgb'],
    '05_combined_extensions.ipynb': ['static_stacking', 'combined_stacking'],
    '06_tree_feature_ablations.ipynb': ['static_rf', 'static_xgb', 'temporal_rf', 'temporal_xgb', 'burden_rf', 'burden_xgb', 'embedding_rf', 'embedding_xgb'],
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
        text = ['## Experiment status and conclusions (updated automatically)', f'Run mode: `{options["profile"]}`; method: balanced fixed parameters; result directory: `{output.relative_to(ROOT)}`.',
                f'LASSO {options["inner_lasso_folds"]} folds/{options["lasso_lambdas"]} λ, stacking {options["stack_folds"]} folds; static and extension models use the same simplified method. No folds of the strict version were mixed in.',
                'The fixed held-out set took no part in this experiment. Tree-model extensions need no internal stacking cross-fitting; the original fixed parameters and within-training-fold preprocessing are unchanged.',
                'The Holm correction covers the paired comparisons currently available; corrected p-values may change when experiments are added later.', '```text', selected.round(5).to_string(index=False), '```']
        if len(paired):
            comparisons = paired[paired.experiment.isin(experiments)]
            text += ['Paired differences:', '```text', comparisons.round(5).to_string(index=False), '```']
            for row in comparisons[comparisons.metric == 'AUROC'].itertuples():
                interpretation = ('internal evidence of improvement after this round of multiple-comparison correction' if row.lo > 0 and row.p_holm_current_family < 0.05 else
                                  'internal evidence of deterioration after this round of multiple-comparison correction' if row.hi < 0 and row.p_holm_current_family < 0.05 else
                                  'no evidence of a difference confirmed after this round of multiple-comparison correction')
                text.append(f'- `{row.experiment}` vs `{row.reference}`: ΔAUROC={row.mean_diff:.4f}, '
                            f'corrected 95% CI [{row.lo:.4f}, {row.hi:.4f}]; {interpretation}. '
                            'The Holm-corrected p-value, AUPRC, calibration and DCA should also be checked.')
        if options['profile'] != 'full':
            text.append('**This is only a small pilot run and must not be written up as a formal research conclusion.**')
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
    cell.source = ('## Experiment status and conclusions (updated automatically)\n\n'
                   f'Feature preparation completed: development set {manifest["n_development"]} patients, '
                   f'{manifest["events_development"]} deaths, {len(manifest["temporal"])} temporal trajectory candidates.\n\n'
                   'This step fitted no imputation, vocabulary, SVD or model, and did not read the held-out labels. '
                   'Descriptive missing rates must not be used directly for whole-sample feature screening; screening happens later within the training folds.')
    nbformat.write(nb, path)
