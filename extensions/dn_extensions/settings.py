from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
V3_ROOT = Path(os.environ.get('BMI5101_V3_ROOT', ROOT / CONFIG['source_project'])).resolve()
if not (V3_ROOT / 'src/lasso.py').is_file():
    raise FileNotFoundError(f'找不到 V3 项目：{V3_ROOT}；可设置 BMI5101_V3_ROOT。')
sys.path.insert(0, str(V3_ROOT))
from src.variables import FEATURES, MODE_VARS

STATIC = FEATURES + ['wbc_diff_missing']
WBC_EXEMPT = ['lymphocytes_mean', 'neutrophils_mean', 'monocytes_mean',
              'nlr', 'plr', 'lmr', 'pnr', 'sii']
BURDEN = ['prior_admission_count', 'prior_distinct_icd_count',
          'prior_icd_event_count', 'has_prior_diagnosis']
DOCUMENT = 'prior_icd_document'
PROCESSED = ROOT / 'data/processed'
V3_PROCESSED = V3_ROOT / 'data/processed'
DATABASE = V3_ROOT / 'data/interim/mimic_subset.duckdb'


def profile_options(profile='full'):
    options = dict(CONFIG)
    if profile == 'pilot':
        options.update(seeds=[42], outer_folds=3, inner_tune_folds=3,
                       inner_lasso_folds=3, lasso_lambdas=30, stack_folds=3)
    elif profile == 'smoke':
        options.update(seeds=[42], outer_folds=2, inner_tune_folds=2,
                       inner_lasso_folds=2, lasso_lambdas=8, stack_folds=2,
                       embedding_components=4, embedding_min_df=2, jobs=1)
    elif profile != 'full':
        raise ValueError(profile)
    options['profile'] = profile
    return options
