"""仅合成数据：验证新实验的对照、续跑与缓存拒绝逻辑，不训练患者模型。"""
import hashlib
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dn_extensions import cache_reuse, evaluation
from dn_extensions.modeling import make_estimator
from dn_extensions.settings import profile_options

NEW = ['temporal_rf', 'temporal_xgb', 'embedding_rf', 'embedding_xgb']


class TreeAblationChecks(unittest.TestCase):
    def test_same_model_parameters_and_representation(self):
        options = profile_options('full')
        for model in ('rf', 'xgb'):
            static = make_estimator(f'static_{model}', options, 42, '/tmp/not-fitted')
            for variant in ('temporal', 'burden', 'embedding'):
                pipeline = make_estimator(f'{variant}_{model}', options, 42, '/tmp/not-fitted')
                self.assertEqual(static.named_steps['model'].get_params(), pipeline.named_steps['model'].get_params())
                reference = make_estimator(f'{variant}_lr', options, 42, '/tmp/not-fitted')
                self.assertEqual(pipeline.named_steps['representation'].get_params(),
                                 reference.named_steps['representation'].get_params())

    def test_synthetic_comparisons_and_resume_without_stacking(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(evaluation, 'ROOT', Path(folder)), patch.object(evaluation, 'code_fingerprint', return_value='synthetic-test'):
                output, summary, pairs = evaluation.run_evaluation(profile_options('smoke'), NEW)
                self.assertEqual(set(summary.experiment), set(NEW + ['static_lr', 'static_rf', 'static_xgb', 'burden_rf', 'burden_xgb']))
                self.assertTrue((summary.n_outer_folds == 2).all())
                observed = set(zip(pairs.experiment, pairs.reference))
                for model in ('rf', 'xgb'):
                    self.assertIn((f'temporal_{model}', f'static_{model}'), observed)
                    self.assertIn((f'embedding_{model}', f'burden_{model}'), observed)
                    self.assertIn((f'embedding_{model}', f'static_{model}'), observed)
                self.assertFalse(summary.experiment.str.contains('stacking').any())
                with patch.object(evaluation, 'make_estimator', side_effect=AssertionError('续跑不应拟合')):
                    _, again, _ = evaluation.run_evaluation(profile_options('smoke'), NEW)
                pd.testing.assert_frame_equal(summary, again)
                self.assertTrue((output / 'pooled_dca.csv').exists())

    def test_old_cache_is_validated_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = pd.DataFrame({'y': [0, 0, 1, 1]})
            options = {'profile': 'full', 'seeds': [42], 'outer_folds': 2}
            previous = {'options': options, 'data_sha256': 'same-data', 'code_sha256': 'reviewed-code',
                        'software_versions': {'example': '1'}, 'synthetic': False}
            signature = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()[:16]
            old_output = root / 'results/full' / signature
            old_cache = root / 'data/cache/full' / signature
            old_output.mkdir(parents=True); old_cache.mkdir(parents=True)
            (old_output / 'run_identity.json').write_text(json.dumps(previous))
            manifest = {'source_signature': signature, 'source_code_sha256': 'reviewed-code',
                        'training_fingerprint': {'training': 'unchanged'}, 'experiments': {'static_rf': ['static', 'RF']}}
            (root / 'cache_compatibility.json').write_text(json.dumps(manifest))
            tr, va = next(StratifiedKFold(2, shuffle=True, random_state=42).split(frame, frame.y))
            record = {'experiment': 'static_rf', 'repeat': 0, 'fold': 0, 'seed': 42,
                      'n_train': len(tr), 'n_test': len(va), 'test_indices': va, 'probability': np.array([.2, .8])}
            name = 'r0_f0_static_rf.pkl'
            (old_cache / name).write_bytes(pickle.dumps(record))
            identity = previous | {'code_sha256': 'extended-code'}
            target = root / 'new-cache'; output = root / 'new-output'
            with patch.object(cache_reuse, 'ROOT', root), patch.object(cache_reuse, 'training_fingerprint', return_value={'training': 'unchanged'}):
                cache_reuse.reuse_approved_cache(signature, identity, frame, target, output, check_only=True)
                self.assertFalse(target.exists())
                for key, value in [('data_sha256', 'changed'), ('software_versions', {'example': '2'})]:
                    with self.assertRaises(ValueError):
                        cache_reuse.reuse_approved_cache(signature, identity | {key: value}, frame, target, output)
                cache_reuse.reuse_approved_cache(signature, identity, frame, target, output)
                self.assertEqual((old_cache / name).read_bytes(), (target / name).read_bytes())
                record['probability'] = np.array([.3, .8])
                (target / name).write_bytes(pickle.dumps(record))
                with self.assertRaises(ValueError):
                    cache_reuse.reuse_approved_cache(signature, identity, frame, target, output)
            with patch.object(cache_reuse, 'ROOT', root), patch.object(cache_reuse, 'training_fingerprint', return_value={'training': 'changed'}):
                with self.assertRaises(ValueError):
                    cache_reuse.reuse_approved_cache(signature, identity, frame, target, output, check_only=True)


if __name__ == '__main__':
    unittest.main()
