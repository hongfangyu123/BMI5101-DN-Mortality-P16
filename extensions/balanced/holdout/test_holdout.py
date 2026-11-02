"""Synthetic checks for training boundaries, frozen recipes, cached resume and paired differences."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
spec = importlib.util.spec_from_file_location('holdout_entry', Path(__file__).with_name('run_holdout.py'))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
from dn_extensions.evaluation import synthetic_frame

class HoldoutChecks(unittest.TestCase):

    def test_only_reviewed_relocated_source_is_accepted(self):
        manifest = json.loads((entry.ROOT / 'holdout/source_compatibility.json').read_text())
        reference = {'code_sha256': manifest['original_code_sha256']}
        self.assertTrue(entry.compatible_source(reference))
        with patch.object(entry, 'code_fingerprint', return_value='unreviewed-change'):
            self.assertFalse(entry.compatible_source(reference))
        self.assertFalse(entry.compatible_source({'code_sha256': 'unreviewed-original'}))

    def test_patient_overlap_rejected(self):
        train = pd.DataFrame({'subject_id': [1, 2], 'stay_id': [11, 12]})
        test = pd.DataFrame({'subject_id': [3, 4], 'stay_id': [13, 14]})
        entry.validate_subjects(train, test)
        with self.assertRaises(ValueError):
            entry.validate_subjects(train, test.assign(subject_id=[2, 4]))
        with self.assertRaises(ValueError):
            entry.validate_subjects(train, test.assign(stay_id=[12, 14]))

    def test_freeze_predict_before_labels_and_resume(self):
        data = synthetic_frame(120)
        development = data.iloc[:80].reset_index(drop=True)
        holdout = data.iloc[80:].drop(columns=['y']).reset_index(drop=True)
        y_test = data.iloc[80:].y.to_numpy(int)
        sequence = []
        options = entry.profile_options('smoke')
        reference = {'code_sha256': 'synthetic', 'software_versions': {}}
        real_assess = entry.assess_predictions
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            class FakeEstimator:

                def fit(self, X, y):
                    self_test.assertEqual(len(X), 80)
                    self_test.assertNotIn('y', X.columns)
                    self_test.assertNotIn('stay_id', X.columns)
                    np.testing.assert_array_equal(y, development.y)
                    protocols = list(root.glob('results/holdout/*/protocol.json'))
                    self_test.assertEqual(len(protocols), 1)
                    frozen = json.loads(protocols[0].read_text())
                    self_test.assertEqual(frozen['experiments'], list(entry.EXPERIMENTS))
                    sequence.append('fit')
                    return self

                def predict_proba(self, X):
                    self_test.assertEqual(len(X), 40)
                    self_test.assertNotIn('y', X.columns)
                    sequence.append('predict')
                    p = np.linspace(0.1, 0.9, len(X))
                    return np.column_stack([1 - p, p])

            def labels(ids):
                self.assertEqual(sequence.count('predict'), 3)
                sequence.append('labels')
                return y_test
            self_test = self
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(entry, 'ROOT', root))
                stack.enter_context(patch.object(entry, 'load_inputs', return_value=(options, reference, development, None, holdout[['stay_id']])))
                stack.enter_context(patch.object(entry, 'build_holdout_features', return_value=holdout))
                factory = stack.enter_context(patch.object(entry, 'make_estimator', side_effect=lambda *a: FakeEstimator()))
                stack.enter_context(patch.object(entry, 'fitted_metadata', return_value={'synthetic': True}))
                label_reader = stack.enter_context(patch.object(entry, 'load_holdout_labels', side_effect=labels))
                stack.enter_context(patch.object(entry, 'assess_predictions', side_effect=lambda y, p: real_assess(y, p, 20)))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                with patch.object(sys, 'argv', ['run_holdout.py', '--check-only']):
                    entry.main()
                factory.assert_not_called()
                label_reader.assert_not_called()
                self.assertFalse((root / 'results').exists())
                with patch.object(sys, 'argv', ['run_holdout.py', '--run']):
                    entry.main()
                self.assertEqual(factory.call_count, 3)
                self.assertEqual(sequence[-1], 'labels')
                factory.side_effect = AssertionError('Resume must not refit completed models')
                with patch.object(sys, 'argv', ['run_holdout.py', '--run']):
                    entry.main()
                out = next((root / 'results/holdout').glob('*/holdout_summary.csv'))
                self.assertEqual(len(pd.read_csv(out)), 3)
                self.assertTrue((out.parent / 'holdout_results.ipynb').exists())
                self.assertFalse({'stay_id', 'y', 'probability'} & set(pd.read_csv(out).columns))

    def test_identical_predictions_have_zero_paired_difference(self):
        y = np.tile([0, 1], 20)
        p = np.linspace(0.1, 0.9, len(y))
        predictions = {name: p.copy() for name in entry.EXPERIMENTS}
        summary, paired, dca = entry.assess_predictions(y, predictions, bootstrap_n=30)
        self.assertEqual(len(summary), 3)
        np.testing.assert_allclose(paired[['difference', 'lo', 'hi']], 0)
        self.assertEqual(len(dca), 56 * 3)
if __name__ == '__main__':
    unittest.main()
