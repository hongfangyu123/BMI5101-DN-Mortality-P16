"""Synthetic checks for deterministic splits, training boundaries, summaries and cached resume."""
import contextlib
import importlib.util
import io
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
spec = importlib.util.spec_from_file_location('repeated_holdout_entry', Path(__file__).with_name('run_repeated_holdout.py'))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
from dn_extensions.evaluation import synthetic_frame
from dn_extensions.settings import profile_options

class RepeatedHoldoutChecks(unittest.TestCase):

    def setUp(self):
        self.frame = synthetic_frame(120)
        self.subjects = np.arange(len(self.frame)) + 1000

    def test_split_reproducibility_and_patient_disjointness(self):
        splits = list(entry.planned_splits(self.frame, self.subjects))
        repeated = list(entry.planned_splits(self.frame, self.subjects))
        self.assertEqual(len(splits), 5)
        self.assertEqual([s[1] for s in splits], [101, 102, 103, 104, 105])
        self.assertEqual(len({tuple(sorted(s[3])) for s in splits}), 5)
        for (_, _, tr, te), (_, _, tr2, te2) in zip(splits, repeated):
            np.testing.assert_array_equal(tr, tr2)
            np.testing.assert_array_equal(te, te2)
            self.assertEqual((len(tr), len(te)), (96, 24))
            self.assertTrue(set(self.subjects[tr]).isdisjoint(self.subjects[te]))
            self.assertEqual(int(self.frame.y.iloc[te].sum()), 12)
        duplicate = self.subjects.copy()
        duplicate[1] = duplicate[0]
        with self.assertRaises(ValueError):
            list(entry.planned_splits(self.frame, duplicate))

    def test_only_check_does_not_train_or_read_full_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(entry, 'ROOT', Path(folder)), patch.object(entry.fixed, 'load_inputs', return_value=({}, {}, self.frame.iloc[:96], None, self.frame.iloc[96:][['stay_id']])), patch.object(entry, 'build_full_frame', side_effect=AssertionError('Check-only must not load full-cohort labels')) as builder:
                with patch.object(sys, 'argv', ['run_repeated_holdout.py', '--check-only']), contextlib.redirect_stdout(io.StringIO()):
                    entry.main()
                builder.assert_not_called()
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_each_split_fits_train_only_and_cache_is_checked(self):
        frame, subjects = (self.frame, self.subjects)
        splits = list(entry.planned_splits(frame, subjects))
        options = profile_options('smoke')
        reference = {'code_sha256': 'synthetic', 'software_versions': {}}
        observed = []
        test_case = self
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def factory(name, current_options, seed, memory):
                repeat_index = len(observed) // 3
                _, split_seed, tr, te = splits[repeat_index]
                expected = frame.iloc[tr].drop(columns=['stay_id', 'y']).reset_index(drop=True)
                expected_test = frame.iloc[te].drop(columns=['stay_id', 'y']).reset_index(drop=True)
                test_case.assertEqual(seed, 42)
                test_case.assertEqual(current_options, options)
                test_case.assertEqual(name, entry.EXPERIMENTS[len(observed) % 3])
                observed.append((split_seed, name))

                class FakeEstimator:

                    def fit(self, X, y):
                        pd.testing.assert_frame_equal(X, expected)
                        np.testing.assert_array_equal(y, frame.y.iloc[tr])
                        protocol_path = next(root.glob('results/repeated_holdout/*/protocol.json'))
                        frozen = json.loads(protocol_path.read_text())
                        test_case.assertEqual(frozen['split_seeds'], list(entry.SPLIT_SEEDS))
                        test_case.assertEqual(frozen['model_seed'], 42)
                        return self

                    def predict_proba(self, X):
                        pd.testing.assert_frame_equal(X, expected_test)
                        p = 1 / (1 + np.exp(-(X.age.to_numpy() - 60) / 10))
                        return np.column_stack([1 - p, p])
                return FakeEstimator()
            with patch.object(entry, 'ROOT', root), patch.object(entry.fixed, 'make_estimator', side_effect=factory) as maker, patch.object(entry.fixed, 'fitted_metadata', return_value={'synthetic': True}), contextlib.redirect_stdout(io.StringIO()):
                output, summary, comparisons = entry.run_analysis(frame, subjects, options, reference)
                self.assertEqual(maker.call_count, 15)
                self.assertTrue((summary.n_splits == 5).all())
                self.assertEqual(len(pd.read_csv(output / 'split_metrics.csv')), 15)
                np.testing.assert_allclose(comparisons.difference_mean, 0)
                self.assertTrue((comparisons.n_equal == 5).all())
                self.assertTrue((output / 'repeated_holdout_results.ipynb').exists())
                self.assertTrue(json.loads((output / 'run_identity.json').read_text())['complete'])
                for path in output.glob('*.csv'):
                    self.assertFalse({'subject_id', 'stay_id', 'y', 'probability'} & set(pd.read_csv(path).columns))
                maker.side_effect = AssertionError('Completed five-split predictions must not be retrained')
                _, again, _ = entry.run_analysis(frame, subjects, options, reference)
                pd.testing.assert_frame_equal(summary, again)
                cache_file = next(root.glob('data/repeated_holdout/*/repeat_1_seed_101/static_lr.pkl'))
                record = pickle.loads(cache_file.read_bytes())
                record['signature'] = 'different-split'
                cache_file.write_bytes(pickle.dumps(record))
                with self.assertRaises(ValueError):
                    entry.run_analysis(frame, subjects, options, reference)
                self.assertFalse(list(root.glob('data/repeated_holdout/*/.running.lock')))
if __name__ == '__main__':
    unittest.main()
