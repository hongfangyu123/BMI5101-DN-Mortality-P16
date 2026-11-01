"""Explicit reuse of predictions from a reviewed version; data, environment, training algorithm and per-fold indices must match."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .modeling import EXPERIMENTS
from .settings import ROOT, V3_ROOT


def training_fingerprint():
    # The registry only adds experiment names and does not change the training algorithm of the old models; the rest of the AST must be identical.
    tree = ast.parse((ROOT / 'dn_extensions/modeling.py').read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EXPERIMENTS'
                                            for t in node.targets))]
    hashes = {'modeling_training_ast': hashlib.sha256(ast.dump(tree).encode()).hexdigest()}
    for name, path in {
        'features': ROOT / 'dn_extensions/features.py',
        'settings': ROOT / 'dn_extensions/settings.py',
        'config': ROOT / 'config.yaml',
        'lasso': V3_ROOT / 'src/lasso.py',
        'variables': V3_ROOT / 'src/variables.py',
    }.items():
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def reuse_approved_cache(signature, identity, frame, target_cache, output, *, check_only=False):
    """Accept only the version recorded before this code extension; never reuse blindly just because "the file exists"."""
    if not re.fullmatch(r'[0-9a-f]{16}', signature):
        raise ValueError('--reuse-from must be a 16-character cache signature')
    manifest = json.loads((ROOT / 'cache_compatibility.json').read_text(encoding='utf-8'))
    if signature != manifest['source_signature']:
        raise ValueError('this cache version has not passed this compatibility review; drop --reuse-from and recompute')
    source_output = ROOT / 'results' / identity['options']['profile'] / signature
    source = ROOT / 'data/cache' / identity['options']['profile'] / signature
    previous = json.loads((source_output / 'run_identity.json').read_text(encoding='utf-8'))
    previous_signature = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()[:16]
    if previous_signature != signature or previous['code_sha256'] != manifest['source_code_sha256']:
        raise ValueError('the run identity of the source cache does not match the reviewed version')
    # code_sha256 may change because of new experiments/report logic; data, options and software environment must not change.
    for key in set(previous) | set(identity):
        if key != 'code_sha256' and previous.get(key) != identity.get(key):
            raise ValueError(f'cannot reuse the source cache: run identity {key} has changed')
    if training_fingerprint() != manifest['training_fingerprint']:
        raise ValueError('cannot reuse the source cache: training code/features/configuration have changed')
    for name, specification in manifest['experiments'].items():
        if list(EXPERIMENTS.get(name, ())) != specification:
            raise ValueError(f'old experiment definition has changed: {name}')
    y = frame.y.to_numpy(int)
    splits = {(repeat, fold): (len(tr), va, seed)
              for repeat, seed in enumerate(identity['options']['seeds'])
              for fold, (tr, va) in enumerate(StratifiedKFold(
                  identity['options']['outer_folds'], shuffle=True, random_state=seed).split(frame, y))}
    validated = []
    for path in sorted(source.glob('r*_f*_*.pkl')):
        with path.open('rb') as stream:
            record = pickle.load(stream)  # Reads only our own verified local cache.
        name = record['experiment']
        if name not in manifest['experiments']:
            raise ValueError(f'the source cache contains an unreviewed experiment: {name}')
        key = record['repeat'], record['fold']
        if key not in splits:
            raise ValueError('source cache repeat/fold is outside the current split')
        n_train, va, seed = splits[key]
        expected = f'r{key[0]}_f{key[1]}_{name}.pkl'
        probability = np.asarray(record['probability'])
        if (path.name != expected or record['seed'] != seed or record['n_train'] != n_train
                or record['n_test'] != len(va) or not np.array_equal(record['test_indices'], va)
                or probability.shape != (len(va),) or not np.isfinite(probability).all()
                or not ((probability >= 0) & (probability <= 1)).all()):
            raise ValueError(f'source cache content or validation fold does not match: {path.name}')
        target = target_cache / path.name
        if target.exists():
            with target.open('rb') as stream:
                current = pickle.load(stream)
            if (not np.array_equal(current['test_indices'], va)
                    or not np.array_equal(current['probability'], probability)
                    or current['experiment'] != name or current['seed'] != seed):
                raise ValueError(f'target cache conflicts with the source cache: {path.name}')
        validated.append((path, target))
    if not validated:
        raise ValueError('the source cache directory has no reusable predictions')
    print(f'verified old cache {signature}: {len(validated)} folds; split, data, parameters and training algorithm match.', flush=True)
    if check_only:
        return None
    target_cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    for path, target in validated:
        if not target.exists():
            temporary = target.with_suffix('.import.tmp')
            shutil.copy2(path, temporary)
            os.replace(temporary, target)
    provenance = {'source_signature': signature, 'source_code_sha256': previous['code_sha256'],
                  'target_code_sha256': identity['code_sha256'], 'folds_imported_or_verified': len(validated),
                  'training_fingerprint': training_fingerprint(),
                  'note': 'source predictions copied to the new signature; old code and data directories kept. The preprocessing joblib cache can be reused for identical inputs.'}
    (output / 'cache_reuse.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    return source / 'pipeline_cache'
