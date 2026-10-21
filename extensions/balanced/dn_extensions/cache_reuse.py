"""显式复用已审查版本的预测；数据、环境、训练算法和每折索引必须一致。"""
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
    # 注册表只增加实验名，不改变旧模型的训练算法；其余 AST 必须完全一致。
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
    """仅接受本次代码扩展前记录的版本，不因“文件存在”而盲目复用。"""
    if not re.fullmatch(r'[0-9a-f]{16}', signature):
        raise ValueError('--reuse-from 必须为16位缓存签名')
    manifest = json.loads((ROOT / 'cache_compatibility.json').read_text(encoding='utf-8'))
    if signature != manifest['source_signature']:
        raise ValueError('此缓存版本没有经过本次兼容性审查，请去掉 --reuse-from 重新计算')
    source_output = ROOT / 'results' / identity['options']['profile'] / signature
    source = ROOT / 'data/cache' / identity['options']['profile'] / signature
    previous = json.loads((source_output / 'run_identity.json').read_text(encoding='utf-8'))
    previous_signature = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()[:16]
    if previous_signature != signature or previous['code_sha256'] != manifest['source_code_sha256']:
        raise ValueError('原缓存运行身份与已审查版本不符')
    # code_sha256 可以因新增实验/报告逻辑变化；数据、选项和软件环境不能变化。
    for key in set(previous) | set(identity):
        if key != 'code_sha256' and previous.get(key) != identity.get(key):
            raise ValueError(f'不能复用原缓存：运行身份 {key} 已变化')
    if training_fingerprint() != manifest['training_fingerprint']:
        raise ValueError('不能复用原缓存：训练代码/特征/配置已经改变')
    for name, specification in manifest['experiments'].items():
        if list(EXPERIMENTS.get(name, ())) != specification:
            raise ValueError(f'旧实验定义已改变：{name}')
    y = frame.y.to_numpy(int)
    splits = {(repeat, fold): (len(tr), va, seed)
              for repeat, seed in enumerate(identity['options']['seeds'])
              for fold, (tr, va) in enumerate(StratifiedKFold(
                  identity['options']['outer_folds'], shuffle=True, random_state=seed).split(frame, y))}
    validated = []
    for path in sorted(source.glob('r*_f*_*.pkl')):
        with path.open('rb') as stream:
            record = pickle.load(stream)  # 仅读取自己的已验证本地缓存。
        name = record['experiment']
        if name not in manifest['experiments']:
            raise ValueError(f'原缓存包含未经审查的实验：{name}')
        key = record['repeat'], record['fold']
        if key not in splits:
            raise ValueError('原缓存 repeat/fold 超出本次划分')
        n_train, va, seed = splits[key]
        expected = f'r{key[0]}_f{key[1]}_{name}.pkl'
        probability = np.asarray(record['probability'])
        if (path.name != expected or record['seed'] != seed or record['n_train'] != n_train
                or record['n_test'] != len(va) or not np.array_equal(record['test_indices'], va)
                or probability.shape != (len(va),) or not np.isfinite(probability).all()
                or not ((probability >= 0) & (probability <= 1)).all()):
            raise ValueError(f'原缓存内容或验证折不匹配：{path.name}')
        target = target_cache / path.name
        if target.exists():
            with target.open('rb') as stream:
                current = pickle.load(stream)
            if (not np.array_equal(current['test_indices'], va)
                    or not np.array_equal(current['probability'], probability)
                    or current['experiment'] != name or current['seed'] != seed):
                raise ValueError(f'目标缓存与原缓存冲突：{path.name}')
        validated.append((path, target))
    if not validated:
        raise ValueError('原缓存目录没有可复用的预测')
    print(f'已验证旧缓存 {signature}：{len(validated)} 个折；划分、数据、参数和训练算法一致。', flush=True)
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
                  'note': '原预测复制到新签名；旧代码与数据目录保留。预处理的 joblib 缓存可按相同输入复用。'}
    (output / 'cache_reuse.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    return source / 'pipeline_cache'
