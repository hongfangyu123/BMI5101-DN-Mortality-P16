"""Data-free checks for a source/aggregate-only release. Never prints data rows or secret contents.

Adapted from the extension author's release check for this repository: only files that git would publish are checked
(tracked files plus the staged index, via `git ls-files --cached`), because the local working tree legitimately holds
patient-level data under the git-ignored data/ directories. Detection rules are unchanged: restricted data/cache paths,
forbidden suffixes, single-patient force plots (fig6c–e), raw/credential files, patient-level field names in CSV
headers and JSON keys, notebook outputs and credential patterns. PDFs are allowed only under paper/ and
results/figures/.
"""
from pathlib import Path
import csv
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_FIELDS = {'subject_id','subject_ids','hadm_id','hadm_ids','stay_id','stay_ids','patient_id','patient_ids',
                    'test_indices','train_indices','probability','probabilities','y_true','y_pred','prior_icd_document',
                    'admittime','dischtime','intime','outtime','deathtime','dod'}
BAD_SUFFIXES = {'.parquet','.duckdb','.pkl','.pickle','.joblib','.feather','.arrow','.sqlite','.npz','.npy','.h5','.hdf5','.pt','.pth','.pem','.key',
                '.log','.zip','.gz'}
PDF_ALLOWED = ('paper/', 'results/figures/')
SECRET_PATTERNS = [re.compile(r'gh[pousr]_[A-Za-z0-9]{25,}'), re.compile(r'github_pat_[A-Za-z0-9_]{30,}'),
                   re.compile(r'sk-[A-Za-z0-9_-]{30,}'), re.compile(r'AKIA[A-Z0-9]{16}'),
                   re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')]


def bad_json(x):
    if isinstance(x,dict): return any(str(k).lower() in FORBIDDEN_FIELDS or bad_json(v) for k,v in x.items())
    if isinstance(x,list): return any(bad_json(v) for v in x)
    return False


def release_files():
    out = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '--cached', '-z'], capture_output=True, check=True).stdout
    return sorted({Path(p) for p in out.decode().split('\0') if p})


def main():
    errors=[]; files=0; notebooks=0
    for rel in release_files():
        p=ROOT/rel
        if not p.is_file():          # staged deletion of a file already removed from the working tree
            continue
        files+=1
        if 'data' in rel.parts or 'pipeline_cache' in rel.parts or 'nested_cache' in rel.parts or 'logs' in rel.parts:
            errors.append(f'restricted local data/cache/log path: {rel}')
        if p.suffix.lower() in BAD_SUFFIXES or p.name.startswith(('fig6c_','fig6d_','fig6e_')):
            errors.append(f'forbidden release file: {rel}')
        if p.suffix.lower()=='.pdf' and not rel.as_posix().startswith(PDF_ALLOWED):
            errors.append(f'PDF outside paper/ or results/figures/: {rel}')
        if p.name.startswith('.env') or p.name.endswith('.csv.gz'):
            errors.append(f'raw/credential file: {rel}')
        if p.suffix=='.csv':
            with p.open(encoding='utf-8-sig',newline='') as stream:
                fields={c.lower() for c in next(csv.reader(stream),[])}
            if fields & (FORBIDDEN_FIELDS|{'y'}): errors.append(f'patient-level CSV field: {rel}')
        elif p.suffix=='.json':
            if bad_json(json.loads(p.read_text(encoding='utf-8'))): errors.append(f'patient-level JSON field: {rel}')
        elif p.suffix=='.ipynb':
            nb=json.loads(p.read_text(encoding='utf-8'));notebooks+=1
            if any(c.get('outputs') or c.get('execution_count') is not None for c in nb['cells'] if c.get('cell_type')=='code'):
                errors.append(f'notebook contains outputs or execution state: {rel}')
        if p.suffix.lower() in {'.py','.md','.yaml','.yml','.json','.txt','.sh','.ps1','.ipynb','.tex','.bib'}:
            text=p.read_text(encoding='utf-8')
            if any(pattern.search(text) for pattern in SECRET_PATTERNS): errors.append(f'credential-like content: {rel}')
    if errors:
        print('\n'.join(errors),file=sys.stderr);return 1
    print(f'Release checks passed: {files} tracked/staged files, {notebooks} clean notebooks; no forbidden paths, schemas or credential patterns.')
    return 0

if __name__=='__main__':raise SystemExit(main())
