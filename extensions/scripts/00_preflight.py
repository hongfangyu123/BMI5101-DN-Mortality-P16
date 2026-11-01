"""Only check dependencies, the V3 development set and the concept table structure; no model is trained."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import duckdb
from dn_extensions.features import CONCEPTS, load_development
from dn_extensions.settings import DATABASE, V3_ROOT


def main():
    train, _, _ = load_development()
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        for concept, variables in CONCEPTS.items():
            names = set(con.execute(f'DESCRIBE mimiciv_derived.{concept}').df().column_name)
            assert set(variables) <= names, f'{concept} is missing columns'
    print(f'check passed: V3 development set {len(train)} patients, {int(train.y.sum())} deaths.')
    print(f'source project: {V3_ROOT}; held-out labels not read; no training started.')


if __name__ == '__main__':
    main()
