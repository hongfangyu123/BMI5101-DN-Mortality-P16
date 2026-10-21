"""只检查依赖、V3 开发集与 concept 表结构，不训练模型。"""
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
            assert set(variables) <= names, f'{concept} 缺少列'
    print(f'检查通过：V3 开发集 {len(train)} 人，死亡 {int(train.y.sum())} 人。')
    print(f'源项目：{V3_ROOT}；留出集标签未读取；没有启动训练。')


if __name__ == '__main__':
    main()
