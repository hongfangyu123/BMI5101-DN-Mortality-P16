"""从 V3 本地数据准备扩展特征；不拟合机器学习模型。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dn_extensions.features import build_features
from dn_extensions.reporting import update_preparation_notebook

if __name__ == '__main__':
    update_preparation_notebook(build_features())
