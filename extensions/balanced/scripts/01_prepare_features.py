"""Prepare extension features from local V3 data; no machine-learning model is fitted."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dn_extensions.features import build_features
from dn_extensions.reporting import update_preparation_notebook

if __name__ == '__main__':
    update_preparation_notebook(build_features())
