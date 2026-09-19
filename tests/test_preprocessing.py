"""
SENTINEL Tests - Preprocessing module
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.config import LABEL_COLUMN, SELECTED_FEATURES
from src.preprocessing import (
    encode_labels, prepare_features, split_data, get_ablation_features,
)


class TestEncodeLabels:
    def test_encode_binary(self):
        y = pd.Series(["BENIGN", "ATTACK", "BENIGN", "ATTACK"])
        encoded, mapping = encode_labels(y)
        assert mapping == {"BENIGN": 0, "ATTACK": 1}
        assert list(encoded) == [0, 1, 0, 1]

    def test_encode_preserves_length(self):
        y = pd.Series(["BENIGN"] * 10 + ["ATTACK"] * 5)
        encoded, _ = encode_labels(y)
        assert len(encoded) == 15


class TestPrepareFeatures:
    def test_returns_xy(self):
        df = pd.DataFrame({
            "feat1": [1, 2, 3],
            "feat2": [4, 5, 6],
            LABEL_COLUMN: ["BENIGN", "ATTACK", "BENIGN"],
        })
        X, y = prepare_features(df, ["feat1", "feat2"])
        assert list(X.columns) == ["feat1", "feat2"]
        assert len(X) == 3
        assert len(y) == 3

    def test_does_not_modify_original(self):
        df = pd.DataFrame({
            "feat1": [1, 2, 3],
            LABEL_COLUMN: ["BENIGN", "ATTACK", "BENIGN"],
        })
        X, y = prepare_features(df, ["feat1"])
        assert "feat1" in X.columns
        assert len(X) == 3


class TestSplitData:
    def test_stratified_split(self):
        X = pd.DataFrame({"f": range(200)})
        y = pd.Series([0] * 150 + [1] * 50)
        Xtr, Xte, ytr, yte = split_data(X, y, test_size=0.2, random_state=42)
        assert len(Xtr) == 160
        assert len(Xte) == 40
        assert len(Xtr) + len(Xte) == 200

    def test_preserves_index(self):
        X = pd.DataFrame({"f": range(100)})
        y = pd.Series([0] * 70 + [1] * 30)
        Xtr, Xte, ytr, yte = split_data(X, y, test_size=0.3, random_state=42)
        assert Xtr.index.is_monotonic_increasing
        assert Xte.index.is_monotonic_increasing

    def test_deterministic(self):
        X = pd.DataFrame({"f": range(200)})
        y = pd.Series([0] * 150 + [1] * 50)
        r1 = split_data(X, y, random_state=42)
        r2 = split_data(X, y, random_state=42)
        assert list(r1[0].values) == list(r2[0].values)


class TestAblationFeatures:
    def test_baseline_all_features(self):
        feats = get_ablation_features([])
        assert len(feats) == len(SELECTED_FEATURES)

    def test_exclude_flow_rates(self):
        feats = get_ablation_features(["flow_rates"])
        assert "Flow Bytes/s" not in feats
        assert "Flow Packets/s" not in feats
        assert len(feats) == len(SELECTED_FEATURES) - 2

    def test_exclude_tcp_flags(self):
        feats = get_ablation_features(["tcp_flags"])
        assert "FIN Flag Count" not in feats
        assert "SYN Flag Count" not in feats
        assert len(feats) == len(SELECTED_FEATURES) - 4

    def test_exclude_multiple(self):
        feats = get_ablation_features(["flow_rates", "tcp_flags"])
        assert len(feats) == len(SELECTED_FEATURES) - 6
