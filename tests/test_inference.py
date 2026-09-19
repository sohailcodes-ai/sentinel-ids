"""
SENTINEL Tests - Inference module
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from src.model import build_pipeline
from src.inference import validate_flow_features, predict_single_flow, predict_batch_csv
from src.config import SELECTED_FEATURES


class TestFlowValidation:
    def test_valid_flow(self):
        data = {f: 0.0 for f in SELECTED_FEATURES}
        is_valid, missing, invalid = validate_flow_features(data)
        assert is_valid is True
        assert len(missing) == 0
        assert len(invalid) == 0

    def test_missing_features(self):
        data = {"Flow Duration": 100.0}
        is_valid, missing, invalid = validate_flow_features(data)
        assert is_valid is False
        assert len(missing) == len(SELECTED_FEATURES) - 1

    def test_invalid_feature(self):
        data = {f: 0.0 for f in SELECTED_FEATURES}
        data["Flow Duration"] = None
        is_valid, missing, invalid = validate_flow_features(data)
        assert is_valid is False
        assert "Flow Duration" in invalid


class TestSingleFlowPrediction:
    def test_predicts_benign_or_attack(self):
        pipe = build_pipeline(kernel="linear")
        X = np.random.randn(100, len(SELECTED_FEATURES))
        y = np.array([0] * 70 + [1] * 30)
        pipe.fit(X, y)
        data = {f: 0.0 for f in SELECTED_FEATURES}
        result = predict_single_flow(pipe, data)
        assert result["error"] is False
        assert result["prediction"] in ["BENIGN", "ATTACK"]
        assert isinstance(result["decision_score"], float)

    def test_error_on_missing(self):
        pipe = build_pipeline(kernel="linear")
        data = {"Flow Duration": 100.0}
        result = predict_single_flow(pipe, data)
        assert result["error"] is True
        assert len(result["missing_features"]) > 0


class TestBatchPrediction:
    def test_batch_csv(self):
        pipe = build_pipeline(kernel="linear")
        X = np.random.randn(100, len(SELECTED_FEATURES))
        y = np.array([0] * 70 + [1] * 30)
        pipe.fit(X, y)

        df = pd.DataFrame(X, columns=SELECTED_FEATURES)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            tmp_path = f.name

        try:
            result_df, info = predict_batch_csv(pipe, tmp_path)
            assert info is not None
            assert info["error"] is False
            assert info["total_rows"] == 100
            assert "prediction" in result_df.columns
            assert "decision_score" in result_df.columns
        finally:
            os.unlink(tmp_path)

    def test_batch_missing_features(self):
        pipe = build_pipeline(kernel="linear")
        df = pd.DataFrame({"random_col": [1, 2, 3]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            tmp_path = f.name
        try:
            result_df, info = predict_batch_csv(pipe, tmp_path)
            assert info["error"] is True
            assert "Missing" in info["message"]
        finally:
            os.unlink(tmp_path)
