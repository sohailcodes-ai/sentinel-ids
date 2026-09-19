"""
SENTINEL Tests - Model module
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from sklearn.pipeline import Pipeline
from src.model import (
    build_pipeline, build_logistic_regression_pipeline,
    build_random_forest_pipeline, train_linear_svm, train_rbf_svm,
    train_benchmark, serialize_pipeline, load_pipeline, get_model_info,
)


class TestPipelineCreation:
    def test_linear_pipeline(self):
        pipe = build_pipeline(kernel="linear", C=1.0)
        assert "scaler" in pipe.named_steps
        assert "svm" in pipe.named_steps
        assert pipe.named_steps["svm"].kernel == "linear"

    def test_rbf_pipeline(self):
        pipe = build_pipeline(kernel="rbf", C=1.0, gamma="scale")
        assert pipe.named_steps["svm"].kernel == "rbf"

    def test_lr_pipeline(self):
        pipe = build_logistic_regression_pipeline()
        assert "scaler" in pipe.named_steps
        assert "clf" in pipe.named_steps

    def test_rf_pipeline(self):
        pipe = build_random_forest_pipeline(n_estimators=10)
        assert "scaler" in pipe.named_steps
        assert "clf" in pipe.named_steps
        assert pipe.named_steps["clf"].n_estimators == 10


class TestLinearSVMTraining:
    def test_trains_and_predicts(self):
        X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
        y = pd.Series([0] * 70 + [1] * 30)
        pipe, elapsed = train_linear_svm(X, y, C=1.0)
        assert isinstance(pipe, Pipeline)
        assert elapsed > 0
        pred = pipe.predict(X)
        assert len(pred) == 100
        assert set(pred).issubset({0, 1})


class TestRBFSVMTraining:
    def test_trains_and_predicts(self):
        X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
        y = pd.Series([0] * 70 + [1] * 30)
        pipe, elapsed = train_rbf_svm(X, y, C=1.0, gamma="scale")
        assert isinstance(pipe, Pipeline)
        pred = pipe.predict(X)
        assert len(pred) == 100


class TestBenchmarkTraining:
    def test_logistic_regression(self):
        X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
        y = pd.Series([0] * 70 + [1] * 30)
        pipe, elapsed = train_benchmark("logistic_regression", X, y)
        assert isinstance(pipe, Pipeline)
        pred = pipe.predict(X)
        assert len(pred) == 100

    def test_random_forest(self):
        X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
        y = pd.Series([0] * 70 + [1] * 30)
        pipe, elapsed = train_benchmark("random_forest", X, y)
        assert isinstance(pipe, Pipeline)

    def test_unknown_raises(self):
        X = pd.DataFrame({"f1": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        with pytest.raises(ValueError):
            train_benchmark("unknown_model", X, y)


class TestSerialization:
    def test_roundtrip(self):
        pipe = build_pipeline(kernel="linear")
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = Path(f.name)
        try:
            serialize_pipeline(pipe, path)
            loaded = load_pipeline(path)
            assert isinstance(loaded, Pipeline)
            assert loaded.named_steps["svm"].kernel == "linear"
        finally:
            os.unlink(path)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_pipeline(Path("/nonexistent/path.joblib"))


class TestModelInfo:
    def test_linear_svm_info(self):
        pipe = build_pipeline(kernel="linear")
        X = np.random.randn(50, 3)
        y = np.array([0] * 35 + [1] * 15)
        pipe.fit(X, y)
        info = get_model_info(pipe)
        assert info["kernel"] == "linear"
        assert "n_support_vectors" in info
        assert info["n_features"] == 3

    def test_rbf_svm_info(self):
        pipe = build_pipeline(kernel="rbf", C=1.0, gamma="scale")
        X = np.random.randn(50, 3)
        y = np.array([0] * 35 + [1] * 15)
        pipe.fit(X, y)
        info = get_model_info(pipe)
        assert info["kernel"] == "rbf"
        assert "gamma" in info

    def test_non_svm_info(self):
        pipe = build_logistic_regression_pipeline()
        X = np.random.randn(50, 3)
        y = np.array([0] * 35 + [1] * 15)
        pipe.fit(X, y)
        info = get_model_info(pipe)
        assert info["type"] == "LogisticRegression"
