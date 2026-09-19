"""
SENTINEL Tests - Evaluation module
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from src.evaluation import (
    compute_metrics, compute_multiclass_metrics, threshold_analysis,
    compute_linear_svm_coefficients, save_confusion_matrix,
    save_classification_report_csv, save_metrics, save_model_comparison,
)
from src.model import build_pipeline


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["true_positives"] == 2
        assert m["true_negatives"] == 2
        assert m["false_positives"] == 0
        assert m["false_negatives"] == 0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0
        assert m["false_positives"] == 2
        assert m["false_negatives"] == 2

    def test_partial_predictions(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        m = compute_metrics(y_true, y_pred)
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0
        # 3 actual positives, 3 actual negatives
        assert m["true_positives"] + m["false_negatives"] == 3
        assert m["true_negatives"] + m["false_positives"] == 3

    def test_returns_confusion_matrix(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        m = compute_metrics(y_true, y_pred)
        cm = m["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2


class TestMulticlassMetrics:
    def test_three_classes(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])
        m = compute_multiclass_metrics(y_true, y_pred, ["A", "B", "C"])
        assert "macro_f1" in m
        assert "weighted_f1" in m
        assert 0.0 <= m["accuracy"] <= 1.0
        assert len(m["confusion_matrix"]) == 3


class TestThresholdAnalysis:
    def test_runs_without_error(self):
        pipe = build_pipeline(kernel="linear")
        X = np.random.randn(100, 3)
        y = np.array([0] * 70 + [1] * 30)
        pipe.fit(X, y)
        tdf = threshold_analysis(pipe, pd.DataFrame(X), y)
        assert len(tdf) > 0
        assert "threshold" in tdf.columns
        assert "precision" in tdf.columns
        assert "recall" in tdf.columns
        assert "f1" in tdf.columns


class TestLinearCoefficients:
    def test_extracts_coefficients(self):
        pipe = build_pipeline(kernel="linear")
        X = np.random.randn(50, 3)
        y = np.array([0] * 35 + [1] * 15)
        pipe.fit(X, y)
        df = compute_linear_svm_coefficients(pipe, ["f1", "f2", "f3"])
        assert len(df) == 3
        assert "coefficient" in df.columns
        assert "abs_coefficient" in df.columns
        assert df.iloc[0]["abs_coefficient"] >= df.iloc[-1]["abs_coefficient"]


class TestSaveArtifacts:
    def test_save_confusion_matrix(self):
        cm = np.array([[100, 5], [3, 92]])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = Path(f.name)
        try:
            save_confusion_matrix(cm, path)
            assert path.exists()
            assert path.stat().st_size > 0
        finally:
            os.unlink(path)

    def test_save_classification_report(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            save_classification_report_csv(y_true, y_pred, path)
            assert path.exists()
            df = pd.read_csv(path)
            assert len(df) > 0
        finally:
            os.unlink(path)

    def test_save_metrics_json(self):
        m = {"accuracy": 0.95, "f1": 0.93}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            save_metrics(m, path)
            assert path.exists()
        finally:
            os.unlink(path)

    def test_save_model_comparison(self):
        results = [{"model": "A", "f1": 0.9}, {"model": "B", "f1": 0.8}]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            save_model_comparison(results, path)
            assert path.exists()
            df = pd.read_csv(path)
            assert len(df) == 2
        finally:
            os.unlink(path)
