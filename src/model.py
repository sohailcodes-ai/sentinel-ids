"""
SENTINEL Model
SVM training, hyperparameter search, benchmark models, and pipeline serialization.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.config import (
    DEFAULT_RANDOM_STATE,
    METADATA_PATH,
    MODELS_DIR,
    PIPELINE_PATH,
    RBF_GRID,
)


def build_pipeline(
    kernel: str = "linear",
    C: float = 1.0,
    gamma: Any = "scale",
    class_weight: str = "balanced",
) -> Pipeline:
    """Build an SVM pipeline with StandardScaler."""
    params: Dict[str, Any] = {
        "kernel": kernel,
        "C": C,
        "class_weight": class_weight,
        "random_state": DEFAULT_RANDOM_STATE,
    }
    if kernel == "rbf":
        params["gamma"] = gamma
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(**params)),
    ])


def build_logistic_regression_pipeline() -> Pipeline:
    """Build a Logistic Regression pipeline with StandardScaler."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=DEFAULT_RANDOM_STATE,
        )),
    ])


def build_random_forest_pipeline(n_estimators: int = 100) -> Pipeline:
    """Build a Random Forest pipeline with StandardScaler."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def train_linear_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 1.0,
) -> Tuple[Pipeline, float]:
    """Train a linear SVM. Returns (pipeline, train_time)."""
    pipe = build_pipeline(kernel="linear", C=C)
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    return pipe, elapsed


def train_rbf_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 1.0,
    gamma: Any = "scale",
) -> Tuple[Pipeline, float]:
    """Train an RBF SVM. Returns (pipeline, train_time)."""
    pipe = build_pipeline(kernel="rbf", C=C, gamma=gamma)
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    return pipe, elapsed


def search_rbf_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict] = None,
    cv: int = 3,
) -> Tuple[Pipeline, List[Dict], float]:
    """Grid search over RBF parameters. Returns (best_pipeline, all_results, total_time)."""
    if param_grid is None:
        param_grid = {
            "svm__C": RBF_GRID["C"],
            "svm__gamma": RBF_GRID["gamma"],
        }

    base_pipe = build_pipeline(kernel="rbf", C=1.0, gamma="scale")

    t0 = time.perf_counter()
    search = GridSearchCV(
        base_pipe,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    total_time = time.perf_counter() - t0

    all_results = []
    for params, mean_score, std_score in zip(
        search.cv_results_["params"],
        search.cv_results_["mean_test_score"],
        search.cv_results_["std_test_score"],
    ):
        all_results.append({
            "kernel": "rbf",
            "C": params.get("svm__C"),
            "gamma": str(params.get("svm__gamma")),
            "cv_f1_mean": round(float(mean_score), 4),
            "cv_f1_std": round(float(std_score), 4),
        })

    return search.best_estimator_, all_results, total_time


def train_benchmark(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Pipeline, float]:
    """Train a benchmark model. Returns (pipeline, train_time)."""
    if model_type == "logistic_regression":
        pipe = build_logistic_regression_pipeline()
    elif model_type == "random_forest":
        pipe = build_random_forest_pipeline()
    else:
        raise ValueError(f"Unknown benchmark model: {model_type}")

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    return pipe, elapsed


def serialize_pipeline(pipe: Pipeline, path: Optional[Path] = None) -> Path:
    """Save the trained pipeline to disk."""
    target = path or PIPELINE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, target)
    return target


def load_pipeline(path: Optional[Path] = None) -> Pipeline:
    """Load a trained pipeline from disk."""
    target = path or PIPELINE_PATH
    if not target.exists():
        raise FileNotFoundError(f"No pipeline found at {target}. Run train.py first.")
    return joblib.load(target)


def save_metadata(meta: Dict, path: Optional[Path] = None) -> Path:
    """Save model metadata as JSON."""
    target = path or METADATA_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    with open(target, "w") as f:
        json.dump(meta, f, indent=2, default=_serialize)
    return target


def load_metadata(path: Optional[Path] = None) -> Dict:
    """Load model metadata from JSON."""
    target = path or METADATA_PATH
    if not target.exists():
        return {}
    with open(target) as f:
        return json.load(f)


def get_model_info(pipe: Pipeline) -> Dict:
    """Extract key info from a trained SVM pipeline."""
    svm = pipe.named_steps.get("svm")
    if svm is None:
        clf = pipe.named_steps.get("clf")
        if clf is not None:
            return {
                "type": type(clf).__name__,
                "kernel": "N/A (not SVM)",
                "n_features": int(clf.n_features_in_) if hasattr(clf, "n_features_in_") else 0,
            }
        return {}

    info: Dict[str, Any] = {
        "type": "SVM",
        "kernel": svm.kernel,
        "C": float(svm.C),
        "n_support_vectors": int(svm.n_support_.sum()),
        "support_vectors_per_class": {
            str(cls): int(n) for cls, n in zip(svm.classes_, svm.n_support_)
        },
        "n_features": int(svm.shape_fit_[1]) if hasattr(svm, "shape_fit_") else 0,
    }
    if svm.kernel == "rbf":
        gamma_val = svm.gamma
        info["gamma"] = str(gamma_val) if isinstance(gamma_val, str) else float(gamma_val)
    return info
