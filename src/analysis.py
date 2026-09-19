"""
SENTINEL Analysis
Feature analysis, PCA visualization, and drift simulation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import LABEL_COLUMN, SELECTED_FEATURES


def simulate_distribution_shift(
    df: pd.DataFrame,
    feature_cols: List[str],
    shift_magnitude: float = 1.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simulate data drift by scaling feature distributions.
    Creates a shifted version where inference-time distributions
    differ from training distributions.
    """
    rng = np.random.RandomState(random_state)
    shifted_df = df.copy()

    n_features_to_shift = max(1, len(feature_cols) // 3)
    features_to_shift = rng.choice(feature_cols, size=n_features_to_shift, replace=False)

    for feat in features_to_shift:
        mean_val = shifted_df[feat].mean()
        std_val = shifted_df[feat].std()
        if std_val > 0:
            noise = rng.normal(0, std_val * (shift_magnitude - 1.0), size=len(shifted_df))
            shifted_df[feat] = shifted_df[feat] + noise

    return shifted_df


def compute_feature_statistics(
    df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """Compute per-feature statistics."""
    stats = []
    for feat in feature_cols:
        col = df[feat]
        stats.append({
            "feature": feat,
            "mean": round(float(col.mean()), 4),
            "std": round(float(col.std()), 4),
            "min": round(float(col.min()), 4),
            "q25": round(float(col.quantile(0.25)), 4),
            "median": round(float(col.median()), 4),
            "q75": round(float(col.quantile(0.75)), 4),
            "max": round(float(col.max()), 4),
            "skewness": round(float(col.skew()), 4),
            "kurtosis": round(float(col.kurtosis()), 4),
            "missing": int(col.isna().sum()),
        })
    return pd.DataFrame(stats)


def compute_distribution_comparison(
    train_df: pd.DataFrame,
    shifted_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Compare distributions between training and shifted data."""
    results = []
    for feat in feature_cols:
        train_mean = train_df[feat].mean()
        shift_mean = shifted_df[feat].mean()
        train_std = train_df[feat].std()
        shift_std = shifted_df[feat].std()

        mean_change = abs(shift_mean - train_mean) / (train_std + 1e-10) * 100
        std_change = abs(shift_std - train_std) / (train_std + 1e-10) * 100

        results.append({
            "feature": feat,
            "train_mean": round(float(train_mean), 4),
            "shifted_mean": round(float(shift_mean), 4),
            "mean_change_pct": round(float(mean_change), 2),
            "train_std": round(float(train_std), 4),
            "shifted_std": round(float(shift_std), 4),
            "std_change_pct": round(float(std_change), 2),
        })

    return pd.DataFrame(results)


def evaluate_drift_impact(
    pipe: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_shifted: np.ndarray,
    y_shifted: np.ndarray,
) -> Dict:
    """Evaluate model performance under distribution shift."""
    from src.evaluation import compute_metrics

    baseline_pred = pipe.predict(X_test)
    baseline_metrics = compute_metrics(y_test, baseline_pred)

    shifted_pred = pipe.predict(X_shifted)
    shifted_metrics = compute_metrics(y_shifted, shifted_pred)

    metric_comparison = {}
    for key in ["accuracy", "precision", "recall", "f1"]:
        baseline_val = baseline_metrics.get(key, 0)
        shifted_val = shifted_metrics.get(key, 0)
        delta = shifted_val - baseline_val
        metric_comparison[key] = {
            "baseline": baseline_val,
            "shifted": shifted_val,
            "delta": round(delta, 4),
            "delta_pct": round(delta / (baseline_val + 1e-10) * 100, 2),
        }

    return {
        "baseline_metrics": baseline_metrics,
        "shifted_metrics": shifted_metrics,
        "metric_comparison": metric_comparison,
    }
