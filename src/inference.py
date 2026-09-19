"""
SENTINEL Inference
Single-flow and batch CSV inference with schema validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import SELECTED_FEATURES, LABEL_COLUMN


def validate_flow_features(
    input_data: Dict[str, float],
    required_features: Optional[List[str]] = None,
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate that required features are present and numeric.
    Returns (is_valid, missing_features, invalid_features).
    """
    features = required_features or SELECTED_FEATURES
    missing = [f for f in features if f not in input_data]
    invalid = []
    for f in features:
        if f in input_data:
            val = input_data[f]
            if val is None:
                invalid.append(f)
            else:
                try:
                    float(val)
                except (TypeError, ValueError):
                    invalid.append(f)

    is_valid = len(missing) == 0 and len(invalid) == 0
    return is_valid, missing, invalid


def predict_single_flow(
    pipe: Pipeline,
    input_data: Dict[str, float],
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Predict a single network flow.
    Returns prediction, decision score, and metadata.
    """
    features = feature_cols or SELECTED_FEATURES

    is_valid, missing, invalid = validate_flow_features(input_data, features)
    if not is_valid:
        return {
            "error": True,
            "missing_features": missing,
            "invalid_features": invalid,
            "prediction": None,
            "decision_score": None,
        }

    input_array = np.array([[input_data[f] for f in features]])
    prediction = int(pipe.predict(input_array)[0])
    decision_score = float(pipe.decision_function(input_array)[0])

    label = "BENIGN" if prediction == 0 else "ATTACK"

    svm = pipe.named_steps.get("svm")
    model_type = "SVM"
    if svm is not None:
        model_type = f"SVM ({svm.kernel})"
    else:
        clf = pipe.named_steps.get("clf")
        if clf is not None:
            model_type = type(clf).__name__

    return {
        "error": False,
        "prediction": label,
        "prediction_int": prediction,
        "decision_score": round(decision_score, 6),
        "model": model_type,
        "features_used": features,
    }


def predict_batch_csv(
    pipe: Pipeline,
    csv_path: str,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    Run inference on a CSV file.
    Returns (result_df, error_info).
    """
    features = feature_cols or SELECTED_FEATURES

    try:
        df = pd.read_csv(csv_path, encoding="latin-1")
    except Exception as e:
        return None, {"error": True, "message": f"Failed to read CSV: {e}"}

    df.columns = [c.strip() for c in df.columns]

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        return None, {
            "error": True,
            "message": f"Missing required features: {missing_features}",
            "available_columns": list(df.columns),
            "required_features": features,
        }

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    nan_rows = df[features].isna().any(axis=1).sum()
    df_clean = df.dropna(subset=features).copy()

    if len(df_clean) == 0:
        return None, {"error": True, "message": "All rows contain invalid/missing values."}

    X = df_clean[features].values
    predictions = pipe.predict(X)
    decision_scores = pipe.decision_function(X)

    result_df = df.copy()
    result_df["prediction"] = ["BENIGN" if p == 0 else "ATTACK" for p in predictions]
    result_df["decision_score"] = np.nan
    result_df.loc[df_clean.index, "decision_score"] = decision_scores

    info = {
        "error": False,
        "total_rows": len(df),
        "valid_rows": len(df_clean),
        "nan_rows_dropped": int(nan_rows),
        "predictions": {
            "BENIGN": int((predictions == 0).sum()),
            "ATTACK": int((predictions == 1).sum()),
        },
    }

    return result_df, info
