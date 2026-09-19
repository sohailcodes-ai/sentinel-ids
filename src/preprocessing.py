"""
SENTINEL Preprocessing
Feature selection, scaling, and train/test splitting.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DEFAULT_RANDOM_STATE,
    LABEL_COLUMN,
    ORIGINAL_LABEL_COLUMN,
    TEST_SIZE,
)


def encode_labels(y: pd.Series) -> Tuple[pd.Series, dict]:
    """Encode BENIGN=0, ATTACK=1. Returns (encoded_series, mapping)."""
    mapping = {"BENIGN": 0, "ATTACK": 1}
    return y.map(mapping), mapping


def prepare_features(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract X and y from the dataframe."""
    X = df[feature_cols].copy()
    y = df[LABEL_COLUMN].copy()
    return X, y


def prepare_multiclass_features(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Extract X and multiclass y with encoding mapping."""
    X = df[feature_cols].copy()
    y_raw = df[ORIGINAL_LABEL_COLUMN].copy()
    classes = sorted(y_raw.unique())
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    y_encoded = y_raw.map(mapping)
    return X, y_encoded, mapping


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def get_ablation_features(exclude_groups: List[str]) -> List[str]:
    """Return feature list after excluding specified groups."""
    from src.config import FEATURE_GROUPS, SELECTED_FEATURES

    exclude_features = set()
    for group_name in exclude_groups:
        if group_name in FEATURE_GROUPS:
            exclude_features.update(FEATURE_GROUPS[group_name])
    return [f for f in SELECTED_FEATURES if f not in exclude_features]
