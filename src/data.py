"""
SENTINEL Data Loading & Validation
Discovers, validates, cleans, and normalizes the network intrusion dataset.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    ATTACK_ALIASES,
    BENIGN_LABEL,
    DATASET_FILENAME,
    DATASET_PATH,
    FEATURE_ALIASES,
    LABEL_COLUMN,
    ORIGINAL_LABEL_COLUMN,
    SELECTED_FEATURES,
    MULTICLASS_MAP,
)


def _compute_file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def discover_dataset(custom_path: Optional[Path] = None) -> Path:
    """Return the dataset path if it exists, else raise."""
    if custom_path and custom_path.exists():
        return custom_path
    if DATASET_PATH.exists():
        return DATASET_PATH
    csv_files = list(DATA_DIR.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    raise FileNotFoundError(
        f"Dataset not found. Expected '{DATASET_FILENAME}' in {DATA_DIR}\n"
        f"See data/README.md for download instructions."
    )


def normalize_column_name(col: str) -> str:
    """Strip whitespace and normalize a column name."""
    return col.strip()


def resolve_feature_columns(columns: List[str]) -> Tuple[List[str], List[str]]:
    """Map selected features to actual column names. Returns (resolved, missing)."""
    col_map = {normalize_column_name(c): c for c in columns}
    resolved, missing = [], []
    for feat in SELECTED_FEATURES:
        candidates = FEATURE_ALIASES.get(feat, [feat])
        found = False
        for alias in candidates:
            key = normalize_column_name(alias)
            if key in col_map:
                resolved.append(col_map[key])
                found = True
                break
        if not found:
            missing.append(feat)
    return resolved, missing


def normalize_label(raw: str) -> str:
    """Map a raw label string to BENIGN or ATTACK."""
    if not isinstance(raw, str):
        return BENIGN_LABEL
    cleaned = raw.strip()
    if cleaned.upper() == BENIGN_LABEL or cleaned == "":
        return BENIGN_LABEL
    key = cleaned.lower()
    if key in ATTACK_ALIASES:
        return ATTACK_ALIASES[key]
    if "benign" in key:
        return BENIGN_LABEL
    return "ATTACK"


def normalize_multiclass_label(raw: str) -> str:
    """Map a raw label to a multiclass category."""
    if not isinstance(raw, str):
        return BENIGN_LABEL
    cleaned = raw.strip().lower()
    return MULTICLASS_MAP.get(cleaned, "Other")


def load_and_validate(
    dataset_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Load the dataset, clean it, and return (cleaned_df, validation_report)."""
    path = discover_dataset(dataset_path)

    file_hash = _compute_file_hash(path)
    raw_df = pd.read_csv(path, encoding="latin-1")
    raw_rows = len(raw_df)

    raw_df.columns = [normalize_column_name(c) for c in raw_df.columns]
    if LABEL_COLUMN not in raw_df.columns:
        label_candidates = [c for c in raw_df.columns if "label" in c.lower()]
        if label_candidates:
            raw_df.rename(columns={label_candidates[0]: LABEL_COLUMN}, inplace=True)
        else:
            raise ValueError(
                f"Label column '{LABEL_COLUMN}' not found. "
                f"Available: {list(raw_df.columns[:20])}"
            )

    raw_label_values = raw_df[LABEL_COLUMN].copy()

    resolved_features, missing_features = resolve_feature_columns(list(raw_df.columns))

    df = raw_df.copy()
    df[ORIGINAL_LABEL_COLUMN] = raw_label_values
    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(normalize_label)

    dup_count = int(df.duplicated(subset=resolved_features + [LABEL_COLUMN]).sum())
    df.drop_duplicates(subset=resolved_features + [LABEL_COLUMN], inplace=True)

    available_cols = resolved_features + [LABEL_COLUMN, ORIGINAL_LABEL_COLUMN]
    df = df[available_cols].copy()

    for col in resolved_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    inf_count = int(np.isinf(df[resolved_features].values).sum())
    df[resolved_features] = df[resolved_features].replace([np.inf, -np.inf], np.nan)

    nan_count = int(df[resolved_features].isna().any(axis=1).sum())
    df.dropna(subset=resolved_features, inplace=True)
    df.reset_index(drop=True, inplace=True)

    class_counts = df[LABEL_COLUMN].value_counts().to_dict()
    original_class_counts = df[ORIGINAL_LABEL_COLUMN].value_counts().to_dict()
    cleaned_rows = len(df)
    removed_rows = raw_rows - cleaned_rows

    total = cleaned_rows
    class_percentages = {}
    for cls, count in class_counts.items():
        class_percentages[cls] = round(count / total * 100, 2) if total > 0 else 0

    imbalance_ratio = 0
    if len(class_counts) == 2:
        vals = list(class_counts.values())
        if min(vals) > 0:
            imbalance_ratio = round(max(vals) / min(vals), 2)

    report = {
        "dataset_path": str(path),
        "dataset_hash": file_hash,
        "raw_rows": raw_rows,
        "cleaned_rows": cleaned_rows,
        "removed_rows": removed_rows,
        "duplicates_removed": dup_count,
        "inf_replaced": inf_count,
        "nan_rows_dropped": nan_count,
        "features_requested": SELECTED_FEATURES,
        "features_resolved": resolved_features,
        "features_missing": missing_features,
        "num_features": len(resolved_features),
        "class_distribution": class_counts,
        "class_percentages": class_percentages,
        "imbalance_ratio": imbalance_ratio,
        "original_class_distribution": original_class_counts,
        "columns": list(df.columns),
    }

    return df, report


def sample_stratified(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int = 42,
    label_col: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Stratified sampling respecting class ratios."""
    max_rows = len(df)
    if sample_size >= max_rows:
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    frac = sample_size / max_rows
    groups = []
    for _, group_df in df.groupby(label_col):
        n_sample = max(1, int(len(group_df) * frac))
        groups.append(group_df.sample(n=n_sample, random_state=random_state))
    sampled = pd.concat(groups, ignore_index=True)
    sampled.reset_index(drop=True, inplace=True)

    if len(sampled) < sample_size:
        remaining = sample_size - len(sampled)
        unsampled = df.loc[~df.index.isin(sampled.index)]
        if len(unsampled) > 0:
            extra = unsampled.sample(
                n=min(remaining, len(unsampled)), random_state=random_state
            )
            sampled = pd.concat([sampled, extra], ignore_index=True)

    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
