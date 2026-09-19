"""
SENTINEL Evaluation
Compute metrics, threshold analysis, feature importance, PCA, and save artifacts.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from src.config import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    COMPARISON_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_IMPORTANCE_PLOT_PATH,
    METRICS_PATH,
    OUTPUTS_DIR,
    PCA_VISUALIZATION_PATH,
    THRESHOLD_ANALYSIS_PATH,
    THRESHOLD_PLOT_PATH,
    TRAINING_SUMMARY_PATH,
    SELECTED_FEATURES,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute full evaluation metrics for binary classification."""
    labels = sorted(set(y_true.tolist() + y_pred.tolist()))
    if len(labels) < 2:
        labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0, pos_label=1)
    recall = recall_score(y_true, y_pred, zero_division=0, pos_label=1)
    f1 = f1_score(y_true, y_pred, zero_division=0, pos_label=1)

    target_names = ["BENIGN", "ATTACK"] if len(labels) == 2 else [str(l) for l in labels]
    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_matrix": cm.tolist(),
        "labels": [str(l) for l in labels],
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }


def compute_multiclass_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> Dict[str, Any]:
    """Compute metrics for multiclass classification."""
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
        "labels": class_names,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }


def threshold_analysis(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Analyze decision function across thresholds."""
    if thresholds is None:
        thresholds = np.linspace(-3, 3, 61)

    decision_scores = pipe.decision_function(X_test)

    results = []
    for t in thresholds:
        preds = (decision_scores >= t).astype(int)
        tp = int(((preds == 1) & (y_test == 1)).sum())
        tn = int(((preds == 0) & (y_test == 0)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(y_test) if len(y_test) > 0 else 0.0

        results.append({
            "threshold": round(float(t), 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        })

    return pd.DataFrame(results)


def save_threshold_analysis(
    threshold_df: pd.DataFrame, path: Optional[Path] = None
) -> Path:
    """Save threshold analysis CSV and plot."""
    csv_path = path or THRESHOLD_ANALYSIS_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(csv_path, index=False)

    plot_path = THRESHOLD_PLOT_PATH
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", linewidth=2)
    ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", linewidth=2)
    ax.plot(threshold_df["threshold"], threshold_df["f1"], label="F1", linewidth=2, linestyle="--")
    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision / Recall / F1 vs Decision Threshold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return csv_path


def compute_linear_svm_coefficients(pipe: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    """Extract and sort linear SVM feature coefficients."""
    svm = pipe.named_steps["svm"]
    coefs = svm.coef_[0]
    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    })
    df = df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return df


def compute_permutation_importance_analysis(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Compute permutation importance for any model."""
    result = permutation_importance(
        pipe, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring="f1",
        n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def save_feature_importance(
    df: pd.DataFrame, method_label: str = "feature_importance"
) -> Tuple[Path, Path]:
    """Save feature importance CSV and plot."""
    csv_path = FEATURE_IMPORTANCE_PATH
    plot_path = FEATURE_IMPORTANCE_PLOT_PATH

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    top_n = df.head(15)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_n)))[::-1]
    bars = ax.barh(range(len(top_n)), top_n.iloc[:, 1].values, color=colors)
    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels(top_n.iloc[:, 0].values, fontsize=9)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"{method_label.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return csv_path, plot_path


def compute_pca_visualization(
    X: pd.DataFrame,
    y: np.ndarray,
    n_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """Compute PCA for visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca[:, 0], X_pca[:, 1], pca


def save_pca_visualization(
    X_pca_0: np.ndarray,
    X_pca_1: np.ndarray,
    y: np.ndarray,
    pca: PCA,
    path: Optional[Path] = None,
) -> Path:
    """Save PCA scatter plot."""
    target = path or PCA_VISUALIZATION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {0: "#3b82f6", 1: "#ef4444"}
    labels_map = {0: "BENIGN", 1: "ATTACK"}

    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(
            X_pca_0[mask], X_pca_1[mask],
            c=colors[cls], label=labels_map[cls],
            alpha=0.5, s=15, edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.set_title("PCA Visualization of Network Flows", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def save_confusion_matrix(cm: np.ndarray, path: Optional[Path] = None) -> Path:
    """Save confusion matrix as a publication-quality plot."""
    target = path or CONFUSION_MATRIX_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    n_classes = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n_classes), max(5, n_classes - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if n_classes == 2:
        classes = ["BENIGN", "ATTACK"]
    else:
        classes = [str(i) for i in range(n_classes)]

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )
    ax.title.set_fontsize(13)
    ax.title.set_fontweight("bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right") if n_classes > 2 else None

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12 if n_classes <= 5 else 8,
                fontweight="bold",
            )

    plt.tight_layout()
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def save_classification_report_csv(
    y_true: np.ndarray, y_pred: np.ndarray, path: Optional[Path] = None
) -> Path:
    """Save classification report as CSV."""
    target = path or CLASSIFICATION_REPORT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    labels = sorted(set(y_true.tolist() + y_pred.tolist()))
    target_names = [str(l) for l in labels]
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )

    rows = []
    for cls_name in target_names:
        if cls_name in report:
            r = report[cls_name]
            rows.append({
                "class": cls_name,
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1-score": round(r["f1-score"], 4),
                "support": int(r["support"]),
            })

    with open(target, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class", "precision", "recall", "f1-score", "support"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return target


def save_model_comparison(results: List[Dict], path: Optional[Path] = None) -> Path:
    """Save model comparison table as CSV."""
    target = path or COMPARISON_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(target, index=False)
    return target


def save_metrics(metrics: Dict, path: Optional[Path] = None) -> Path:
    """Save metrics dict as JSON."""
    target = path or METRICS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(target, "w") as f:
        json.dump(metrics, f, indent=2, default=_clean)
    return target


def save_training_summary(summary: Dict, path: Optional[Path] = None) -> Path:
    """Save training summary as JSON."""
    target = path or TRAINING_SUMMARY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    with open(target, "w") as f:
        json.dump(summary, f, indent=2, default=_clean)
    return target


import pandas as pd
