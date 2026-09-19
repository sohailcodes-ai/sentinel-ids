#!/usr/bin/env python3
"""
SENTINEL Training CLI
End-to-end SVM training pipeline with structured output.

Usage:
    python train.py
    python train.py --sample-size 20000 --random-state 42
    python train.py --benchmark --sample-size 20000
    python train.py --multiclass --sample-size 20000
    python train.py --ablation --sample-size 20000
    python train.py --full --sample-size 20000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    ABLATION_CONFIGS, DEFAULT_RANDOM_STATE, DEFAULT_SAMPLE_SIZE,
    MODELS_DIR, OUTPUTS_DIR, SELECTED_FEATURES, ORIGINAL_LABEL_COLUMN,
    LABEL_COLUMN,
)
from src.data import load_and_validate, sample_stratified, normalize_multiclass_label
from src.evaluation import (
    compute_metrics, compute_multiclass_metrics, save_classification_report_csv,
    save_confusion_matrix, save_metrics, save_model_comparison,
    save_training_summary, threshold_analysis, save_threshold_analysis,
    compute_linear_svm_coefficients, compute_permutation_importance_analysis,
    save_feature_importance, compute_pca_visualization, save_pca_visualization,
)
from src.model import (
    get_model_info, save_metadata, search_rbf_svm, serialize_pipeline,
    train_linear_svm, train_benchmark,
)
from src.preprocessing import (
    encode_labels, prepare_features, split_data, get_ablation_features,
)
from src.experiments import (
    save_experiment_config, save_experiment_features, save_experiment_metrics,
    save_experiment_readme,
)
from src.utils import environment_info, generate_experiment_id, log


def parse_args():
    p = argparse.ArgumentParser(description="SENTINEL: Train SVM network intrusion classifiers")
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--multiclass", action="store_true")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--full", action="store_true")
    return p.parse_args()


def run_binary_svm_experiment(X_train, X_test, y_train, y_test, feature_cols):
    log("Experiment: Linear SVM")
    linear_pipe, linear_time = train_linear_svm(X_train, y_train, C=1.0)
    linear_pred = linear_pipe.predict(X_test)
    linear_metrics = compute_metrics(y_test.values, linear_pred)
    linear_info = get_model_info(linear_pipe)
    linear_results = {
        "model": "Linear SVM", "kernel": "linear", "C": 1.0, "gamma": "N/A",
        "train_time": round(linear_time, 4),
        "accuracy": linear_metrics["accuracy"], "precision": linear_metrics["precision"],
        "recall": linear_metrics["recall"], "f1": linear_metrics["f1"],
    }
    log(f"  C=1.0 | Acc={linear_metrics['accuracy']} | F1={linear_metrics['f1']} | Time={linear_time:.3f}s")

    log("Experiment: RBF SVM")
    rbf_best, rbf_search, rbf_time = search_rbf_svm(X_train, y_train)
    rbf_pred = rbf_best.predict(X_test)
    rbf_metrics = compute_metrics(y_test.values, rbf_pred)
    rbf_info = get_model_info(rbf_best)
    rbf_results = {
        "model": "RBF SVM (best)", "kernel": "rbf", "C": rbf_info["C"],
        "gamma": rbf_info.get("gamma", "scale"), "train_time": round(rbf_time, 4),
        "accuracy": rbf_metrics["accuracy"], "precision": rbf_metrics["precision"],
        "recall": rbf_metrics["recall"], "f1": rbf_metrics["f1"],
    }
    log(f"  Best C={rbf_info['C']} gamma={rbf_info.get('gamma','scale')} | F1={rbf_metrics['f1']} | Time={rbf_time:.3f}s")

    if linear_metrics["f1"] >= rbf_metrics["f1"]:
        best_pipe, best_metrics, best_info, best_name = linear_pipe, linear_metrics, linear_info, "Linear SVM"
    else:
        best_pipe, best_metrics, best_info, best_name = rbf_best, rbf_metrics, rbf_info, "RBF SVM (best)"

    log(f"  Selected: {best_name} | F1={best_metrics['f1']}")

    cm = np.array(best_metrics["confusion_matrix"])
    save_confusion_matrix(cm)
    save_classification_report_csv(y_test.values, linear_pred)
    save_model_comparison([linear_results, rbf_results])
    save_metrics(best_metrics)

    log("Threshold analysis")
    try:
        tdf = threshold_analysis(best_pipe, X_test, y_test.values)
        save_threshold_analysis(tdf)
    except Exception as e:
        log(f"  Threshold analysis failed: {e}", tag="WARN")

    log("Feature importance")
    try:
        if best_info.get("kernel") == "linear":
            coef_df = compute_linear_svm_coefficients(best_pipe, feature_cols)
            coef_df.rename(columns={"coefficient": "importance_mean"}, inplace=True)
        else:
            coef_df = compute_permutation_importance_analysis(
                best_pipe, X_test, y_test.values, feature_cols
            )
        save_feature_importance(coef_df)
    except Exception as e:
        log(f"  Feature importance failed: {e}", tag="WARN")

    log("PCA visualization")
    try:
        scaler = best_pipe.named_steps["scaler"]
        X_scaled = scaler.transform(X_test)
        pc0, pc1, pca_model = compute_pca_visualization(X_scaled, y_test.values)
        save_pca_visualization(pc0, pc1, y_test.values, pca_model)
    except Exception as e:
        log(f"  PCA failed: {e}", tag="WARN")

    return {
        "best_pipe": best_pipe, "best_metrics": best_metrics,
        "best_info": best_info, "best_name": best_name,
        "linear_results": linear_results, "rbf_results": rbf_results,
        "rbf_search": rbf_search,
    }


def run_benchmark_experiment(X_train, X_test, y_train, y_test):
    results = []
    for model_type, display in [("logistic_regression", "Logistic Regression"),
                                 ("random_forest", "Random Forest")]:
        log(f"  Benchmark: {display}")
        pipe, t = train_benchmark(model_type, X_train, y_train)
        pred = pipe.predict(X_test)
        m = compute_metrics(y_test.values, pred)
        results.append({
            "model": display, "kernel": "N/A", "C": "N/A", "gamma": "N/A",
            "train_time": round(t, 4), "accuracy": m["accuracy"],
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
        })
        log(f"    Acc={m['accuracy']} | F1={m['f1']} | Time={t:.3f}s")
    return results


def run_ablation_experiments(df, args):
    results = []
    for name, cfg in ABLATION_CONFIGS.items():
        log(f"  Ablation: {name} - {cfg['description']}")
        feats = get_ablation_features(cfg["exclude_groups"])
        if len(feats) < 2:
            log(f"    Skipped: too few features ({len(feats)})")
            continue
        ds = sample_stratified(df, args.sample_size, args.random_state)
        X, y = prepare_features(ds, feats)
        y_enc, _ = encode_labels(y)
        Xtr, Xte, ytr, yte = split_data(X, y_enc, random_state=args.random_state)
        pipe, t = train_linear_svm(Xtr, ytr, C=1.0)
        pred = pipe.predict(Xte)
        m = compute_metrics(yte.values, pred)
        results.append({
            "ablation": name, "description": cfg["description"],
            "n_features": len(feats), "features": feats,
            "accuracy": m["accuracy"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"],
            "train_time": round(t, 4),
        })
        log(f"    F1={m['f1']} | Features={len(feats)}")
    return results


def main():
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    experiment_id = generate_experiment_id()
    log(f"Experiment ID: {experiment_id}")

    log("Dataset validation")
    try:
        dataset_path = Path(args.dataset) if args.dataset else None
        df, vr = load_and_validate(dataset_path)
    except FileNotFoundError as e:
        log(str(e), tag="ERROR")
        sys.exit(1)
    except Exception as e:
        log(f"Dataset validation failed: {e}", tag="ERROR")
        sys.exit(1)

    log(f"  Raw rows: {vr['raw_rows']}")
    log(f"  Cleaned rows: {vr['cleaned_rows']}")
    log(f"  Features resolved: {vr['num_features']}")
    log(f"  Missing: {vr['features_missing']}")
    log(f"  Class dist: {vr['class_distribution']}")
    log(f"  Imbalance: {vr['imbalance_ratio']}")

    feature_cols = vr["features_resolved"]

    log("Sampling")
    df_sampled = sample_stratified(df, args.sample_size, args.random_state)
    log(f"  Sampled: {len(df_sampled)}")

    log("Train/test split")
    X, y = prepare_features(df_sampled, feature_cols)
    y_encoded, label_mapping = encode_labels(y)
    X_train, X_test, y_train, y_test = split_data(X, y_encoded, random_state=args.random_state)
    log(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    env = environment_info()
    svm_result = run_binary_svm_experiment(X_train, X_test, y_train, y_test, feature_cols)
    all_comp = [svm_result["linear_results"], svm_result["rbf_results"]]

    if args.benchmark or args.full:
        log("Benchmark models")
        bench = run_benchmark_experiment(X_train, X_test, y_train, y_test)
        all_comp.extend(bench)
        save_model_comparison(all_comp)

    if args.multiclass or args.full:
        log("Multiclass experiment")
        try:
            df_mc = sample_stratified(df, args.sample_size, args.random_state)
            df_mc["mc_label"] = df_mc[ORIGINAL_LABEL_COLUMN].apply(normalize_multiclass_label)
            mc_classes = sorted(df_mc["mc_label"].unique())
            if len(mc_classes) > 2:
                df_mc[LABEL_COLUMN] = df_mc["mc_label"]
                X_mc, y_mc_raw = prepare_features(df_mc, feature_cols)
                mapping = {c: i for i, c in enumerate(mc_classes)}
                y_mc = y_mc_raw.map(mapping)
                Xtr, Xte, ytr, yte = split_data(X_mc, y_mc, random_state=args.random_state)
                mc_pipe, mc_t = train_linear_svm(Xtr, ytr, C=1.0)
                mc_pred = mc_pipe.predict(Xte)
                mc_m = compute_multiclass_metrics(yte.values, mc_pred, mc_classes)
                log(f"  Macro F1: {mc_m['macro_f1']} | Weighted F1: {mc_m['weighted_f1']}")
                from src.config import MULTICLASS_PIPELINE_PATH
                serialize_pipeline(mc_pipe, MULTICLASS_PIPELINE_PATH)
        except Exception as e:
            log(f"  Multiclass failed: {e}", tag="WARN")

    if args.ablation or args.full:
        log("Ablation experiments")
        abl = run_ablation_experiments(df, args)
        with open(OUTPUTS_DIR / "ablation_results.json", "w") as f:
            json.dump(abl, f, indent=2)
        log(f"  Saved to {OUTPUTS_DIR / 'ablation_results.json'}")

    log("Serialization")
    pipe_path = serialize_pipeline(svm_result["best_pipe"])
    meta = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": vr["dataset_path"],
        "dataset_hash": vr["dataset_hash"],
        "random_state": args.random_state,
        "sample_size": args.sample_size,
        "features": feature_cols,
        "label_mapping": label_mapping,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "best_model": svm_result["best_name"],
        "linear_results": svm_result["linear_results"],
        "rbf_results": svm_result["rbf_results"],
        "rbf_search": svm_result["rbf_search"],
        "metrics": svm_result["best_metrics"],
        "model_info": svm_result["best_info"],
        "environment": env,
        "validation_report": vr,
        "benchmark_requested": args.benchmark or args.full,
        "multiclass_requested": args.multiclass or args.full,
        "ablation_requested": args.ablation or args.full,
    }
    meta_path = save_metadata(meta)

    save_experiment_config(meta, experiment_id)
    save_experiment_features(feature_cols, experiment_id)
    save_experiment_metrics(svm_result["best_metrics"], experiment_id)
    save_experiment_readme(
        experiment_id,
        f"Binary SVM classification: {svm_result['best_name']}",
        svm_result["best_info"],
        svm_result["best_metrics"],
    )

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_id": experiment_id,
        "dataset": vr["dataset_path"],
        "dataset_hash": vr["dataset_hash"],
        "random_state": args.random_state,
        "sample_size": args.sample_size,
        "features": feature_cols,
        "label_mapping": label_mapping,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "best_model": svm_result["best_name"],
        "linear_results": svm_result["linear_results"],
        "rbf_results": svm_result["rbf_results"],
        "rbf_search": svm_result["rbf_search"],
        "metrics": svm_result["best_metrics"],
        "environment": env,
        "validation_report": vr,
    }
    save_training_summary(summary)

    log(f"  Pipeline: {pipe_path}")
    log(f"  Metadata: {meta_path}")
    log(f"  Experiment: {EXPERIMENTS_DIR / experiment_id}")
    log("Done")


if __name__ == "__main__":
    main()
