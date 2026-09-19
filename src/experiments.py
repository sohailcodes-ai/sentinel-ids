"""
SENTINEL Experiment Tracking
Local experiment management with structured artifacts.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import EXPERIMENTS_DIR
from src.utils import generate_experiment_id, utc_now_iso


def create_experiment_dir(experiment_id: str) -> Path:
    """Create and return the experiment directory."""
    exp_dir = EXPERIMENTS_DIR / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_experiment_config(config: Dict, experiment_id: str) -> Path:
    """Save experiment configuration."""
    exp_dir = create_experiment_dir(experiment_id)
    path = exp_dir / "config.json"

    def _clean(obj):
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=_clean)
    return path


def save_experiment_metrics(metrics: Dict, experiment_id: str) -> Path:
    """Save experiment metrics."""
    exp_dir = create_experiment_dir(experiment_id)
    path = exp_dir / "metrics.json"

    def _clean(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, list):
            return [_clean(i) for i in obj]
        return obj

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=_clean)
    return path


def save_experiment_features(features: List[str], experiment_id: str) -> Path:
    """Save experiment feature list."""
    exp_dir = create_experiment_dir(experiment_id)
    path = exp_dir / "feature_list.json"
    with open(path, "w") as f:
        json.dump({"features": features, "count": len(features)}, f, indent=2)
    return path


def save_experiment_readme(
    experiment_id: str,
    description: str,
    model_info: Dict,
    metrics: Dict,
) -> Path:
    """Save a README for the experiment."""
    exp_dir = create_experiment_dir(experiment_id)
    path = exp_dir / "README.md"

    content = f"""# {experiment_id}

## Description
{description}

## Model
- Type: {model_info.get('type', 'N/A')}
- Kernel: {model_info.get('kernel', 'N/A')}
- C: {model_info.get('C', 'N/A')}
- Gamma: {model_info.get('gamma', 'N/A')}

## Metrics
- Accuracy: {metrics.get('accuracy', 'N/A')}
- Precision: {metrics.get('precision', 'N/A')}
- Recall: {metrics.get('recall', 'N/A')}
- F1: {metrics.get('f1', 'N/A')}

## Timestamp
{utc_now_iso()}
"""
    with open(path, "w") as f:
        f.write(content)
    return path


def list_experiments() -> List[Dict]:
    """List all experiments in the experiments directory."""
    experiments = []
    if not EXPERIMENTS_DIR.exists():
        return experiments

    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if exp_dir.is_dir() and exp_dir.name.startswith("EXP-"):
            config_path = exp_dir / "config.json"
            metrics_path = exp_dir / "metrics.json"

            info = {"id": exp_dir.name, "path": str(exp_dir)}

            if config_path.exists():
                with open(config_path) as f:
                    info["config"] = json.load(f)

            if metrics_path.exists():
                with open(metrics_path) as f:
                    info["metrics"] = json.load(f)

            experiments.append(info)

    return experiments
