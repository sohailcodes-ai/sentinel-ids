"""
SENTINEL Configuration
Central configuration for dataset, features, paths, and experiment parameters.
"""

from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

DATASET_FILENAME = "Friday-WorkingHours-.pcap_ISCX.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME

LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"
ORIGINAL_LABEL_COLUMN = "Original_Label"

ATTACK_ALIASES: Dict[str, str] = {
    "dos slowloris": "ATTACK",
    "dos slowhttptest": "ATTACK",
    "dos hulk": "ATTACK",
    "dos goldeneye": "ATTACK",
    "ftppatric": "ATTACK",
    "ssh-patric": "ATTACK",
    "web attack - brute force": "ATTACK",
    "web attack \u2013 brute force": "ATTACK",
    "web attack \u2013 xss": "ATTACK",
    "web attack - xss": "ATTACK",
    "web attack \u2013 sql injection": "ATTACK",
    "web attack - sql injection": "ATTACK",
    "infiltration": "ATTACK",
    "heartbleed": "ATTACK",
    "portscan": "ATTACK",
    "portscan ": "ATTACK",
    "ddos": "ATTACK",
    "bot": "ATTACK",
}

MULTICLASS_MAP: Dict[str, str] = {
    "benign": "BENIGN",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "ddos": "DDoS",
    "portscan": "PortScan",
    "portscan ": "PortScan",
    "bot": "Bot",
    "infiltration": "Infiltration",
    "heartbleed": "Heartbleed",
    "web attack - brute force": "Web Attack",
    "web attack \u2013 brute force": "Web Attack",
    "web attack \u2013 xss": "Web Attack",
    "web attack - xss": "Web Attack",
    "web attack \u2013 sql injection": "Web Attack",
    "web attack - sql injection": "Web Attack",
    "ftppatric": "FTP-Patator",
    "ssh-patric": "SSH-Patator",
}

FEATURE_ALIASES: Dict[str, List[str]] = {
    "Flow Duration": ["Flow Duration", "flow duration"],
    "Total Fwd Packets": ["Total Fwd Packets", "total fwd packets"],
    "Total Backward Packets": ["Total Backward Packets", "total backward packets"],
    "Total Length of Fwd Packets": ["Total Length of Fwd Packets", "total length of fwd packets"],
    "Total Length of Bwd Packets": ["Total Length of Bwd Packets", "total length of bwd packets"],
    "Fwd Packet Length Mean": ["Fwd Packet Length Mean", "fwd packet length mean"],
    "Bwd Packet Length Mean": ["Bwd Packet Length Mean", "bwd packet length mean"],
    "Flow Bytes/s": ["Flow Bytes/s", "flow bytes/s", "Flow Bytes/s "],
    "Flow Packets/s": ["Flow Packets/s", "flow packets/s", "Flow Packets/s "],
    "Packet Length Mean": ["Packet Length Mean", "packet length mean"],
    "Packet Length Std": ["Packet Length Std", "packet length std"],
    "FIN Flag Count": ["FIN Flag Count", "fin flag count"],
    "SYN Flag Count": ["SYN Flag Count", "syn flag count"],
    "RST Flag Count": ["RST Flag Count", "rst flag count"],
    "ACK Flag Count": ["ACK Flag Count", "ack flag count"],
}

SELECTED_FEATURES: List[str] = list(FEATURE_ALIASES.keys())

FEATURE_GROUPS: Dict[str, List[str]] = {
    "flow_duration": ["Flow Duration"],
    "packet_counts": [
        "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    ],
    "packet_stats": [
        "Fwd Packet Length Mean", "Bwd Packet Length Mean",
        "Packet Length Mean", "Packet Length Std",
    ],
    "flow_rates": ["Flow Bytes/s", "Flow Packets/s"],
    "tcp_flags": ["FIN Flag Count", "SYN Flag Count", "RST Flag Count", "ACK Flag Count"],
}

ABLATION_CONFIGS: Dict[str, Dict] = {
    "baseline": {
        "description": "All selected features",
        "exclude_groups": [],
    },
    "no_flow_rates": {
        "description": "Remove flow rate features (Flow Bytes/s, Flow Packets/s)",
        "exclude_groups": ["flow_rates"],
    },
    "no_tcp_flags": {
        "description": "Remove TCP flag count features",
        "exclude_groups": ["tcp_flags"],
    },
    "reduced": {
        "description": "Core features only: duration, packet counts, and packet stats",
        "exclude_groups": ["flow_rates", "tcp_flags"],
    },
}

DEFAULT_SAMPLE_SIZE: int = 20000
DEFAULT_RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 3

RBF_GRID: Dict[str, list] = {
    "C": [0.1, 1.0, 10.0],
    "gamma": ["scale", 0.01, 0.1],
}

PIPELINE_PATH = MODELS_DIR / "svm_pipeline.joblib"
MULTICLASS_PIPELINE_PATH = MODELS_DIR / "multiclass_svm_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
COMPARISON_PATH = OUTPUTS_DIR / "model_comparison.csv"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = OUTPUTS_DIR / "classification_report.csv"
TRAINING_SUMMARY_PATH = OUTPUTS_DIR / "training_summary.json"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_PLOT_PATH = OUTPUTS_DIR / "feature_importance.png"
THRESHOLD_ANALYSIS_PATH = OUTPUTS_DIR / "threshold_analysis.csv"
THRESHOLD_PLOT_PATH = OUTPUTS_DIR / "threshold_analysis.png"
PCA_VISUALIZATION_PATH = OUTPUTS_DIR / "pca_visualization.png"

DESCRIPTION = (
    "SVM-Based Network Intrusion Classification System \u2014 "
    "a reproducible supervised-learning pipeline comparing "
    "linear and nonlinear (RBF) SVM classifiers on real network-flow data."
)
