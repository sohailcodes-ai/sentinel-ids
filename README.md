# SENTINEL

**SVM-Based Network Intrusion Classification System**

A reproducible supervised-learning pipeline comparing linear and nonlinear (RBF) SVM classifiers on real network-flow data, with controlled experiments, feature analysis, ablation studies, and an interactive analytical interface.

---

## Overview

SENTINEL answers one question:

> Given a network flow represented by numerical traffic features, can an SVM distinguish BENIGN traffic from ATTACK traffic?

The system makes the entire data-mining methodology visible:

```
Raw Data → Clean → Select Features → Sample → Split → Scale → Train → Compare → Evaluate → Analyze
```

---

## Architecture

```
sentinel-ids/
├── app.py              # Streamlit application (11 pages)
├── train.py            # Training CLI
├── api.py              # FastAPI REST service
├── requirements.txt
├── README.md
├── MODEL_CARD.md
├── LICENSE
├── data/
│   └── README.md       # Dataset documentation
├── src/
│   ├── config.py       # Central configuration
│   ├── data.py         # Data loading & validation
│   ├── preprocessing.py # Feature selection, splitting
│   ├── model.py        # SVM training, benchmarks, serialization
│   ├── evaluation.py   # Metrics, threshold analysis, PCA, feature importance
│   ├── experiments.py  # Local experiment tracking
│   ├── analysis.py     # Drift simulation, feature statistics
│   ├── inference.py    # Single-flow & batch inference
│   └── utils.py        # Logging, environment info
├── models/             # Serialized pipelines
├── outputs/            # Evaluation artifacts
├── experiments/        # Experiment tracking
└── tests/              # Unit tests
```

---

## Dataset

**CIC-IDS2017** by the Canadian Institute for Cybersecurity.

- **Source:** https://www.unb.ca/cic/datasets/ids-2017.html
- **Format:** CSV (one day recommended)
- **Labels:** BENIGN vs 14 attack types (normalized to binary)
- **Features:** 15 selected network-flow features

See `data/README.md` for download instructions and provenance.

---

## Machine Learning Pipeline

1. **Data Validation** — discover, validate, normalize columns
2. **Cleaning** — remove duplicates, replace infinity, drop NaN rows
3. **Label Normalization** — map all attack types to ATTACK, preserve originals
4. **Feature Selection** — 15 interpretable network-flow features with alias resolution
5. **Stratified Sampling** — configurable sample size respecting class ratios
6. **Train/Test Split** — 80/20 stratified
7. **Standardization** — StandardScaler fitted on training data only (no leakage)
8. **SVM Training** — Linear SVM + RBF SVM with grid search
9. **Evaluation** — accuracy, precision, recall, F1, confusion matrix, threshold analysis
10. **Serialization** — complete pipeline saved for inference

---

## SVM Models

### Linear SVM
- Linear decision boundary in standardized feature space
- Interpretable via learned coefficients
- Fast training

### RBF SVM
- Nonlinear decision boundaries via kernel trick
- Hyperparameters (C, gamma) selected via cross-validation
- Grid search: C ∈ {0.1, 1, 10}, gamma ∈ {scale, 0.01, 0.1}

### Baselines
- Logistic Regression
- Random Forest

All models use comparable preprocessing and evaluation methodology.

---

## Evaluation

Models are evaluated on a **held-out test set** using:

- Accuracy, Precision, Recall, F1
- Confusion matrix (TP, TN, FP, FN)
- Threshold analysis (precision/recall/F1 vs decision threshold)
- Feature importance (linear coefficients or permutation importance)
- PCA visualization (2D projection for analysis)

---

## Experiments

Every training run generates a unique experiment ID and saves:

```
experiments/EXP-YYYY-MM-DD-HHMMSS/
├── config.json
├── metrics.json
├── feature_list.json
└── README.md
```

### Ablation Studies
- **Baseline:** All 15 features
- **No Flow Rates:** Remove Flow Bytes/s, Flow Packets/s
- **No TCP Flags:** Remove FIN, SYN, RST, ACK counts
- **Reduced:** Core features only

### Multiclass
Optional experiment using original attack taxonomy (DoS, DDoS, PortScan, Bot, etc.)

### Drift Simulation
Controlled distribution shift experiment investigating model behavior when inference distributions differ from training.

---

## Interface

### Streamlit Application (11 pages)

| Page | Purpose |
|------|---------|
| 01 Overview | System status, project statistics |
| 02 Dataset | Data health, class distribution, feature distributions |
| 03 Model Lab | Linear vs RBF comparison, grid search results |
| 04 Evaluation | Confusion matrix, metrics, threshold analysis |
| 05 Traffic Analyzer | Single-flow classification with decision score |
| 06 Batch Inference | CSV upload, batch prediction, download results |
| 07 Feature Analysis | Feature importance, PCA visualization |
| 08 Experiments | Experiment history from local artifacts |
| 09 Benchmark | SVM vs Logistic Regression vs Random Forest |
| 10 Drift Analysis | Simulated distribution shift experiment |
| 11 Methodology | Technical explanation of the complete pipeline |

### REST API (FastAPI)

```
GET  /health          — Model status
GET  /metadata        — Model metadata
POST /predict         — Single flow classification
POST /batch-predict   — CSV batch classification
```

---

## Setup

### Prerequisites
- Python 3.11+
- CIC-IDS2017 dataset CSV

### Installation

```bash
git clone https://github.com/sohailcodes-ai/sentinel-ids.git
cd sentinel-ids
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Dataset Setup

1. Download from [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Place one daily CSV as `data/Friday-WorkingHours-.pcap_ISCX.csv`
3. See `data/README.md` for details

---

## Training

```bash
# Basic training (Linear vs RBF SVM)
python train.py --sample-size 20000 --random-state 42

# With benchmark models
python train.py --benchmark --sample-size 20000

# Full experiment suite (SVM + benchmark + ablation + multiclass)
python train.py --full --sample-size 20000

# Ablation only
python train.py --ablation --sample-size 20000

# Multiclass only
python train.py --multiclass --sample-size 20000
```

---

## Running the Application

```bash
streamlit run app.py
```

## Running the API

```bash
uvicorn api:app --reload
```

## Running Tests

```bash
pytest tests/ -v
```

---

## Reproducibility

- Random seed is configurable (`--random-state`, default: 42)
- Dataset hash recorded in metadata
- Python version and package versions logged
- Train/test splits are deterministic with same seed
- Scaler fitted on training data only
- All artifacts saved with timestamps

---

## Limitations

- Binary classification (attack subtypes collapsed by default)
- Single-day dataset subset
- SVC training is expensive on large data without sampling
- Offline analysis only, not a production IDS
- Decision function scores are NOT probabilities
- Class imbalance present in real network data
- Results are generated locally and intentionally not hardcoded

---

## License

MIT License — see [LICENSE](LICENSE)

Dataset licensing/provenance is separate from the software license.
