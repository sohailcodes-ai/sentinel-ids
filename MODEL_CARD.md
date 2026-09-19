# SENTINEL Model Card

## Model Purpose

SENTINEL is an offline network-flow classification system. Given numerical traffic features extracted from a network flow record, the model classifies it as **BENIGN** or **ATTACK** using Support Vector Machines.

## Intended Use

- Offline analysis of pre-captured network flow records
- Data mining experimentation and SVM methodology study
- Educational demonstration of supervised learning on network data

## Non-Intended Use

- Production intrusion detection or prevention
- Real-time packet interception or network monitoring
- Autonomous security response
- Penetration testing or offensive security
- Live network traffic analysis

## Dataset

**CIC-IDS2017** from the Canadian Institute for Cybersecurity (UNB).

- Source: https://www.unb.ca/cic/datasets/ids-2017.html
- One daily CSV subset is used (e.g., Friday working hours)
- Labels normalized to binary: BENIGN vs ATTACK
- Original attack taxonomy preserved in `Original_Label` column

## Features

15 network flow features:

| Feature | Description |
|---------|-------------|
| Flow Duration | Microseconds |
| Total Fwd Packets | Count |
| Total Backward Packets | Count |
| Total Length of Fwd Packets | Bytes |
| Total Length of Bwd Packets | Bytes |
| Fwd Packet Length Mean | Bytes |
| Bwd Packet Length Mean | Bytes |
| Flow Bytes/s | Rate |
| Flow Packets/s | Rate |
| Packet Length Mean | Bytes |
| Packet Length Std | Bytes |
| FIN Flag Count | Count |
| SYN Flag Count | Count |
| RST Flag Count | Count |
| ACK Flag Count | Count |

## Training Procedure

1. CSV loaded and validated
2. Columns normalized, duplicates removed
3. Infinity values replaced, NaN rows dropped
4. Labels mapped to binary (BENIGN=0, ATTACK=1)
5. Stratified sampling (default: 20,000 rows)
6. 80/20 stratified train/test split
7. StandardScaler fitted on training data only
8. Linear SVM and RBF SVM trained
9. RBF hyperparameters selected via 3-fold cross-validation on training set
10. Best model selected by F1 score
11. Final evaluation on held-out test set

## Metrics

Metrics are generated locally by `python train.py` and are intentionally not hardcoded. After training, metrics appear in:

- `outputs/metrics.json`
- `outputs/model_comparison.csv`
- `outputs/classification_report.csv`

## Limitations

- Binary classification only (attack subtypes collapsed)
- Single-day dataset subset
- SVC training is expensive without sampling
- Offline analysis only, not real-time
- Class imbalance present (BENIGN typically dominates)
- Decision function scores are NOT probabilities
- Feature distributions may shift over time (distribution drift)

## Known Failure Modes

- **False Positives:** Benign traffic classified as attack, causing false alerts
- **False Negatives:** Attack traffic classified as benign, missing actual intrusions
- **Rare attack types** may be underrepresented in the training sample
- **Distribution shift** can degrade performance on traffic from different time periods
