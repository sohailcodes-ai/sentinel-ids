# Dataset

## Overview

SENTINEL expects a network intrusion detection dataset in CSV format. The primary supported dataset is **CIC-IDS2017**.

## Source

- **Name:** CIC-IDS2017 (Canadian Institute for Cybersecurity Intrusion Detection System 2017)
- **URL:** https://www.unb.ca/cic/datasets/ids-2017.html
- **Paper:** Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. ICISSP 2018.

## Expected File

Place one of the daily CSV files in this directory:

```
data/Friday-WorkingHours-.pcap_ISCX.csv
```

Other supported filenames (any single day from CIC-IDS2017):
- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv`

## Download Instructions

1. Visit https://www.unb.ca/cic/datasets/ids-2017.html
2. Download the dataset (PCAP files + CSV files)
3. Extract the CSV files
4. Place **one** daily CSV file in the `data/` directory
5. Rename it to `Friday-WorkingHours-.pcap_ISCX.csv` if using a different day

## Why a Single Day?

The full CIC-IDS2017 dataset spans an entire week and contains millions of rows. For a college Data Mining project focused on SVM classification:

- **Memory:** Full dataset exceeds typical laptop RAM (8GB)
- **Training time:** SVC training on millions of rows is prohibitively slow on CPU
- **Pedagogical clarity:** A single day provides sufficient class diversity while remaining manageable
- **Reproducibility:** Smaller data enables faster iteration

## Label Column

The dataset uses a column named `Label` with values like:
- `BENIGN` - normal traffic
- `DoS Hulk`, `DDoS`, `PortScan`, `Bot`, `Infiltration`, `Web Attack - Brute Force`, etc.

SENTINEL normalizes all non-BENIGN labels to `ATTACK`.

## Feature Columns

SENTINEL uses 15 selected features (with alias resolution):

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

## Known Limitations

- Column names may have trailing whitespace; the loader normalizes these
- Some columns contain infinity values; these are replaced with NaN and dropped
- The `Flow Bytes/s` and `Flow Packets/s` columns may have different exact names across days
- Encoding may vary; the loader attempts latin-1 as fallback

## Dataset Absence

If the dataset is not present, SENTINEL will display a clear "Dataset Not Configured" state in the application and the training CLI will exit with explicit instructions. No synthetic data is ever generated as a fallback.
