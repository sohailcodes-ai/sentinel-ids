"""
SENTINEL Tests - Data module
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.config import LABEL_COLUMN, SELECTED_FEATURES, BENIGN_LABEL, ORIGINAL_LABEL_COLUMN
from src.data import (
    normalize_label, normalize_multiclass_label, resolve_feature_columns,
    sample_stratified, normalize_column_name,
)


class TestColumnNormalization:
    def test_strip_whitespace(self):
        assert normalize_column_name("  Flow Duration  ") == "Flow Duration"
        assert normalize_column_name("Flow Bytes/s ") == "Flow Bytes/s"
        assert normalize_column_name("no_change") == "no_change"


class TestLabelNormalization:
    def test_benign_variants(self):
        for v in ["BENIGN", "benign", " Benign ", "", "BENIGN "]:
            assert normalize_label(v) == "BENIGN"

    def test_attack_variants(self):
        attacks = ["DoS Hulk", "DDoS", "PortScan", "Bot", "Infiltration",
                    "Web Attack - Brute Force", "Heartbleed", "FTP-Patator", "SSH-Patator"]
        for a in attacks:
            assert normalize_label(a) == "ATTACK"

    def test_unknown_is_attack(self):
        assert normalize_label("SomeUnknownAttack") == "ATTACK"

    def test_non_string(self):
        assert normalize_label(None) == "BENIGN"
        assert normalize_label(42) == "BENIGN"
        assert normalize_label(3.14) == "BENIGN"


class TestMulticlassLabelNormalization:
    def test_known_labels(self):
        assert normalize_multiclass_label("benign") == "BENIGN"
        assert normalize_multiclass_label("dos hulk") == "DoS"
        assert normalize_multiclass_label("ddos") == "DDoS"
        assert normalize_multiclass_label("portscan") == "PortScan"
        assert normalize_multiclass_label("bot") == "Bot"

    def test_unknown_returns_other(self):
        assert normalize_multiclass_label("UnknownAttack") == "Other"


class TestFeatureResolution:
    def test_resolve_all(self):
        cols = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                "Total Length of Fwd Packets", "Total Length of Bwd Packets",
                "Fwd Packet Length Mean", "Bwd Packet Length Mean",
                "Flow Bytes/s", "Flow Packets/s", "Packet Length Mean",
                "Packet Length Std", "FIN Flag Count", "SYN Flag Count",
                "RST Flag Count", "ACK Flag Count", "Label"]
        resolved, missing = resolve_feature_columns(cols)
        assert len(resolved) == 15
        assert len(missing) == 0

    def test_resolve_whitespace_aliases(self):
        cols = ["Flow Bytes/s ", "Flow Packets/s ", "Flow Duration", "Label"]
        resolved, missing = resolve_feature_columns(cols)
        assert "Flow Bytes/s " in resolved
        assert "Flow Packets/s " in resolved
        assert len(missing) == 12  # Flow Duration, Flow Bytes/s, Flow Packets/s resolved

    def test_all_missing(self):
        cols = ["Unrelated", "Columns"]
        resolved, missing = resolve_feature_columns(cols)
        assert len(resolved) == 0
        assert len(missing) == 15

    def test_partial_resolution(self):
        cols = ["Flow Duration", "Total Fwd Packets", "Label"]
        resolved, missing = resolve_feature_columns(cols)
        assert len(resolved) == 2
        assert len(missing) == 13


class TestSampling:
    def test_preserves_ratios(self):
        df = pd.DataFrame({
            "feat1": range(1000),
            LABEL_COLUMN: ["BENIGN"] * 800 + ["ATTACK"] * 200,
        })
        sampled = sample_stratified(df, sample_size=200, random_state=42)
        ratio = sampled[LABEL_COLUMN].value_counts().get("ATTACK", 0) / len(sampled)
        assert 0.15 < ratio < 0.35

    def test_oversample_limit(self):
        df = pd.DataFrame({"feat1": range(50), LABEL_COLUMN: ["BENIGN"] * 40 + ["ATTACK"] * 10})
        sampled = sample_stratified(df, sample_size=100, random_state=42)
        assert len(sampled) == 50

    def test_exact_sample_size(self):
        df = pd.DataFrame({"feat1": range(1000), LABEL_COLUMN: ["BENIGN"] * 800 + ["ATTACK"] * 200})
        sampled = sample_stratified(df, sample_size=500, random_state=42)
        assert len(sampled) == 500
