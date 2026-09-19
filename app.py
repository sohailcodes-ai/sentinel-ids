#!/usr/bin/env python3
"""
SENTINEL Streamlit Application
SVM-Based Network Intrusion Classification - Interactive Analysis Interface
11 pages: Overview, Dataset, Model Lab, Evaluation, Traffic Analyzer,
Batch Inference, Feature Analysis, Experiments, Benchmark, Drift Analysis, Methodology
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import io
import os
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SENTINEL", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

from src.config import (
    DATASET_PATH, FEATURE_ALIASES, METADATA_PATH, MODELS_DIR, OUTPUTS_DIR,
    SELECTED_FEATURES, FEATURE_GROUPS, ABLATION_CONFIGS, EXPERIMENTS_DIR,
)
from src.data import load_and_validate
from src.model import load_metadata, load_pipeline, get_model_info
from src.inference import predict_single_flow, predict_batch_csv

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{--bg0:#0a0a0b;--bg1:#111113;--bg2:#16161a;--bg3:#1c1c21;--brd:#27272a;--brd2:#1e1e22;--t1:#fafafa;--t2:#a1a1aa;--t3:#71717a;--acc:#3b82f6;--grn:#22c55e;--red:#ef4444;--ylw:#eab308;--fs:'Inter',sans-serif;--fm:'JetBrains Mono',monospace}
.stApp{background:var(--bg0);color:var(--t1);font-family:var(--fs)}
[data-testid="stSidebar"]{background:var(--bg1);border-right:1px solid var(--brd2)}
[data-testid="stSidebar"] .stRadio label{font-family:var(--fs);font-size:.85rem;color:var(--t2);padding:.25rem 0}
[data-testid="stSidebar"] .stRadio label[data-checked="true"]{color:var(--t1);font-weight:600}
h1,h2,h3,h4,h5,h6{font-family:var(--fs)!important;color:var(--t1)!important}
h1{font-weight:800!important;letter-spacing:-.03em!important}
h2{font-weight:700!important;letter-spacing:-.02em!important;font-size:1.4rem!important}
h3{font-weight:600!important;font-size:1.1rem!important}
.stMetric{background:var(--bg2);border:1px solid var(--brd2);border-radius:10px;padding:1rem 1.2rem}
.stMetric label{font-family:var(--fs)!important;font-size:.75rem!important;color:var(--t3)!important;text-transform:uppercase;letter-spacing:.06em;font-weight:500!important}
.stMetric [data-testid="stMetricValue"]{font-family:var(--fm)!important;font-size:1.6rem!important;font-weight:700!important;color:var(--t1)!important}
.block-container{padding-top:2rem!important;max-width:1200px}
.stTabs [data-baseweb="tab-list"]{gap:0;background:var(--bg2);border-radius:8px;padding:3px;border:1px solid var(--brd2)}
.stTabs [data-baseweb="tab"]{font-family:var(--fs);font-size:.82rem;font-weight:500;color:var(--t3);border-radius:6px;padding:.5rem 1rem}
.stTabs [aria-selected="true"]{background:var(--bg3)!important;color:var(--t1)!important;font-weight:600!important;border:1px solid var(--brd)!important}
div[data-testid="stDataFrame"]{border:1px solid var(--brd2);border-radius:8px;overflow:hidden}
code,.stCode{font-family:var(--fm)!important;font-size:.82rem!important}
hr{border:none;border-top:1px solid var(--brd2);margin:1.5rem 0}
.stAlert>div{background:var(--bg2)!important;border:1px solid var(--brd)!important}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource
def _load_pipeline_cached():
    try: return load_pipeline()
    except FileNotFoundError: return None

def _has_pipeline(): return (MODELS_DIR / "svm_pipeline.joblib").exists()
def _has_artifacts(): return (OUTPUTS_DIR / "training_summary.json").exists()

def _load_training_summary() -> Dict:
    p = OUTPUTS_DIR / "training_summary.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return {}

def _load_meta() -> Dict:
    try: return load_metadata()
    except: return {}

def _load_data_report():
    try: _, r = load_and_validate(); return r
    except: return None

def _load_comparison_df():
    p = OUTPUTS_DIR / "model_comparison.csv"
    if p.exists(): return pd.read_csv(p)
    return None

def _load_threshold_df():
    p = OUTPUTS_DIR / "threshold_analysis.csv"
    if p.exists(): return pd.read_csv(p)
    return None

def _load_feature_importance():
    p = OUTPUTS_DIR / "feature_importance.csv"
    if p.exists(): return pd.read_csv(p)
    return None

def _load_ablation():
    p = OUTPUTS_DIR / "ablation_results.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return []

def _load_experiments():
    experiments = []
    if not EXPERIMENTS_DIR.exists(): return experiments
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("EXP-"):
            info = {"id": d.name}
            cp = d / "config.json"
            mp = d / "metrics.json"
            if cp.exists():
                with open(cp) as f: info["config"] = json.load(f)
            if mp.exists():
                with open(mp) as f: info["metrics"] = json.load(f)
            experiments.append(info)
    return experiments

FD = {
    "Flow Duration": "Total duration of the flow in microseconds",
    "Total Fwd Packets": "Number of packets sent in the forward direction",
    "Total Backward Packets": "Number of packets sent in the backward direction",
    "Total Length of Fwd Packets": "Total bytes in forward packets",
    "Total Length of Bwd Packets": "Total bytes in backward packets",
    "Fwd Packet Length Mean": "Mean size of forward packets (bytes)",
    "Bwd Packet Length Mean": "Mean size of backward packets (bytes)",
    "Flow Bytes/s": "Average flow rate in bytes per second",
    "Flow Packets/s": "Average packet rate in packets per second",
    "Packet Length Mean": "Mean packet length across the flow",
    "Packet Length Std": "Standard deviation of packet lengths",
    "FIN Flag Count": "Number of FIN flags observed",
    "SYN Flag Count": "Number of SYN flags observed",
    "RST Flag Count": "Number of RST flags observed",
    "ACK Flag Count": "Number of ACK flags observed",
}

with st.sidebar:
    st.markdown("# SENTINEL")
    st.caption("SVM Network Intrusion Classification")
    st.markdown("---")
    nav = st.radio("Navigation", [
        "01  Overview", "02  Dataset", "03  Model Lab", "04  Evaluation",
        "05  Traffic Analyzer", "06  Batch Inference", "07  Feature Analysis",
        "08  Experiments", "09  Benchmark", "10  Drift Analysis", "11  Methodology",
    ], label_visibility="collapsed")
    st.markdown("---")
    p_ok = _has_pipeline(); a_ok = _has_artifacts()
    dg = '<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#22c55e;margin-right:6px"></span>'
    dr = '<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#ef4444;margin-right:6px"></span>'
    dy = '<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#eab308;margin-right:6px"></span>'
    st.markdown(f'{dg if p_ok else dr} MODEL {"READY" if p_ok else "NOT TRAINED"}', unsafe_allow_html=True)
    st.markdown(f'{dg if a_ok else dy} ARTIFACTS {"READY" if a_ok else "MISSING"}', unsafe_allow_html=True)

# === PAGE 01: OVERVIEW ===
if nav.startswith("01"):
    st.markdown("# SENTINEL")
    st.markdown("### SVM-Based Network Intrusion Classification")
    st.markdown("---")
    summary = _load_training_summary()
    vr = _load_data_report()
    if not summary:
        st.warning("**Dataset not configured or training not yet run.** Place the CIC-IDS2017 CSV in `data/` and run `python train.py`.")
        st.markdown("""**Quick start:**
1. Download from [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Place CSV as `data/Friday-WorkingHours-.pcap_ISCX.csv`
3. Run `python train.py`
4. Run `streamlit run app.py`""")
        st.markdown("---")
        st.markdown("#### Pipeline Architecture")
        steps = ["DATA","CLEAN","SELECT","SAMPLE","SPLIT","SCALE","TRAIN","EVALUATE","SERIALIZE"]
        cols = st.columns(len(steps))
        for i,(col,s) in enumerate(zip(cols,steps)):
            with col:
                st.markdown(f'<div style="background:#16161a;border:1px solid #27272a;border-radius:8px;padding:0.8rem;text-align:center"><div style="font-size:0.65rem;color:#71717a;text-transform:uppercase;letter-spacing:0.05em">STEP {i+1}</div><div style="font-family:monospace;font-weight:600;color:#fafafa;font-size:0.85rem;margin-top:4px">{s}</div></div>', unsafe_allow_html=True)
    else:
        m = summary.get("metrics",{})
        v = summary.get("validation_report",{})
        st.markdown("#### System Status")
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(f'<div style="background:#16161a;border:1px solid #27272a;border-radius:8px;padding:12px"><span style="font-size:0.7rem;color:#71717a;text-transform:uppercase;letter-spacing:0.06em">Dataset</span><br><span style="font-family:monospace;font-size:0.9rem;color:#22c55e;font-weight:600">LOADED</span></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div style="background:#16161a;border:1px solid #27272a;border-radius:8px;padding:12px"><span style="font-size:0.7rem;color:#71717a;text-transform:uppercase;letter-spacing:0.06em">Model</span><br><span style="font-family:monospace;font-size:0.9rem;color:#22c55e;font-weight:600">{summary.get("best_model","?")}</span></div>', unsafe_allow_html=True)
        with c3: st.markdown('<div style="background:#16161a;border:1px solid #27272a;border-radius:8px;padding:12px"><span style="font-size:0.7rem;color:#71717a;text-transform:uppercase;letter-spacing:0.06em">Artifacts</span><br><span style="font-family:monospace;font-size:0.9rem;color:#22c55e;font-weight:600">READY</span></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### Project Statistics")
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("Dataset Rows", f"{v.get('cleaned_rows',0):,}")
        mc2.metric("Features", v.get("num_features",0))
        mc3.metric("Training Rows", f"{summary.get('train_size',0):,}")
        mc4.metric("Test Rows", f"{summary.get('test_size',0):,}")
        mc5,mc6,mc7,mc8 = st.columns(4)
        mc5.metric("Model", summary.get("best_model","?"))
        mc6.metric("Accuracy", f"{m.get('accuracy',0):.4f}")
        mc7.metric("F1 Score", f"{m.get('f1',0):.4f}")
        mc8.metric("Recall", f"{m.get('recall',0):.4f}")
        st.markdown("---")
        st.markdown("#### Class Distribution")
        cd = v.get("class_distribution",{})
        if cd:
            cdf = pd.DataFrame({"Class": list(cd.keys()), "Count": list(cd.values())})
            st.bar_chart(cdf.set_index("Class"))

# === PAGE 02: DATASET ===
elif nav.startswith("02"):
    st.markdown("## Dataset")
    st.markdown("---")
    dr = _load_data_report()
    if not dr:
        st.warning("Dataset not found. Place the CIC-IDS2017 CSV in `data/`.")
        st.markdown("**Expected:** `data/Friday-WorkingHours-.pcap_ISCX.csv`\n\n**Source:** [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)\n\nSee `data/README.md` for setup instructions.")
    else:
        st.markdown("#### Dataset Health")
        h1,h2,h3,h4 = st.columns(4)
        h1.metric("Raw Rows", f"{dr['raw_rows']:,}")
        h2.metric("Cleaned Rows", f"{dr['cleaned_rows']:,}")
        h3.metric("Removed", f"{dr['removed_rows']:,}")
        h4.metric("Duplicates", f"{dr['duplicates_removed']:,}")
        h5,h6,h7,h8 = st.columns(4)
        h5.metric("Inf Replaced", dr['inf_replaced'])
        h6.metric("NaN Dropped", dr['nan_rows_dropped'])
        h7.metric("Imbalance Ratio", f"{dr['imbalance_ratio']}:1")
        h8.metric("Features", dr['num_features'])
        st.markdown("---")
        st.markdown("#### Class Distribution")
        dist = dr.get("class_distribution",{})
        if dist:
            c1,c2 = st.columns([1,1])
            with c1:
                cdf = pd.DataFrame({"Class": list(dist.keys()), "Count": list(dist.values())})
                st.dataframe(cdf, use_container_width=True, hide_index=True)
            with c2:
                st.bar_chart(cdf.set_index("Class"))
        st.markdown("---")
        st.markdown("#### Original Attack Taxonomy")
        orig = dr.get("original_class_distribution",{})
        if orig:
            odf = pd.DataFrame({"Attack Type": list(orig.keys()), "Count": list(orig.values())})
            st.dataframe(odf, use_container_width=True, hide_index=True)
        st.markdown("---")
        st.markdown("#### Feature Inventory")
        resolved = dr.get("features_resolved",[])
        missing = dr.get("features_missing",[])
        feat_data = [{"Feature": f, "Status": "Available" if f in resolved else "Missing", "Description": FD.get(f,"")} for f in SELECTED_FEATURES]
        st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)
        if missing: st.info(f"Missing features: {', '.join(missing)}")
        st.markdown("---")
        st.markdown("#### Feature Distributions")
        try:
            df_full, _ = load_and_validate()
            sel = st.selectbox("Select feature", resolved)
            if sel:
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{sel}** - All traffic")
                    sd = df_full[sel].dropna()
                    if len(sd) > 5000: sd = sd.sample(5000, random_state=42)
                    st.area_chart(sd.reset_index(drop=True))
                with c2:
                    st.markdown(f"**{sel}** - By class")
                    for label in df_full["Label"].unique():
                        sub = df_full[df_full["Label"] == label][sel].dropna()
                        if len(sub) > 2000: sub = sub.sample(2000, random_state=42)
                        st.line_chart(sub.reset_index(drop=True))
        except: st.warning("Could not load dataset for distribution plots.")

# === PAGE 03: MODEL LAB ===
elif nav.startswith("03"):
    st.markdown("## Model Lab")
    st.markdown("---")
    summary = _load_training_summary()
    comp = _load_comparison_df()
    if not summary:
        st.warning("No training results. Run `python train.py` first.")
    else:
        st.markdown("### Linear SVM vs RBF SVM")
        st.markdown("Two SVM classifiers trained and compared using stratified train/test splits.")
        st.markdown("---")
        if comp is not None:
            st.markdown("#### Experiment Results")
            st.dataframe(comp, use_container_width=True, hide_index=True)
        linear = summary.get("linear_results",{})
        rbf = summary.get("rbf_results",{})
        st.markdown("#### Side-by-Side")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Linear SVM**")
            st.metric("Accuracy", f"{linear.get('accuracy',0):.4f}")
            st.metric("F1", f"{linear.get('f1',0):.4f}")
            st.metric("Precision", f"{linear.get('precision',0):.4f}")
            st.metric("Recall", f"{linear.get('recall',0):.4f}")
            st.metric("Train Time", f"{linear.get('train_time',0):.3f}s")
        with c2:
            st.markdown("**RBF SVM**")
            st.metric("Accuracy", f"{rbf.get('accuracy',0):.4f}")
            st.metric("F1", f"{rbf.get('f1',0):.4f}")
            st.metric("Precision", f"{rbf.get('precision',0):.4f}")
            st.metric("Recall", f"{rbf.get('recall',0):.4f}")
            st.metric("Train Time", f"{rbf.get('train_time',0):.3f}s")
        st.markdown("---")
        st.markdown("#### What Changed")
        st.markdown(f"- **Linear SVM:** C=1.0, kernel=linear. Linear hyperplane in standardized feature space.")
        st.markdown(f"- **RBF SVM:** C={rbf.get('C','?')}, gamma={rbf.get('gamma','?')}, kernel=rbf. Nonlinear boundaries via kernel trick.")
        st.markdown(f"- **Selected:** {summary.get('best_model','?')} by F1 score.")
        st.markdown("---")
        st.markdown("#### Model Details")
        meta = _load_meta()
        if meta:
            info = meta.get("model_info",{})
            st.json({"kernel": info.get("kernel","?"), "C": info.get("C","?"), "gamma": info.get("gamma","N/A"),
                      "n_support_vectors": info.get("n_support_vectors","?"),
                      "support_vectors_per_class": info.get("support_vectors_per_class",{}),
                      "n_features": info.get("n_features","?")})
        st.markdown("---")
        st.markdown("#### RBF Grid Search Results")
        rbf_search = summary.get("rbf_search",[])
        if rbf_search:
            st.dataframe(pd.DataFrame(rbf_search), use_container_width=True, hide_index=True)

# === PAGE 04: EVALUATION ===
elif nav.startswith("04"):
    st.markdown("## Evaluation")
    st.markdown("---")
    summary = _load_training_summary()
    if not summary:
        st.warning("No evaluation results. Run `python train.py` first.")
    else:
        m = summary.get("metrics",{})
        st.markdown("#### Metric Cards")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy", f"{m.get('accuracy',0):.4f}")
        c2.metric("Precision", f"{m.get('precision',0):.4f}")
        c3.metric("Recall", f"{m.get('recall',0):.4f}")
        c4.metric("F1 Score", f"{m.get('f1',0):.4f}")
        st.markdown("---")
        st.markdown("#### Confusion Matrix")
        cm_path = OUTPUTS_DIR / "confusion_matrix.png"
        if cm_path.exists(): st.image(str(cm_path), width=500)
        else: st.info("Confusion matrix not found.")
        st.markdown("---")
        st.markdown("#### Classification Report")
        cr_path = OUTPUTS_DIR / "classification_report.csv"
        if cr_path.exists():
            st.dataframe(pd.read_csv(cr_path), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.markdown("#### Error Analysis")
        e1,e2,e3,e4 = st.columns(4)
        e1.metric("True Positives", m.get("true_positives",0))
        e2.metric("True Negatives", m.get("true_negatives",0))
        e3.metric("False Positives", m.get("false_positives",0))
        e4.metric("False Negatives", m.get("false_negatives",0))
        st.markdown("""**Why this matters:**
- **False Positives:** legitimate traffic flagged as attack - causes alert fatigue
- **False Negatives:** attack traffic missed - real security risk
- In IDS, **recall** (catching attacks) is often prioritized over precision
- **F1** balances both concerns""")
        st.markdown("---")
        st.markdown("#### Threshold Analysis")
        tdf = _load_threshold_df()
        if tdf is not None:
            st.dataframe(tdf, use_container_width=True, hide_index=True)
            thr_path = OUTPUTS_DIR / "threshold_analysis.png"
            if thr_path.exists(): st.image(str(thr_path), width=600)
        else: st.info("Threshold analysis not available. Run training first.")

# === PAGE 05: TRAFFIC ANALYZER ===
elif nav.startswith("05"):
    st.markdown("## Traffic Analyzer")
    st.markdown("---")
    pipe = _load_pipeline_cached()
    if pipe is None:
        st.warning("No trained model. Run `python train.py` first.")
    else:
        st.markdown("Enter network flow features to classify as **BENIGN** or **ATTACK**.")
        st.markdown("---")
        input_vals = {}
        cols = st.columns(3)
        for i, feat in enumerate(SELECTED_FEATURES):
            with cols[i % 3]:
                input_vals[feat] = st.number_input(label=feat, value=0.0, help=FD.get(feat,""), format="%.4f")
        st.markdown("---")
        if st.button("Classify Flow", type="primary"):
            result = predict_single_flow(pipe, input_vals)
            if result.get("error"):
                st.error(f"Validation failed. Missing: {result.get('missing_features',[])} Invalid: {result.get('invalid_features',[])}")
            else:
                label = result["prediction"]
                if label == "BENIGN":
                    st.markdown(f'<div style="background:rgba(34,197,94,0.08);border:2px solid #22c55e;border-radius:10px;padding:1.5rem;text-align:center;font-family:monospace;font-weight:700;font-size:1.4rem;color:#22c55e;letter-spacing:0.04em">PREDICTION: {label}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="background:rgba(239,68,68,0.08);border:2px solid #ef4444;border-radius:10px;padding:1.5rem;text-align:center;font-family:monospace;font-weight:700;font-size:1.4rem;color:#ef4444;letter-spacing:0.04em">PREDICTION: {label}</div>', unsafe_allow_html=True)
                st.markdown("---")
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown("**Decision Function Score:**")
                    st.code(f"{result['decision_score']:.6f}")
                    st.caption("Signed distance from decision boundary. NOT a probability.")
                with c2:
                    st.markdown("**Model:**")
                    st.code(result["model"])
                st.markdown("---")
                st.markdown("**Input Vector:**")
                st.dataframe(pd.DataFrame(input_vals, index=[0]), use_container_width=True, hide_index=True)

# === PAGE 06: BATCH INFERENCE ===
elif nav.startswith("06"):
    st.markdown("## Batch Inference")
    st.markdown("---")
    pipe = _load_pipeline_cached()
    if pipe is None:
        st.warning("No trained model. Run `python train.py` first.")
    else:
        st.markdown("Upload a CSV with the required network-flow features.")
        st.markdown(f"**Required features:** {', '.join(SELECTED_FEATURES)}")
        st.markdown("---")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            try:
                result_df, info = predict_batch_csv(pipe, tmp_path)
            finally:
                os.unlink(tmp_path)
            if info is None or info.get("error"):
                st.error(f"Error: {info}")
            else:
                st.markdown("#### Results Summary")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Total Rows", info["total_rows"])
                c2.metric("Valid Rows", info["valid_rows"])
                c3.metric("BENIGN", info["predictions"]["BENIGN"])
                c4.metric("ATTACK", info["predictions"]["ATTACK"])
                st.markdown("---")
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                csv_out = result_df.to_csv(index=False)
                st.download_button("Download Results CSV", csv_out, "sentinel_predictions.csv", "text/csv")

# === PAGE 07: FEATURE ANALYSIS ===
elif nav.startswith("07"):
    st.markdown("## Feature Analysis")
    st.markdown("---")
    fi = _load_feature_importance()
    if fi is not None:
        st.markdown("#### Feature Importance (Linear SVM Coefficients / RBF Permutation Importance)")
        st.dataframe(fi, use_container_width=True, hide_index=True)
        fi_path = OUTPUTS_DIR / "feature_importance.png"
        if fi_path.exists(): st.image(str(fi_path), width=600)
    else: st.info("Feature importance not available. Run training first.")
    st.markdown("---")
    st.markdown("#### PCA Visualization")
    pca_path = OUTPUTS_DIR / "pca_visualization.png"
    if pca_path.exists():
        st.image(str(pca_path), width=600)
        st.caption("PCA is for visualization only. The classifier operates in the original feature space.")
    else: st.info("PCA visualization not available. Run training first.")
    st.markdown("---")
    st.markdown("#### Feature Groups")
    for gname, gfeats in FEATURE_GROUPS.items():
        st.markdown(f"**{gname}:** {', '.join(gfeats)}")

# === PAGE 08: EXPERIMENTS ===
elif nav.startswith("08"):
    st.markdown("## Experiments")
    st.markdown("---")
    exps = _load_experiments()
    if not exps:
        st.info("No experiments found. Run `python train.py` to create experiment artifacts in `experiments/`.")
    else:
        st.markdown(f"#### {len(exps)} experiment(s) found")
        for exp in exps:
            with st.expander(f"**{exp['id']}**"):
                cfg = exp.get("config",{})
                met = exp.get("metrics",{})
                st.markdown(f"**Model:** {cfg.get('best_model','?')}")
                st.markdown(f"**Sample Size:** {cfg.get('sample_size','?')}")
                st.markdown(f"**Random State:** {cfg.get('random_state','?')}")
                st.markdown(f"**Features:** {len(cfg.get('features',[]))}")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Accuracy", f"{met.get('accuracy',0):.4f}")
                c2.metric("Precision", f"{met.get('precision',0):.4f}")
                c3.metric("Recall", f"{met.get('recall',0):.4f}")
                c4.metric("F1", f"{met.get('f1',0):.4f}")

# === PAGE 09: BENCHMARK ===
elif nav.startswith("09"):
    st.markdown("## Benchmark")
    st.markdown("---")
    comp = _load_comparison_df()
    if comp is None:
        st.info("No comparison data. Run `python train.py --benchmark` or `--full`.")
    else:
        st.markdown("#### Model Comparison")
        st.dataframe(comp, use_container_width=True, hide_index=True)
        if "f1" in comp.columns:
            st.markdown("#### F1 Score Comparison")
            st.bar_chart(comp.set_index("model")["f1"])
        if "train_time" in comp.columns:
            st.markdown("#### Training Time Comparison")
            st.bar_chart(comp.set_index("model")["train_time"])

# === PAGE 10: DRIFT ANALYSIS ===
elif nav.startswith("10"):
    st.markdown("## Drift Analysis")
    st.markdown("---")
    st.markdown("#### SIMULATED DATA DRIFT")
    st.info("This experiment investigates model behavior under distribution shift. This is a controlled simulation, not a real-world attack scenario.")
    st.markdown("---")
    pipe = _load_pipeline_cached()
    if pipe is None:
        st.warning("No trained model. Run `python train.py` first.")
    else:
        try:
            from src.analysis import simulate_distribution_shift, compute_distribution_comparison, evaluate_drift_impact
            df_full, vr = load_and_validate()
            feature_cols = vr["features_resolved"]
            from src.data import sample_stratified
            from src.preprocessing import encode_labels, prepare_features, split_data
            ds = sample_stratified(df_full, min(5000, len(df_full)), random_state=42)
            X, y = prepare_features(ds, feature_cols)
            y_enc, _ = encode_labels(y)
            Xtr, Xte, ytr, yte = split_data(X, y_enc, random_state=42)

            shift_mag = st.slider("Shift magnitude", 1.0, 3.0, 1.5, 0.1)
            if st.button("Run Drift Simulation"):
                with st.spinner("Running..."):
                    shifted = simulate_distribution_shift(pd.DataFrame(Xte, columns=feature_cols), feature_cols, shift_mag)
                    dist_comp = compute_distribution_comparison(pd.DataFrame(Xte, columns=feature_cols), shifted, feature_cols)
                    X_shifted = shifted.values
                    drift_result = evaluate_drift_impact(pipe, Xte.values, yte.values, X_shifted, yte.values)
                st.markdown("#### Distribution Comparison")
                st.dataframe(dist_comp, use_container_width=True, hide_index=True)
                st.markdown("#### Performance Impact")
                mc = drift_result["metric_comparison"]
                for key in ["accuracy", "precision", "recall", "f1"]:
                    v = mc[key]
                    st.metric(f"{key.title()} Delta", f"{v['delta']:+.4f}", f"{v['delta_pct']:+.2f}%")
        except Exception as e:
            st.error(f"Drift analysis failed: {e}")

# === PAGE 11: METHODOLOGY ===
elif nav.startswith("11"):
    st.markdown("## Methodology")
    st.markdown("---")
    st.markdown("### 1. Problem")
    st.markdown("Given a network flow represented by numerical traffic features, can an SVM distinguish BENIGN traffic from ATTACK traffic?")
    st.markdown("### 2. Dataset")
    st.markdown("[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) from the Canadian Institute for Cybersecurity.")
    st.markdown("### 3. Cleaning")
    st.markdown("- Column names normalized\n- Duplicates removed\n- Infinity replaced with NaN\n- NaN rows dropped\n- Labels mapped to BENIGN/ATTACK\n- Original labels preserved for multiclass")
    st.markdown("### 4. Feature Selection")
    st.markdown("15 network flow features selected for interpretability:")
    for f in SELECTED_FEATURES:
        st.markdown(f"- **{f}**: {FD.get(f,'')}")
    st.markdown("### 5. Standardization")
    st.markdown("`StandardScaler` (zero mean, unit variance). Fit **only on training data** to prevent leakage.")
    st.markdown("### 6. SVM")
    st.markdown("Maximum-margin classification. The optimal hyperplane maximizes the distance to the nearest training points (support vectors).")
    st.markdown("### 7. Kernel Comparison")
    st.markdown("- **Linear:** Linear decision boundary. Fast, interpretable via coefficients.\n- **RBF:** Nonlinear via kernel trick. Tuned via grid search over C and gamma.")
    st.markdown("### 8. Evaluation")
    st.markdown("Held-out test set. Metrics: accuracy, precision, recall, F1, confusion matrix, threshold analysis.")
    st.markdown("### 9. Limitations")
    st.markdown("- Binary classification (attack subtypes collapsed)\n- Single-day dataset subset\n- SVC expensive on large data\n- NOT a production IDS\n- Decision scores are NOT probabilities\n- Offline analysis only")
    st.markdown("---")
    st.markdown("### Key Concepts")
    st.markdown("""**Margin:** The distance between the decision boundary and the nearest training points.

**Support Vectors:** Training points closest to the decision boundary. They define the boundary.

**Kernel:** A function that maps data to higher dimensions where it may be linearly separable.

**C:** Regularization parameter. High C = low bias, high variance. Low C = high bias, low variance.

**Gamma:** RBF kernel width. High gamma = complex boundary. Low gamma = smooth boundary.

**Standardization:** SVMs are sensitive to feature scale. Scaling ensures all features contribute equally.
""")
    st.markdown("---")
    st.markdown("### Pipeline")
    st.markdown(" \u2192 ".join([f"`{s}`" for s in ["RAW DATA","CLEANING","LABEL NORMALIZATION","FEATURE SELECTION","STRATIFIED SAMPLING","TRAIN/TEST SPLIT","STANDARDIZATION","SVM TRAINING","EVALUATION","SERIALIZATION"]]))
