#!/usr/bin/env python3
"""
SENTINEL API
Lightweight FastAPI service for network flow classification.

Endpoints:
    GET  /health
    GET  /metadata
    POST /predict
    POST /batch-predict
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import tempfile
import os

from src.config import SELECTED_FEATURES, MODELS_DIR
from src.model import load_pipeline, load_metadata
from src.inference import predict_single_flow, predict_batch_csv

app = FastAPI(
    title="SENTINEL API",
    description="SVM-based network intrusion classification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None
_metadata = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = load_pipeline()
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Model not trained. Run train.py first.")
    return _pipeline


def get_metadata():
    global _metadata
    if _metadata is None:
        _metadata = load_metadata()
    return _metadata


class FlowInput(BaseModel):
    features: Dict[str, float] = Field(..., description="Network flow features")


class BatchResult(BaseModel):
    prediction: str
    decision_score: float


@app.get("/health")
def health():
    model_path = MODELS_DIR / "svm_pipeline.joblib"
    return {
        "status": "healthy",
        "model_loaded": model_path.exists(),
        "model_path": str(model_path),
    }


@app.get("/metadata")
def metadata():
    meta = get_metadata()
    if not meta:
        raise HTTPException(status_code=404, detail="No model metadata found.")
    return meta


@app.post("/predict")
def predict(flow: FlowInput):
    pipe = get_pipeline()
    result = predict_single_flow(pipe, flow.features)
    if result.get("error"):
        raise HTTPException(status_code=422, detail={
            "missing_features": result.get("missing_features", []),
            "invalid_features": result.get("invalid_features", []),
        })
    return {
        "prediction": result["prediction"],
        "decision_score": result["decision_score"],
        "model": result["model"],
    }


@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    pipe = get_pipeline()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result_df, info = predict_batch_csv(pipe, tmp_path)
    finally:
        os.unlink(tmp_path)

    if info is None or info.get("error"):
        raise HTTPException(status_code=422, detail=info)

    results = result_df[["prediction", "decision_score"]].to_dict(orient="records")
    return {
        "total_rows": info["total_rows"],
        "valid_rows": info["valid_rows"],
        "predictions_summary": info["predictions"],
        "results": results,
    }
