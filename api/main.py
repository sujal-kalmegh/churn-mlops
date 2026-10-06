"""FastAPI service: real-time + batch churn inference with Prometheus metrics."""
import io
import json
import logging
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)

from api.schemas import BatchPrediction, Customer, Prediction
from src import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("churn-api")

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

REQUESTS = Counter("churn_api_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram("churn_api_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("churn_predictions_total", "Predictions by class", ["prediction"])
PROBA = Histogram("churn_probability", "Predicted churn probability",
                  buckets=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
MODEL_F1 = Gauge("churn_model_f1", "F1 of the currently loaded model")
_log_lock = threading.Lock()


def _paths():
    d = Path(os.getenv("MODELS_DIR", cfg.MODELS_DIR))
    return d / "model.joblib", d / "metrics.json"


@lru_cache(maxsize=4)
def _load(model_path: str, mtime: float):
    log.info("Loading model %s", model_path)
    return joblib.load(model_path)


def get_model():
    mp, _ = _paths()
    if not mp.exists():
        raise HTTPException(503, "Model not available. Run `python -m src.train` first.")
    return _load(str(mp), mp.stat().st_mtime)   # reloads automatically after retraining


def _risk(p: float) -> str:
    return "high" if p >= 0.7 else "medium" if p >= 0.4 else "low"


def _log_features(df: pd.DataFrame):
    """Append served features so the drift job can compare them to training data."""
    log_path = Path(os.getenv("PRODUCTION_LOG", cfg.PRODUCTION_LOG))
    with _log_lock:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        df[cfg.FEATURES].to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def _predict(df: pd.DataFrame) -> list[Prediction]:
    proba = get_model().predict_proba(df[cfg.FEATURES])[:, 1]
    out = []
    for p in proba:
        p = float(p)
        PROBA.observe(p)
        PREDICTIONS.labels("churn" if p >= .5 else "stay").inc()
        out.append(Prediction(churn_probability=round(p, 4), churn_prediction=p >= .5,
                              risk_level=_risk(p)))
    _log_features(df)
    return out


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        resp = await call_next(request)
        status = resp.status_code
        return resp
    finally:
        ep = request.url.path
        if ep != "/metrics":
            LATENCY.labels(ep).observe(time.perf_counter() - start)
            REQUESTS.labels(ep, str(status)).inc()
            log.info("%s %s -> %s (%.1f ms)", request.method, ep, status,
                     (time.perf_counter() - start) * 1000)


@app.get("/health")
def health():
    mp, _ = _paths()
    return {"status": "ok", "model_loaded": mp.exists()}


@app.get("/model-info")
def model_info():
    _, meta = _paths()
    if not meta.exists():
        raise HTTPException(503, "No model metadata")
    return json.loads(meta.read_text())


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return _predict(pd.DataFrame([customer.model_dump()]))[0]


@app.post("/predict_batch", response_model=BatchPrediction)
async def predict_batch(file: UploadFile = File(...)):
    """Batch inference: upload a CSV with the same columns as the Telco dataset."""
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        missing = [c for c in cfg.FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Invalid CSV: {e}")
    preds = _predict(df)
    return BatchPrediction(count=len(preds), predictions=preds)


@app.get("/metrics")
def metrics():
    _, meta = _paths()
    if meta.exists():
        MODEL_F1.set(json.loads(meta.read_text()).get("f1", 0))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
