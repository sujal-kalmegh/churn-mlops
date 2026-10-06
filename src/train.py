"""Train candidate models, track with MLflow, register the best one.

Usage:
    python -m src.train                       # always promote best model
    python -m src.train --promote-if-better   # retraining mode (only replaces
                                              # the live model if it improves)
"""
import argparse
import json
import logging

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config as cfg
from src.data import build_preprocessor, load_and_clean

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def candidates(scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=0.5),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=cfg.RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=cfg.RANDOM_STATE),
    }


def evaluate(pipe, X, y) -> dict:
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def train(data_path=None, models_dir=None, promote_if_better=False,
          register=True, tracking_uri=None) -> dict:
    data_path = data_path or cfg.DATA_PATH
    models_dir = models_dir or cfg.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(data_path)
    X, y = df[cfg.FEATURES], df[cfg.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=cfg.RANDOM_STATE)
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    mlflow.set_tracking_uri(tracking_uri or cfg.MLFLOW_URI)
    mlflow.set_experiment(cfg.EXPERIMENT)

    best = {"f1": -1.0}
    for name, model in candidates(spw).items():
        with mlflow.start_run(run_name=name) as run:
            pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
            pipe.fit(X_tr, y_tr)
            m = evaluate(pipe, X_te, y_te)
            mlflow.log_params({"model": name, "rows": len(df),
                               **{f"hp_{k}": v for k, v in model.get_params().items()
                                  if isinstance(v, (int, float, str, bool))}})
            mlflow.log_metrics(m)
            log.info("%-20s f1=%.3f auc=%.3f recall=%.3f", name, m["f1"], m["roc_auc"], m["recall"])
            if m["f1"] > best["f1"]:
                best = {**m, "name": name, "pipe": pipe, "run_id": run.info.run_id}

    # ---- decide whether the new model replaces the live one
    metrics_file = models_dir / "metrics.json"
    if promote_if_better and metrics_file.exists():
        old = json.loads(metrics_file.read_text())
        if round(best["f1"], 4) < old["f1"]:
            log.warning("New model f1=%.3f < current %.3f -> NOT promoted", best["f1"], old["f1"])
            return {"promoted": False, "new_f1": best["f1"], "old_f1": old["f1"]}

    version = None
    if register:
        with mlflow.start_run(run_id=best["run_id"]):
            info = mlflow.sklearn.log_model(
                best["pipe"], name="model", registered_model_name=cfg.REGISTERED_MODEL,
                serialization_format="cloudpickle")
        client = MlflowClient()
        version = info.registered_model_version
        client.set_registered_model_alias(cfg.REGISTERED_MODEL, "production", version)
        log.info("Registered %s v%s and set alias 'production'", cfg.REGISTERED_MODEL, version)

    joblib.dump(best["pipe"], models_dir / "model.joblib")
    metrics = {k: round(float(best[k]), 4) for k in
               ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    metrics.update(model_name=best["name"], registry_version=version,
                   trained_rows=int(len(df)))
    metrics_file.write_text(json.dumps(metrics, indent=2))
    # reference snapshot = what the model was trained on (used for drift detection)
    X_tr.to_csv(cfg.REFERENCE_PATH if models_dir == cfg.MODELS_DIR
                else models_dir / "reference.csv", index=False)
    log.info("Best model: %s %s", best["name"], metrics)
    return {"promoted": True, **metrics}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--promote-if-better", action="store_true")
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    from pathlib import Path
    out = train(Path(a.data) if a.data else None, promote_if_better=a.promote_if_better,
                register=not a.no_register)
    print(json.dumps({k: (float(v) if isinstance(v, np.floating) else v)
                      for k, v in out.items()}, indent=2))
