import json

import joblib

from src.config import FEATURES
from src.data import load_and_clean


def test_artifacts_created(trained_dir):
    assert (trained_dir / "model.joblib").exists()
    assert (trained_dir / "metrics.json").exists()


def test_quality_gate(trained_dir):
    """CI quality gate: fail the pipeline if the model is no better than chance."""
    m = json.loads((trained_dir / "metrics.json").read_text())
    assert m["roc_auc"] > 0.70, m


def test_predict_proba_valid(trained_dir, synthetic_csv):
    model = joblib.load(trained_dir / "model.joblib")
    X = load_and_clean(synthetic_csv)[FEATURES].head(20)
    p = model.predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
