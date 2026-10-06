import pytest
from fastapi.testclient import TestClient

SAMPLE = {
    "gender": "Female", "SeniorCitizen": "0", "Partner": "Yes", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 285.0}


@pytest.fixture
def client(trained_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(trained_dir))
    monkeypatch.setenv("PRODUCTION_LOG", str(tmp_path / "prod.csv"))
    from api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}


def test_predict_validation_error(client):
    bad = {**SAMPLE, "Contract": "Weekly"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch(client, synthetic_csv):
    with open(synthetic_csv, "rb") as f:
        r = client.post("/predict_batch", files={"file": ("x.csv", f, "text/csv")})
    assert r.status_code == 200 and r.json()["count"] == 600


def test_metrics_endpoint(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "churn_predictions_total" in text and "churn_api_requests_total" in text


def test_model_info(client):
    assert "f1" in client.get("/model-info").json()
