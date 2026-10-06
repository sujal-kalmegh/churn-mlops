"""Central configuration shared by training, API, drift and DAG code."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "data" / "telco.csv"))
REFERENCE_PATH = ROOT / "data" / "reference.csv"      # training snapshot for drift
PRODUCTION_LOG = ROOT / "data" / "production_log.csv"  # features seen by the API
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
REPORTS_DIR = ROOT / "reports"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
EXPERIMENT = "customer-churn"
REGISTERED_MODEL = "churn-model"

TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL
RANDOM_STATE = 42
