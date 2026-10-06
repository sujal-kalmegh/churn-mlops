"""Data loading, cleaning and a synthetic generator used by tests/CI."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL, FEATURES, NUMERIC, TARGET


def load_and_clean(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"], errors="ignore")
    # TotalCharges contains blank strings for brand-new customers (tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = (df[TARGET].str.strip() == "Yes").astype(int)
    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def make_synthetic(n=600, seed=0) -> pd.DataFrame:
    """Small Telco-like dataset so tests/CI never depend on a download."""
    r = np.random.default_rng(seed)
    pick = lambda opts: r.choice(opts, n)  # noqa: E731
    tenure = r.integers(0, 73, n)
    monthly = np.round(r.uniform(18, 118, n), 2)
    contract = pick(["Month-to-month", "One year", "Two year"])
    df = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(n)],
        "gender": pick(["Male", "Female"]), "SeniorCitizen": r.integers(0, 2, n),
        "Partner": pick(["Yes", "No"]), "Dependents": pick(["Yes", "No"]),
        "tenure": tenure, "PhoneService": pick(["Yes", "No"]),
        "MultipleLines": pick(["Yes", "No", "No phone service"]),
        "InternetService": pick(["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": pick(["Yes", "No"]), "OnlineBackup": pick(["Yes", "No"]),
        "DeviceProtection": pick(["Yes", "No"]), "TechSupport": pick(["Yes", "No"]),
        "StreamingTV": pick(["Yes", "No"]), "StreamingMovies": pick(["Yes", "No"]),
        "Contract": contract, "PaperlessBilling": pick(["Yes", "No"]),
        "PaymentMethod": pick(["Electronic check", "Mailed check",
                               "Bank transfer (automatic)", "Credit card (automatic)"]),
        "MonthlyCharges": monthly, "TotalCharges": np.round(monthly * tenure, 2),
    })
    z = (-1.0 + 1.6 * (contract == "Month-to-month") - 0.03 * tenure
         + 0.015 * monthly + r.normal(0, 0.7, n))
    df["Churn"] = np.where(z > 0.4, "Yes", "No")
    return df
