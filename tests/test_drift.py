from src.config import FEATURES
from src.data import load_and_clean
from src.drift import detect


def test_no_drift_on_same_distribution(synthetic_csv):
    df = load_and_clean(synthetic_csv)[FEATURES]
    rep = detect(df.iloc[:300], df.iloc[300:])
    assert not rep["dataset_drift"]


def test_detects_drift(synthetic_csv):
    df = load_and_clean(synthetic_csv)[FEATURES]
    ref, cur = df.iloc[:300].copy(), df.iloc[300:].copy()
    cur["MonthlyCharges"] = cur["MonthlyCharges"] * 2
    cur["tenure"] = cur["tenure"] * 0 + 1
    cur["Contract"] = "Month-to-month"
    cur["PaymentMethod"] = "Electronic check"
    rep = detect(ref, cur)
    assert rep["dataset_drift"] and "MonthlyCharges" in rep["drifted_features"]
