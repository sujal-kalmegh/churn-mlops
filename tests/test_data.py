from src.config import FEATURES, TARGET
from src.data import build_preprocessor, load_and_clean


def test_cleaning(synthetic_csv):
    df = load_and_clean(synthetic_csv)
    assert "customerID" not in df.columns
    assert set(df[TARGET].unique()) <= {0, 1}
    assert df["TotalCharges"].isna().sum() == 0


def test_blank_total_charges_handled(tmp_path, synthetic_csv):
    import pandas as pd
    df = pd.read_csv(synthetic_csv)
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    df.loc[0, "TotalCharges"] = " "
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    assert load_and_clean(p).loc[0, "TotalCharges"] == 0.0


def test_preprocessor_output(synthetic_csv):
    df = load_and_clean(synthetic_csv)
    X = build_preprocessor().fit_transform(df[FEATURES])
    assert X.shape[0] == len(df) and X.shape[1] > len(FEATURES)
