import pytest

from src.data import make_synthetic


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "synthetic.csv"
    make_synthetic(600, seed=3).to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def trained_dir(synthetic_csv, tmp_path_factory):
    from src.train import train
    d = tmp_path_factory.mktemp("models")
    tr = tmp_path_factory.mktemp("mlruns")
    res = train(synthetic_csv, d, register=False, tracking_uri=f"sqlite:///{tr}/t.db")
    assert res["promoted"]
    return d
