"""Download the IBM Telco churn dataset; fall back to synthetic data (CI/offline)."""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_PATH  # noqa: E402
from src.data import make_synthetic  # noqa: E402

URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
       "master/data/Telco-Customer-Churn.csv")

if DATA_PATH.exists():
    print(f"{DATA_PATH} already present")
    sys.exit(0)
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
try:
    urllib.request.urlretrieve(URL, DATA_PATH)
    print("Downloaded real Telco dataset")
except Exception as e:  # noqa: BLE001
    print(f"Download failed ({e}); generating synthetic data")
    make_synthetic(3000, seed=1).to_csv(DATA_PATH, index=False)
