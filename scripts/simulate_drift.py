"""Create a drifted 'production' dataset to demo drift detection + retraining.

Simulates a price hike + shift to month-to-month, e-check customers.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402
from src.data import load_and_clean  # noqa: E402

r = np.random.default_rng(7)
df = load_and_clean(cfg.DATA_PATH).sample(800, random_state=1).reset_index(drop=True)
df["MonthlyCharges"] = (df["MonthlyCharges"] * 1.35).round(2)
df["tenure"] = (df["tenure"] * 0.5).astype(int)
df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).round(2)
df["Contract"] = r.choice(["Month-to-month", "One year", "Two year"], len(df), p=[.85, .1, .05])
df["PaymentMethod"] = r.choice(["Electronic check", "Mailed check"], len(df), p=[.8, .2])
cfg.PRODUCTION_LOG.parent.mkdir(exist_ok=True)
df[cfg.FEATURES].to_csv(cfg.PRODUCTION_LOG, index=False)
print(f"Wrote drifted data -> {cfg.PRODUCTION_LOG} ({len(df)} rows)")
