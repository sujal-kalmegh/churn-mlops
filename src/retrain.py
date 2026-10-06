"""Automated retraining: check drift -> retrain -> promote only if better."""
import argparse
import json

from src import drift
from src import config as cfg
from src.train import train


def main(force=False, current=None) -> dict:
    current = current or cfg.PRODUCTION_LOG
    if not force:
        if not current.exists():
            return {"action": "skipped", "reason": "no production data yet"}
        rep = drift.run(current)
        if not rep.get("dataset_drift"):
            return {"action": "skipped", "reason": "no drift detected",
                    "drift_share": rep.get("drift_share")}
    return {"action": "retrained", **train(promote_if_better=True)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--current", default=None)
    a = ap.parse_args()
    from pathlib import Path
    print(json.dumps(main(a.force, Path(a.current) if a.current else None), indent=2, default=str))
