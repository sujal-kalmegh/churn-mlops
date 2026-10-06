"""Data-drift detection: KS-test + PSI for numeric, chi-square + PSI for categorical.

Usage:
    python -m src.drift --current data/production_log.csv
Exit code 1 if drift is detected (so Airflow/CI can branch on it).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import config as cfg

PSI_THRESHOLD = 0.2     # >0.2 = significant shift (industry rule of thumb)
P_THRESHOLD = 0.01
DATASET_DRIFT_SHARE = 0.2   # drift if >=20% of features drifted
MIN_ROWS = 30


def psi(expected: pd.Series, actual: pd.Series, bins=10) -> float:
    if expected.dtype.kind in "if":
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = pd.cut(expected, edges).value_counts(normalize=True, sort=False)
        a = pd.cut(actual, edges).value_counts(normalize=True, sort=False)
    else:
        cats = sorted(set(expected.astype(str)) | set(actual.astype(str)))
        e = expected.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
        a = actual.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    e, a = e.clip(lower=1e-4), a.clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())


def detect(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    feats = {}
    for col in cfg.FEATURES:
        ref, cur = reference[col], current[col]
        if col in cfg.NUMERIC:
            p = float(stats.ks_2samp(ref, cur).pvalue)
        else:
            cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
            table = np.array([
                ref.astype(str).value_counts().reindex(cats, fill_value=0).values,
                cur.astype(str).value_counts().reindex(cats, fill_value=0).values])
            table = table[:, table.sum(axis=0) > 0]
            p = float(stats.chi2_contingency(table)[1]) if table.shape[1] > 1 else 1.0
        score = psi(ref, cur)
        feats[col] = {"p_value": round(p, 5), "psi": round(score, 4),
                      "drifted": bool(p < P_THRESHOLD and score > PSI_THRESHOLD)}
    share = sum(f["drifted"] for f in feats.values()) / len(feats)
    return {"n_reference": len(reference), "n_current": len(current),
            "drift_share": round(share, 3), "dataset_drift": share >= DATASET_DRIFT_SHARE,
            "drifted_features": [c for c, f in feats.items() if f["drifted"]],
            "features": feats}


def run(current_path, reference_path=None, out=None) -> dict:
    reference = pd.read_csv(reference_path or cfg.REFERENCE_PATH)
    current = pd.read_csv(current_path)
    if len(current) < MIN_ROWS:
        return {"dataset_drift": False, "skipped": f"only {len(current)} rows (< {MIN_ROWS})"}
    report = detect(reference, current[cfg.FEATURES])
    cfg.REPORTS_DIR.mkdir(exist_ok=True)
    Path(out or cfg.REPORTS_DIR / "drift_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", default=str(cfg.PRODUCTION_LOG))
    ap.add_argument("--reference", default=None)
    a = ap.parse_args()
    rep = run(a.current, a.reference)
    print(json.dumps({k: v for k, v in rep.items() if k != "features"}, indent=2))
    sys.exit(1 if rep.get("dataset_drift") else 0)
