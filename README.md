# Customer Churn Prediction: End-to-End MLOps Project

**Course:** 7-CT MLOps and AI Development, YCCE Nagpur, Computer Technology Dept.
**Group:** `[Member 1 · Member 2 · Member 3 · Member 4]` **GitHub:** `[repo link]`

Predicts whether a telecom customer will churn and wraps the model in a complete MLOps lifecycle:

```
Develop → Train → Evaluate → Version → Register → Test → CI/CD → Containerize
        → Deploy (FastAPI) → Monitor → Detect Drift → Retrain → Redeploy
```

## Architecture

```
 data/telco.csv ─► src/train.py ─► MLflow (runs + Model Registry) ─► models/model.joblib
                        ▲                                                   │
                        │ retrain (only if better)                          ▼
 Airflow DAG ─► src/drift.py ◄── data/production_log.csv ◄──── FastAPI /predict, /predict_batch
 (weekly)                                                           │ /metrics
                                                                    ▼
 GitHub Actions: lint → pytest (quality gate) → train → docker build/push      Prometheus → Grafana
```

## Dataset
IBM **Telco Customer Churn** (7,043 customers, 20 features, ~26.5% churn).
Source: <https://github.com/IBM/telco-customer-churn-on-icp4d> (also on Kaggle). Downloaded by `scripts/get_data.py`
(falls back to synthetic data offline/CI).

## Results (real run on the dataset, 80/20 stratified split)
| Model | F1 | ROC-AUC | Recall |
|---|---|---|---|
| Logistic Regression | 0.613 | 0.842 | 0.783 |
| **Random Forest (selected)** | **0.629** | 0.842 | 0.754 |
| XGBoost | 0.622 | 0.842 | 0.765 |

Class imbalance is handled with class weights / `scale_pos_weight`; the best model is chosen by F1 because
missing a churner (recall) is costlier than a false alarm. Your numbers may differ slightly by library version.

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
make install        # pip install -r requirements.txt
make data           # download dataset
make train          # trains 3 models, logs to MLflow, registers best, writes models/model.joblib
make test           # 14 tests: data, model quality gate, drift, API
make run            # API at http://localhost:8000/docs
make mlflow         # MLflow UI at http://localhost:5000
```
(No `make`? Run the commands inside the Makefile directly, e.g. `python -m src.train`.)

## Docker
```bash
make train                  # model must exist before building the image
docker compose up -d --build
# API :8000 | MLflow :5000 | Prometheus :9090 | Grafana :3000 (admin/admin, dashboard "Churn API Monitoring")
docker compose --profile airflow up -d      # Airflow UI :8080 (password printed in container logs)
```

## API
| Endpoint | Purpose |
|---|---|
| `POST /predict` | Real-time inference for one customer → probability, label, risk level |
| `POST /predict_batch` | Batch inference: upload a CSV with Telco columns |
| `GET /health`, `GET /model-info` | Liveness; metrics of the loaded model |
| `GET /metrics` | Prometheus metrics (requests, latency, predictions by class, probability histogram, model F1) |

Every served request's features are appended to `data/production_log.csv`, the input for drift detection.
The API reloads `models/model.joblib` automatically when retraining replaces it.

## Drift detection & automated retraining
* `src/drift.py`: per-feature **KS-test** (numeric) / **chi-square** (categorical) plus **PSI**. A feature drifts when p < 0.01
  *and* PSI > 0.2; the dataset drifts when ≥ 20% of features drift (exit code 1).
* `src/retrain.py`: drift check → retrain → **promote only if F1 ≥ current** → new MLflow registry version + `production` alias.
* `dags/churn_retraining_dag.py`: weekly Airflow DAG `validate_data → drift_check → retrain | skip → done`.
* Concept drift needs ground-truth churn labels. Append newly labelled customers to the training CSV and the same
  retraining job re-evaluates the model on them.

## Live demo script (follows the order in the college notice)
1. **GitHub**: show branches, PRs, commit history (`git log --graph --oneline --all`).
2. **Training + versioning + registry**: `make train` → MLflow UI → Experiments (3 runs) → Models → `churn-model` v1, alias `production`.
3. **Automated testing / CI/CD**: `make test`; show a green GitHub Actions run (lint → tests → train → docker).
4. **Docker**: `docker compose up -d --build`; `docker ps`.
5. **API**: open `/docs`, send the example to `/predict`, upload a CSV to `/predict_batch`.
6. **Monitoring**: generate traffic, open Prometheus (`:9090`) and Grafana (`:3000`).
7. **Drift**: `python scripts/simulate_drift.py && python -m src.drift` → shows `dataset_drift: true`.
8. **Retraining**: `python -m src.retrain` (or trigger the Airflow DAG) → new version registered only if better.

## Branching strategy
`main` (stable, protected) ← `develop` ← `feature/*`. Every change goes through a Pull Request; CI must pass before merge.

## Repository map
| Notice requirement | Where |
|---|---|
| Training, evaluation, experiment tracking, registry | `src/train.py`, MLflow |
| Data preprocessing | `src/data.py` |
| Dependency management | `requirements*.txt`, venv |
| Automated tests | `tests/` |
| CI/CD | `.github/workflows/ci.yml` |
| Docker / Compose | `Dockerfile`, `docker-compose.yml` |
| REST API | `api/` |
| Monitoring | `monitoring/`, `/metrics` |
| Drift / retraining | `src/drift.py`, `src/retrain.py` |
| Airflow DAG | `dags/churn_retraining_dag.py` |

## Team contributions
| Member | Contribution |
|---|---|
| `[Name]` | Data preprocessing, model training, MLflow |
| `[Name]` | FastAPI service, tests |
| `[Name]` | Docker, Compose, CI/CD |
| `[Name]` | Monitoring, drift detection, Airflow |

## Future scope
Kubernetes deployment with autoscaling, feature store, canary/A-B releases, SHAP explainability, cloud deployment (AWS/GCP/Azure).

## References
IBM Telco Customer Churn dataset; scikit-learn, XGBoost, MLflow, FastAPI, Prometheus, Grafana and Apache Airflow documentation.
