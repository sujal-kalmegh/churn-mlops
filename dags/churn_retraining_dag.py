"""Airflow DAG: weekly drift check -> (retrain if drifted) -> report.

validate_data -> drift_check -> retrain (if drift) or skip (no drift) -> done
"""
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator

PROJECT = "/opt/project"   # path mounted in docker-compose.yml
sys.path.insert(0, PROJECT)


def decide():
    from src import config as cfg
    from src import drift
    if not cfg.PRODUCTION_LOG.exists():
        return "skip"
    return "retrain" if drift.run(cfg.PRODUCTION_LOG).get("dataset_drift") else "skip"


with DAG(
    dag_id="churn_retraining",
    description="Drift detection and automated retraining for the churn model",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["mlops", "churn"],
) as dag:
    validate = BashOperator(task_id="validate_data",
                            bash_command=f"cd {PROJECT} && python scripts/get_data.py")
    drift_check = BranchPythonOperator(task_id="drift_check", python_callable=decide)
    retrain = BashOperator(task_id="retrain",
                           bash_command=f"cd {PROJECT} && python -m src.retrain --force")
    skip = EmptyOperator(task_id="skip")
    done = EmptyOperator(task_id="done", trigger_rule="none_failed_min_one_success")

    validate >> drift_check >> [retrain, skip] >> done
