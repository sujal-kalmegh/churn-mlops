.PHONY: install data train test lint run drift retrain up down
install: ; pip install -r requirements.txt
data:    ; python scripts/get_data.py
train:   ; python -m src.train
test:    ; pytest -q
lint:    ; flake8 src api tests scripts
run:     ; uvicorn api.main:app --reload --port 8000
drift:   ; python scripts/simulate_drift.py && python -m src.drift
retrain: ; python -m src.retrain
mlflow:  ; mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
up:      ; docker compose up -d --build
down:    ; docker compose down
