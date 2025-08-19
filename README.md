# MLOps Project: MLflow + Sklearn (make_classification) + Flask

This project demonstrates a full tiny MLOps workflow:
- Generate synthetic classification data via `sklearn.datasets.make_classification`.
- Train several simple models with different hyperparameters.
- Track everything with MLflow.
- Pick the best model and **register** it, transitioning it to the **Production** stage automatically.
- Serve predictions from a Flask web app that **always** loads the current Production model.

## 0) Python environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## 1) Start MLflow Tracking + Model Registry locally
Use a SQLite backend so the **Model Registry** is available.
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
# Windows (PowerShell): $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"

# Start MLflow server in terminal-1 from the project root:
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5001
```
Open the UI at http://127.0.0.1:5001

> If you see an error about the artifact root folder not existing, create it: `mkdir mlruns`.

## 2) Run experiments (train + tune + log + register best)
In **terminal-2** (with the same virtualenv), run:
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
python src/train.py --experiment-name makeclassif-demo --registered-model BestMakeClassif
```
The script will:
- Sweep several dataset variants (simulating data enrichment).
- Train LogisticRegression & RandomForest with a small hyperparameter grid.
- Log parameters/metrics/artifacts to MLflow.
- Pick the best run by F1-score (then accuracy as tie-breaker).
- **Register** that run’s model to the Model Registry under `BestMakeClassif` and move it to **Production**, archiving any previous Production model.

You can verify all runs in the MLflow UI. The "Models" tab will show `BestMakeClassif` with version numbers and the current stage.

## 3) Serve the Production model via Flask
In **terminal-3**:
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
python app/app.py
# or (recommended for production-like serving)
gunicorn -w 2 -b 127.0.0.1:8000 app.app:app
```
The Flask app will load `models:/BestMakeClassif/Production` and expose:
- `GET /health` -> returns status ok.
- `POST /predict` -> JSON body with `features`: 2D array, e.g.:
  ```json
  {"features": [[0.1, -0.2, 1.3, ...]]}
  ```

Example request:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.5, -1.2, 0.3, 1.1, 0.0, -0.8, 1.9, 0.4, -0.1, 0.2, 0.7, -1.3, 1.5, 0.6, -0.9, 0.8, -0.4, 0.3, 0.2, -0.5]]]}'
```
(Adjust feature length to match the trained model’s expected number of features; check MLflow UI -> model signature.)

## 4) Re-training after data changes
Just re-run `src/train.py`. If a new run beats the current Production model, the script will register it as a new version and transition it to Production automatically. The Flask app will **not** need a restart if you code it to reload periodically; in this simple demo, restart the app to refresh the loaded model.

## 5) Project structure
```
.
├── app
│   └── app.py
├── scripts
│   ├── start_mlflow.sh
│   ├── run_train.sh
│   └── run_app.sh
├── src
│   └── train.py
├── requirements.txt
└── README.md
```
