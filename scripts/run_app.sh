#!/usr/bin/env bash
set -euo pipefail
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:5001}
gunicorn -w 2 -b 127.0.0.1:8000 app.app:app
