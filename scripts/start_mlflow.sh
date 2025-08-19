#!/usr/bin/env bash
set -euo pipefail
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:5001}
mkdir -p mlruns
echo "Starting MLflow at $MLFLOW_TRACKING_URI ..."
mlflow server       --backend-store-uri sqlite:///mlflow.db       --default-artifact-root ./mlruns       --host 127.0.0.1       --port ${MLFLOW_TRACKING_URI##*:}
