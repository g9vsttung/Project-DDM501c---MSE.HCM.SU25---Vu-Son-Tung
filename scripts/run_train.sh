#!/usr/bin/env bash
set -euo pipefail
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:5001}
python src/train.py --experiment-name makeclassif-demo --registered-model BestMakeClassif "$@"
