import os
from typing import List

import mlflow.pyfunc
from flask import Flask, jsonify, request, render_template
import mlflow
# Ensure the app points to the same MLflow server you used for training.
# Example:
# export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
mlflow.set_tracking_uri("http://127.0.0.1:5001")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
REGISTERED_MODEL = os.environ.get("REGISTERED_MODEL", "BestMakeClassif")
MODEL_URI = f"models:/{REGISTERED_MODEL}/Production"

app = Flask(__name__)

def load_model():
    # This will fetch the latest model version in the Production stage.
    model = mlflow.pyfunc.load_model(MODEL_URI)
    return model

model = load_model()
@app.route("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": REGISTERED_MODEL, "stage": "Production"})

@app.post("/predict")
def predict():
    data = request.get_json(force=True)
    feats = data.get("features", None)

    if feats is None:
        return jsonify({"error": "Missing 'features' field"}), 400

    import numpy as np

    # Nếu input là 1D (vd: [2,8,6,7]) thì convert thành [[2,8,6,7]]
    if isinstance(feats[0], (int, float)):
        X = [feats]
    else:
        X = feats

    try:
        # ép sang numpy float64
        X = np.array(feats, dtype=np.float64)
        preds = model.predict(X)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Nếu chỉ có 1 sample thì trả về 1 số duy nhất
    if len(preds) == 1:
        return jsonify({"predictions": int(preds[0])})
    return jsonify({"predictions": preds.tolist()})

if __name__ == "__main__":
    # For local dev; use gunicorn for production-like serving.
    app.run(host="127.0.0.1", port=8000, debug=True)
