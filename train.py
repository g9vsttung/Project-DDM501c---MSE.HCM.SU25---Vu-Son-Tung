import argparse
import itertools
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---------------------------- Helpers ----------------------------
@dataclass
class DatasetConfig:
    name: str
    n_samples: int = 2000
    n_features: int = 20
    n_informative: int = 10
    n_redundant: int = 2
    n_repeated: int = 0
    n_classes: int = 2
    weights: Tuple[float, ...] = (0.5, 0.5)
    class_sep: float = 1.0
    flip_y: float = 0.01
    random_state: int = 42

def build_datasets(seed: int) -> List[DatasetConfig]:
    return [
        DatasetConfig(name="base", random_state=seed),
        DatasetConfig(name="more_samples", n_samples=5000, random_state=seed),
        DatasetConfig(name="more_features", n_features=30, n_informative=15, random_state=seed),
        DatasetConfig(name="harder", class_sep=0.7, flip_y=0.03, random_state=seed),
    ]

def gen_data(cfg: DatasetConfig):
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_redundant=cfg.n_redundant,
        n_repeated=cfg.n_repeated,
        n_classes=cfg.n_classes,
        weights=list(cfg.weights) if cfg.weights else None,
        class_sep=cfg.class_sep,
        flip_y=cfg.flip_y,
        random_state=cfg.random_state,
    )
    return X, y

def eval_binary(y_true, prob_1, y_pred) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob_1))
    except Exception:
        pass
    return metrics

# ---------------------------- Training ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and register best model with MLflow")
    parser.add_argument("--experiment-name", type=str, default="makeclassif-demo")
    parser.add_argument("--registered-model", type=str, default="BestMakeClassif")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)

    datasets = build_datasets(args.seed)

    # Model families and small grids
    model_spaces = {
        "logreg": {
            "estimator": LogisticRegression(max_iter=1000),
            "grid": {
                "C": [0.1, 1.0, 10.0],
                "solver": ["liblinear"],
                "penalty": ["l2"],
            },
        },
        "rf": {
            "estimator": RandomForestClassifier(random_state=args.seed),
            "grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
        },
    }

    best = {
        "score": -1.0,
        "acc": -1.0,
        "run_id": None,
        "model_uri": None,
        "dataset": None,
        "family": None,
        "params": None,
    }

    for ds in datasets:
        X, y = gen_data(ds)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )

        for family, spec in model_spaces.items():
            keys, values = zip(*spec["grid"].items())  # type: ignore
            for comb in itertools.product(*values):
                params = dict(zip(keys, comb))
                # Fresh estimator each loop
                if family == "logreg":
                    model = LogisticRegression(max_iter=1000, **params)
                elif family == "rf":
                    model = RandomForestClassifier(random_state=args.seed, **params)
                else:
                    continue

                run_name = f"{ds.name}-{family}-{json.dumps(params, sort_keys=True)}"
                with mlflow.start_run(run_name=run_name, nested=False) as run:
                    mlflow.set_tags({
                        "dataset": ds.name,
                        "family": family,
                        "framework": "sklearn",
                        "problem_type": "binary_classification",
                    })
                    # Log dataset config + hyperparams
                    for k, v in asdict(ds).items():
                        mlflow.log_param(f"data__{k}", v)
                    for k, v in params.items():
                        mlflow.log_param(f"model__{k}", v)

                    # Train
                    model.fit(X_train, y_train)

                    # Predict
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_test)[:, 1]
                    else:
                        # Fall back: use decision_function if available
                        if hasattr(model, "decision_function"):
                            z = model.decision_function(X_test)
                            # map to [0,1] via logistic-ish transform
                            proba = 1 / (1 + np.exp(-z))
                        else:
                            proba = np.zeros_like(y_test, dtype=float)

                    y_pred = model.predict(X_test)
                    metrics = eval_binary(y_test, proba, y_pred)

                    # Log metrics
                    for mk, mv in metrics.items():
                        mlflow.log_metric(mk, mv)

                    # Log model
                    signature = infer_signature(X_test, y_pred)
                    input_example = X_test[:3]
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        input_example=input_example,
                        signature=signature,
                    )

                    # Track best by F1 then accuracy
                    f1 = metrics.get("f1", -1.0)
                    acc = metrics.get("accuracy", -1.0)
                    if (f1 > best["score"]) or (f1 == best["score"] and acc > best["acc"]):
                        best.update({
                            "score": f1,
                            "acc": acc,
                            "run_id": run.info.run_id,
                            "model_uri": f"runs:/{run.info.run_id}/model",
                            "dataset": ds.name,
                            "family": family,
                            "params": params,
                        })

    if best["model_uri"] is None:
        raise RuntimeError("No runs were completed; cannot register a model.")

    # Register best model and move to Production
    client = MlflowClient()
    name = args.registered_model
    print(f"Registering best model from run {best['run_id']} with F1={best['score']:.4f} ...")
    res = mlflow.register_model(model_uri=best["model_uri"], name=name)

    # Transition to Production, archive any previous
    client.transition_model_version_stage(
        name=name,
        version=res.version,
        stage="Production",
        archive_existing_versions=True,
    )

    # Save a small report
    report = {
        "registered_model": name,
        "new_version": res.version,
        "stage": "Production",
        "best_run_id": best["run_id"],
        "best_f1": best["score"],
        "best_acc": best["acc"],
        "best_dataset": best["dataset"],
        "best_family": best["family"],
        "best_params": best["params"],
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/best_model.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Done. Summary saved to reports/best_model.json")
    print(json.dumps(report, indent=2))

mlflow.set_tracking_uri("http://127.0.0.1:5001")
if __name__ == "__main__":
    main()
