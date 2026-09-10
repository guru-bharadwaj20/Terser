"""Dummy MLflow run proving tracking + registry work: ``python -m terser.tracking.dummy_run``."""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.dummy import DummyClassifier

from terser.tracking.runs import promote, register_model, setup_tracking, start_run

MODEL_NAME = "terser-dummy"


def main() -> str:
    """Log a dummy run, register its model and alias it ``staging``; return the run id."""
    uri = setup_tracking()
    params = {"strategy": "most_frequent", "seed": 0}
    with start_run(
        phase=0, row="0.2.4", task="tracking", run_name="0-tracking-dummy-baseline-0", params=params
    ) as run:
        for step in range(5):
            mlflow.log_metric("loss", 1.0 / (step + 1), step=step)
            mlflow.log_metric("accuracy", 0.5 + 0.1 * step, step=step)
        mlflow.log_text("Terser dummy run.\n", "notes/readme.txt")
        mlflow.log_dict({"row": "0.2.4", "ok": True}, "notes/summary.json")
        X = np.zeros((4, 1))
        clf = DummyClassifier(strategy="most_frequent").fit(X, np.array([0, 1, 1, 1]))
        mlflow.sklearn.log_model(clf, name="model", pip_requirements=["scikit-learn"])
        run_id = run.info.run_id
    mv = register_model(run_id, "model", MODEL_NAME)
    promote(MODEL_NAME, mv.version, "Staging")
    print(f"run_id: {run_id}")
    print(f"registered: {MODEL_NAME} v{mv.version} (alias 'staging')")
    print(f'UI: mlflow ui --backend-store-uri "{uri}"')
    return run_id


if __name__ == "__main__":
    main()
