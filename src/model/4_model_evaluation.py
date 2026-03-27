from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.constants import DEFAULT_LOCAL_MODEL_PATH
from src.common.io_utils import (
    load_dataset,
    load_pickle,
    load_yaml,
    save_json,
    split_features_target,
    summarize_class_distribution,
)
from src.common.modeling import get_model_name
from src.common.tracking import configure_mlflow, maybe_start_run


def create_confusion_matrix_plot(
    y_true: pd.Series,
    y_pred: list[int],
    output_path: str,
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    params = load_yaml("params.yaml")
    test_data = load_dataset("./data/processed/test_processed.csv")
    train_data = load_dataset("./data/processed/train_processed.csv")
    model = load_pickle(DEFAULT_LOCAL_MODEL_PATH)

    x_test, y_test = split_features_target(test_data)
    _, y_train = split_features_target(train_data)

    y_pred = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if probabilities is not None else None,
    }

    class_distribution = {
        "train": summarize_class_distribution(y_train),
        "test": summarize_class_distribution(y_test),
    }

    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    metrics_path = os.path.join(reports_dir, "metrics.json")
    run_info_path = os.path.join(reports_dir, "run_info.json")
    class_distribution_path = os.path.join(reports_dir, "class_distribution.json")
    confusion_matrix_path = os.path.join(reports_dir, "confusion_matrix.png")
    model_name = params.get("model", {}).get("registered_name", "Best Model")

    create_confusion_matrix_plot(
        y_test,
        y_pred,
        confusion_matrix_path,
        f"Confusion Matrix for {get_model_name(params)}",
    )

    tracking_enabled, tracking_reason = configure_mlflow(params)
    run_info = {
        "tracking_enabled": tracking_enabled,
        "tracking_message": tracking_reason,
        "model_name": model_name,
        "registered_stage": params.get("tracking", {}).get("registry_stage", "Staging"),
        "run_id": None,
    }

    with maybe_start_run(tracking_enabled, run_name=f"{get_model_name(params)}_evaluation") as run:
        if tracking_enabled:
            import mlflow
            import mlflow.sklearn
            from mlflow.models import infer_signature

            mlflow.log_metrics({key: value for key, value in metrics.items() if value is not None})
            mlflow.log_params(
                {
                    "test_size": params.get("data_collection", {}).get("test_size", 0.2),
                    "selected_model": get_model_name(params),
                }
            )
            mlflow.log_artifact(DEFAULT_LOCAL_MODEL_PATH)
            mlflow.log_artifact(confusion_matrix_path)
            signature = infer_signature(x_test, model.predict(x_test))
            mlflow.sklearn.log_model(model, model_name, signature=signature)
            run_info["run_id"] = run.info.run_id

    save_json(metrics, metrics_path)
    save_json(class_distribution, class_distribution_path)
    save_json(run_info, run_info_path)


if __name__ == "__main__":
    main()
