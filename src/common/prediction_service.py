"""Shared prediction loading and validation helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.common.constants import DEFAULT_LOCAL_MODEL_PATH, DEFAULT_TRACKING_STAGE, FEATURE_COLUMNS
from src.common.io_utils import load_pickle, load_yaml, validate_columns
from src.common.tracking import configure_mlflow, get_tracking_settings


def validate_prediction_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    validated = validate_columns(dataframe, include_target=False).copy()
    for column in FEATURE_COLUMNS:
        validated[column] = pd.to_numeric(validated[column], errors="raise")
    return validated


def create_input_dataframe(input_payload: dict[str, list[Any]] | list[dict[str, Any]]) -> pd.DataFrame:
    dataframe = pd.DataFrame(input_payload)
    return validate_prediction_dataframe(dataframe)


def load_model_for_inference(
    *,
    params_path: str = "params.yaml",
    stage: str | None = None,
    local_model_path: str = DEFAULT_LOCAL_MODEL_PATH,
):
    params = load_yaml(params_path)
    tracking_enabled, _ = configure_mlflow(params)

    if tracking_enabled:
        import mlflow

        tracking_settings = get_tracking_settings(params)
        target_stage = stage or tracking_settings["registry_stage"] or DEFAULT_TRACKING_STAGE
        model_name = params.get("model", {}).get("registered_name", "Best Model")
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[target_stage])
        if versions:
            latest_version = versions[0]
            logged_model = f"runs:/{latest_version.run_id}/{model_name}"
            return mlflow.pyfunc.load_model(logged_model), {
                "source": "mlflow",
                "stage": target_stage,
                "model_name": model_name,
                "run_id": latest_version.run_id,
                "version": latest_version.version,
            }

    return load_pickle(local_model_path), {
        "source": "local",
        "path": local_model_path,
    }


def predict_dataframe(model, dataframe: pd.DataFrame) -> list[int]:
    validated = validate_prediction_dataframe(dataframe)
    predictions = model.predict(validated)
    return [int(value) for value in predictions]
