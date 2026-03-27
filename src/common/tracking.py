"""MLflow and DagsHub integration helpers."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

from src.common.constants import DEFAULT_TRACKING_STAGE


def get_tracking_settings(params: dict[str, Any]) -> dict[str, Any]:
    tracking = params.get("tracking", {})
    return {
        "enabled": bool(tracking.get("enabled", False)),
        "repo_owner": tracking.get("repo_owner", "sumitrwk90"),
        "repo_name": tracking.get("repo_name", "water-test"),
        "experiment_name": tracking.get("experiment_name", "Final_model"),
        "registry_stage": tracking.get("registry_stage", DEFAULT_TRACKING_STAGE),
    }


def configure_mlflow(params: dict[str, Any]) -> tuple[bool, str]:
    settings = get_tracking_settings(params)
    if not settings["enabled"]:
        return False, "Tracking disabled in params.yaml"

    try:
        import mlflow
    except ImportError:
        return False, "Tracking disabled because mlflow is not installed"

    dagshub_token = os.getenv("DAGSHUB_TOKENS")
    if not dagshub_token:
        return False, "Tracking disabled because DAGSHUB_TOKENS is not set"

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    tracking_uri = (
        f"https://dagshub.com/{settings['repo_owner']}/{settings['repo_name']}.mlflow"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(settings["experiment_name"])
    return True, "Tracking configured"


def maybe_start_run(enabled: bool, run_name: str):
    if enabled:
        import mlflow

        return mlflow.start_run(run_name=run_name)
    return nullcontext(None)
