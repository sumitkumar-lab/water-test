from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io_utils import load_json, load_yaml, save_json
from src.common.tracking import configure_mlflow, get_tracking_settings


def main() -> None:
    params = load_yaml("params.yaml")
    run_info = load_json("reports/run_info.json")
    tracking_enabled, tracking_reason = configure_mlflow(params)
    tracking_settings = get_tracking_settings(params)

    registry_info = {
        "status": "skipped",
        "message": tracking_reason,
        "model_name": run_info.get("model_name"),
        "target_stage": tracking_settings["registry_stage"],
        "version": None,
    }

    if tracking_enabled and run_info.get("run_id"):
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        model_uri = f"runs:/{run_info['run_id']}/{run_info['model_name']}"
        registered_model = mlflow.register_model(model_uri, run_info["model_name"])

        client.transition_model_version_stage(
            name=run_info["model_name"],
            version=registered_model.version,
            stage=tracking_settings["registry_stage"],
            archive_existing_versions=True,
        )

        registry_info.update(
            {
                "status": "registered",
                "message": "Model registered successfully",
                "version": registered_model.version,
            }
        )

    save_json(registry_info, "reports/registry_info.json")


if __name__ == "__main__":
    main()
