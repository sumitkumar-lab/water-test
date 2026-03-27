"""Model building utilities."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.common.constants import FEATURE_COLUMNS

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def get_model_name(params: dict[str, Any]) -> str:
    model_name = params.get("model", {}).get("name", "random_forest")
    if not isinstance(model_name, str):
        raise ValueError("model.name must be a string")
    return model_name.strip().lower()


def build_estimator(params: dict[str, Any]):
    model_name = get_model_name(params)

    if model_name == "random_forest":
        config = params.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=int(config.get("n_estimators", 1000)),
            max_depth=config.get("max_depth"),
            random_state=42,
        )

    if model_name == "logistic_regression":
        config = params.get("logistic_regression", {})
        return LogisticRegression(
            max_iter=int(config.get("max_iter", 1000)),
            random_state=42,
        )

    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ValueError("xgboost model was selected but xgboost is not installed")
        config = params.get("xgboost", {})
        return XGBClassifier(
            n_estimators=int(config.get("n_estimators", 300)),
            max_depth=int(config.get("max_depth", 6)),
            learning_rate=float(config.get("learning_rate", 0.1)),
            eval_metric="logloss",
            random_state=42,
        )

    raise ValueError(
        "Unsupported model.name. Expected one of: random_forest, logistic_regression, xgboost"
    )


def build_training_pipeline(params: dict[str, Any]) -> Pipeline:
    estimator = build_estimator(params)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", estimator),
        ]
    )


def get_feature_columns() -> list[str]:
    return FEATURE_COLUMNS.copy()
