"""I/O and dataset helpers shared across the project."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.common.constants import FEATURE_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN


def load_yaml(filepath: str | os.PathLike[str]) -> dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object in {filepath}")
    return data


def ensure_directory(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def validate_columns(
    dataframe: pd.DataFrame,
    *,
    include_target: bool = True,
) -> pd.DataFrame:
    expected = REQUIRED_COLUMNS if include_target else FEATURE_COLUMNS
    missing = [column for column in expected if column not in dataframe.columns]
    extra = [column for column in dataframe.columns if column not in expected]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected columns found: {extra}")

    return dataframe[expected] if include_target else dataframe[FEATURE_COLUMNS]


def load_dataset(
    filepath: str | os.PathLike[str],
    *,
    include_target: bool = True,
) -> pd.DataFrame:
    dataframe = pd.read_csv(filepath)
    return validate_columns(dataframe, include_target=include_target).copy()


def save_dataframe(dataframe: pd.DataFrame, filepath: str | os.PathLike[str]) -> None:
    target = Path(filepath)
    ensure_directory(target.parent)
    dataframe.to_csv(target, index=False)


def save_json(payload: dict[str, Any], filepath: str | os.PathLike[str]) -> None:
    target = Path(filepath)
    ensure_directory(target.parent)
    with open(target, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def load_json(filepath: str | os.PathLike[str]) -> dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {filepath}")
    return data


def save_pickle(model: Any, filepath: str | os.PathLike[str]) -> None:
    target = Path(filepath)
    ensure_directory(target.parent)
    with open(target, "wb") as file:
        pickle.dump(model, file)


def load_pickle(filepath: str | os.PathLike[str]) -> Any:
    with open(filepath, "rb") as file:
        return pickle.load(file)


def split_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validated = validate_columns(dataframe, include_target=True)
    features = validated[FEATURE_COLUMNS].copy()
    target = validated[TARGET_COLUMN].copy()
    if target.isnull().any():
        raise ValueError("Target column contains missing values")
    return features, target


def summarize_class_distribution(target: pd.Series) -> dict[str, dict[str, float | int]]:
    counts = target.value_counts(dropna=False).sort_index()
    total = int(counts.sum())
    return {
        str(label): {
            "count": int(count),
            "ratio": float(count / total) if total else 0.0,
        }
        for label, count in counts.items()
    }
