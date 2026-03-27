from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.constants import FEATURE_COLUMNS
from src.common.io_utils import load_dataset, save_dataframe


def impute_numeric_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    processed = dataframe.copy()
    for column in FEATURE_COLUMNS:
        processed[column] = pd.to_numeric(processed[column], errors="raise")
        processed[column] = processed[column].fillna(processed[column].mean())
    return processed


def main() -> None:
    raw_data_path = os.path.join("data", "raw")
    processed_data_path = os.path.join("data", "processed")

    train_data = load_dataset(os.path.join(raw_data_path, "train.csv"))
    test_data = load_dataset(os.path.join(raw_data_path, "test.csv"))

    train_processed_data = impute_numeric_features(train_data)
    test_processed_data = impute_numeric_features(test_data)

    save_dataframe(
        train_processed_data,
        os.path.join(processed_data_path, "train_processed.csv"),
    )
    save_dataframe(
        test_processed_data,
        os.path.join(processed_data_path, "test_processed.csv"),
    )


if __name__ == "__main__":
    main()
