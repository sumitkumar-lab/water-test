from __future__ import annotations

import os
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.constants import DEFAULT_SOURCE_DATA_PATH
from src.common.io_utils import load_dataset, load_yaml, save_dataframe


def main() -> None:
    params = load_yaml("params.yaml")
    test_size = float(params.get("data_collection", {}).get("test_size", 0.2))
    raw_data_path = os.path.join("data", "raw")

    dataset = load_dataset(DEFAULT_SOURCE_DATA_PATH)
    train_data, test_data = train_test_split(
        dataset,
        test_size=test_size,
        random_state=42,
        stratify=dataset["Potability"],
    )

    save_dataframe(train_data, os.path.join(raw_data_path, "train.csv"))
    save_dataframe(test_data, os.path.join(raw_data_path, "test.csv"))


if __name__ == "__main__":
    main()
