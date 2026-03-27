from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.constants import DEFAULT_LOCAL_MODEL_PATH
from src.common.io_utils import load_dataset, load_yaml, save_pickle, split_features_target
from src.common.modeling import build_training_pipeline


def main() -> None:
    params = load_yaml("params.yaml")
    train_data = load_dataset("./data/processed/train_processed.csv")
    features, target = split_features_target(train_data)

    model_pipeline = build_training_pipeline(params)
    model_pipeline.fit(features, target)

    save_pickle(model_pipeline, DEFAULT_LOCAL_MODEL_PATH)


if __name__ == "__main__":
    main()
