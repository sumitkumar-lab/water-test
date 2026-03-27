import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.common.constants import FEATURE_COLUMNS, TARGET_COLUMN
from src.common.io_utils import load_pickle, save_pickle, split_features_target
from src.common.modeling import build_training_pipeline
from src.common.prediction_service import validate_prediction_dataframe


class PipelineV2Tests(unittest.TestCase):
    def setUp(self):
        base_row = {
            "ph": 7.0,
            "Hardness": 200.0,
            "Solids": 15000.0,
            "Chloramines": 6.5,
            "Sulfate": 300.0,
            "Conductivity": 450.0,
            "Organic_carbon": 12.0,
            "Trihalomethanes": 70.0,
            "Turbidity": 4.0,
        }
        rows = []
        for index in range(10):
            row = base_row.copy()
            row["ph"] += index * 0.1
            row["Hardness"] += index
            row["Potability"] = index % 2
            rows.append(row)
        self.dataset = pd.DataFrame(rows)
        self.dataset.loc[0, "Sulfate"] = None

    def test_split_features_target_preserves_expected_columns(self):
        features, target = split_features_target(self.dataset)
        self.assertEqual(list(features.columns), FEATURE_COLUMNS)
        self.assertEqual(target.name, TARGET_COLUMN)

    def test_training_pipeline_can_fit_and_reload(self):
        params = {
            "model": {"name": "random_forest"},
            "random_forest": {"n_estimators": 10, "max_depth": 3},
        }
        features, target = split_features_target(self.dataset)
        pipeline = build_training_pipeline(params)
        pipeline.fit(features, target)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pkl"
            save_pickle(pipeline, path)
            reloaded = load_pickle(path)
            predictions = reloaded.predict(features)
            self.assertEqual(len(predictions), len(features))

    def test_prediction_validation_rejects_missing_columns(self):
        invalid = self.dataset[FEATURE_COLUMNS[:-1]].copy()
        with self.assertRaises(ValueError):
            validate_prediction_dataframe(invalid)


if __name__ == "__main__":
    unittest.main()
