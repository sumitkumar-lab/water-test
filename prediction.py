from __future__ import annotations

import argparse

from src.common.prediction_service import create_input_dataframe, load_model_for_inference, predict_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sample water potability prediction.")
    parser.add_argument("--stage", default=None, help="Remote registry stage to load, defaults to params.yaml")
    parser.add_argument("--local-model-path", default="models/model.pkl", help="Fallback local model path")
    args = parser.parse_args()

    sample_input = {
        "ph": [3.71608],
        "Hardness": [204.89045],
        "Solids": [20791.318981],
        "Chloramines": [7.300212],
        "Sulfate": [368.516441],
        "Conductivity": [564.308654],
        "Organic_carbon": [10.379783],
        "Trihalomethanes": [86.99097],
        "Turbidity": [2.963135],
    }

    model, metadata = load_model_for_inference(
        stage=args.stage,
        local_model_path=args.local_model_path,
    )
    print("======= Dataframe is creating =======")
    dataframe = create_input_dataframe(sample_input)
    print("======= Model started predicting =======")
    prediction = predict_dataframe(model, dataframe)

    print(f"Your Model source is: {metadata}")
    print(f"Your Prediction is: {prediction}")


if __name__ == "__main__":
    main()
