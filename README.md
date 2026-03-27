# Water Potability Prediction

This project predicts whether a water sample is potable using a DVC-managed machine learning pipeline. The V2 workflow standardizes preprocessing and training inside a single sklearn pipeline so training, evaluation, and inference all use the same model contract.

## Pipeline flow

The DVC pipeline runs these stages:

1. `data_collection`  
   Loads `data_storage/water_potability.csv`, validates the schema, and creates stratified train/test splits in `data/raw/`.
2. `data_preparation`  
   Validates the raw splits and writes imputed snapshots to `data/processed/`.
3. `model_building`  
   Builds a fitted sklearn pipeline with mean imputation plus the configured model and saves it to `models/model.pkl`.
4. `model_evaluation`  
   Evaluates the saved pipeline, writes metrics and artifacts under `reports/`, and optionally logs to MLflow/DagsHub.
5. `model_registration`  
   Optionally registers the evaluated model and promotes it to the configured registry stage.

## Model configuration

Configuration lives in `params.yaml`.

- `model.name` selects the model to train.
- Supported values are `random_forest`, `logistic_regression`, and `xgboost`.
- `tracking.enabled` controls whether MLflow/DagsHub logging is used.
- `tracking.registry_stage` controls which stage registration targets and which stage inference uses by default.

## Running locally

Install dependencies and run the DVC pipeline:

```bash
pip install -r requirements.txt
dvc repro
```

Run the unit tests:

```bash
python -m unittest discover -s tests
```

Run a sample prediction:

```bash
python prediction.py
python prediction.py --stage Production
```

Launch the desktop GUI:

```bash
python GUI.py
```

## Remote tracking

Remote MLflow/DagsHub logging is optional. For local-only runs, keep `tracking.enabled: false`.

To enable remote tracking:

1. Set `tracking.enabled: true` in `params.yaml`.
2. Export `DAGSHUB_TOKENS` in your environment.
3. Keep the repo owner, repo name, and experiment name aligned with your DagsHub project.

If remote tracking is unavailable, evaluation still runs locally and registration is skipped gracefully.

## Inference inputs

Prediction expects these numeric inputs:

- `ph`
- `Hardness`
- `Solids`
- `Chloramines`
- `Sulfate`
- `Conductivity`
- `Organic_carbon`
- `Trihalomethanes`
- `Turbidity`

Both the CLI script and the Tkinter GUI use the same shared prediction service and input validation.
