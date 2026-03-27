from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.common.constants import DEFAULT_LOCAL_MODEL_PATH, FEATURE_COLUMNS
from src.common.prediction_service import create_input_dataframe, load_model_for_inference, predict_dataframe


class WaterSample(BaseModel):
    ph: float = Field(..., description="pH value of the water sample")
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


class PredictionRequest(BaseModel):
    samples: list[WaterSample] = Field(..., min_length=1, description="One or more water samples")


class PredictionResponse(BaseModel):
    predictions: list[int]
    model_source: dict[str, Any]
    feature_order: list[str]


@lru_cache(maxsize=1)
def get_model_bundle():
    stage = os.getenv("MODEL_STAGE")
    local_model_path = os.getenv("LOCAL_MODEL_PATH", DEFAULT_LOCAL_MODEL_PATH)
    return load_model_for_inference(stage=stage, local_model_path=local_model_path)


app = FastAPI(
    title="Water Potability API",
    version="1.0.0",
    description="FastAPI service for water potability predictions using the shared project pipeline.",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Water Potability API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health() -> dict[str, str]:
    try:
        _, metadata = get_model_bundle()
        return {
            "status": "ok",
            "model_source": str(metadata.get("source", "unknown")),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model, metadata = get_model_bundle()
        dataframe = create_input_dataframe([sample.model_dump() for sample in request.samples])
        predictions = predict_dataframe(model, dataframe)
        return PredictionResponse(
            predictions=predictions,
            model_source=metadata,
            feature_order=FEATURE_COLUMNS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Local model file not found: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
