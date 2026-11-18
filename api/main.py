"""FastAPI service for LAN-based PathSnap trip prediction."""

from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.hmm import HMMConfig
from src.pipeline import (
    candidate_radius,
    load_model_artifacts,
    predict_trip,
)

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pathsnap.api")

_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app = FastAPI(title="PathSnap", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    trip_id: str


def _environment_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _hmm_config() -> HMMConfig:
    params_path = ROOT / "models" / "hmm_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.is_file() else {}
    return HMMConfig(
        emission_sigma_m=_environment_float(
            "HMM_SIGMA_M", float(params.get("emission_sigma_m", 25.0))
        ),
        transition_beta_m=_environment_float(
            "HMM_BETA_M", float(params.get("transition_beta_m", 50.0))
        ),
        gap_threshold_s=_environment_float(
            "HMM_GAP_THRESHOLD_S", float(params.get("gap_threshold_s", 5.0))
        ),
        model_weight=_environment_float(
            "HMM_MODEL_WEIGHT", float(params.get("model_weight", 0.75))
        ),
        max_search_m=_environment_float(
            "HMM_MAX_SEARCH_M", float(params.get("max_search_m", 3000.0))
        ),
    )


HMM_CONFIG = _hmm_config()
RADIUS_M = _environment_float("CANDIDATE_RADIUS_M", candidate_radius())
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "final_model.joblib"))
FEATURE_COLUMNS_PATH = Path(
    os.getenv("FEATURE_COLUMNS_PATH", ROOT / "models" / "feature_cols.json")
)
MODEL, FEATURE_COLUMNS, PREDICTOR_MODE = load_model_artifacts(
    MODEL_PATH, FEATURE_COLUMNS_PATH
)
if MODEL is None:
    logger.warning(
        "No model artifact at %s; using the distance-only baseline. "
        "Train/export models/final_model.joblib to enable classifier scores.",
        MODEL_PATH,
    )
else:
    logger.info("Loaded model artifact from %s", MODEL_PATH)


def _trip_root() -> Path:
    configured = os.getenv("EVAL_DIR")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_dir():
            raise RuntimeError(f"EVAL_DIR does not exist: {path}")
        return path

    evaluator_path = ROOT / "data" / "eval"
    if evaluator_path.is_dir():
        return evaluator_path
    return ROOT / "data" / "student_test"


def _trip_directories() -> dict[str, Path]:
    root = _trip_root()
    if not root.is_dir():
        raise RuntimeError(f"Trip directory does not exist: {root}")
    return {
        child.name: child
        for child in sorted(root.iterdir())
        if child.is_dir()
        and (child / f"{child.name}.track").is_file()
        and (child / f"{child.name}.nodes").is_file()
        and (child / f"{child.name}.arcs").is_file()
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "predictor": PREDICTOR_MODE}


@app.get("/trips")
def trips() -> dict[str, Any]:
    try:
        ids = sorted(_trip_directories())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"trips": [{"trip_id": trip_id} for trip_id in ids], "predictor": PREDICTOR_MODE}


@app.post("/predict")
def predict(payload: PredictRequest, request: Request) -> dict[str, Any]:
    try:
        directories = _trip_directories()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    trip_dir = directories.get(payload.trip_id)
    if trip_dir is None:
        raise HTTPException(status_code=404, detail="Unknown trip_id")

    client = request.client.host if request.client else "unknown"
    logger.info("Prediction requested trip=%s client=%s", payload.trip_id, client)
    try:
        result = predict_trip(
            trip_dir,
            model=MODEL,
            feature_columns=FEATURE_COLUMNS,
            radius_m=RADIUS_M,
            config=HMM_CONFIG,
        )
    except Exception as exc:
        logger.exception("Prediction failed trip=%s client=%s", payload.trip_id, client)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    logger.info(
        "Prediction complete trip=%s client=%s predictor=%s n_obs=%s time_ms=%s",
        payload.trip_id,
        client,
        result["predictor"],
        result["metrics"]["n_obs"],
        result["metrics"]["time_ms"],
    )
    return result
