"""End-to-end trip inference and server-side scoring."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .data_io import load_trip
from .features import build_candidate_features
from .geo import TripProjection
from .hmm import HMMConfig, reconstruct_route, route_coordinates, viterbi
from .labels import derive_point_labels

FEATURE_COLUMNS = [
    "dist_m",
    "distance_rank",
    "heading_diff_deg",
    "speed_mps",
    "arc_length_m",
    "from_degree",
    "to_degree",
    "candidate_count",
    "distance_gap_m",
    "offset_fraction",
]


def load_model_artifacts(
    model_path: str | Path,
    feature_columns_path: str | Path,
) -> tuple[Any | None, list[str], str]:
    """Load the optional trained classifier; report distance-only mode if absent."""
    model_file = Path(model_path)
    columns_file = Path(feature_columns_path)
    if not model_file.is_file():
        return None, FEATURE_COLUMNS, "distance-only"
    if not columns_file.is_file():
        raise FileNotFoundError(
            f"Model feature schema is missing: {columns_file}. "
            "Train/export the model or remove the incomplete model artifact."
        )
    model = joblib.load(model_file)
    columns = json.loads(columns_file.read_text(encoding="utf-8"))
    if not isinstance(columns, list) or not columns or not all(
        isinstance(column, str) for column in columns
    ):
        raise ValueError(f"Invalid feature schema in {columns_file}")
    return model, columns, "classifier+hmm"


def candidate_radius(default_m: float = 80.0) -> float:
    """Use the radius selected by train-only preprocessing, when available."""
    root = Path(__file__).resolve().parents[1]
    path = root / "results" / "preprocessing.json"
    if not path.is_file():
        return default_m
    settings = json.loads(path.read_text(encoding="utf-8"))
    radius = float(settings["radius_m"])
    if radius <= 0:
        raise ValueError(f"Invalid candidate radius in {path}: {radius}")
    return radius


def score_candidates(
    candidates: pd.DataFrame,
    model: Any | None,
    feature_columns: list[str] | None = None,
    *,
    emission_sigma_m: float = 25.0,
) -> pd.DataFrame:
    """Attach classifier probabilities, or explicit Gaussian distance scores."""
    result = candidates.copy()
    if result.empty:
        result["prob"] = pd.Series(dtype=float)
        return result
    if model is None:
        distance = result["dist_m"].to_numpy(dtype=float)
        result["prob"] = np.exp(-0.5 * (distance / emission_sigma_m) ** 2)
        return result

    columns = feature_columns or FEATURE_COLUMNS
    missing = [column for column in columns if column not in result.columns]
    if missing:
        raise ValueError(f"Candidate table is missing model features: {missing}")
    features = result[columns].replace([np.inf, -np.inf], np.nan).copy()
    if "distance_gap_m" in features:
        features["distance_gap_m"] = features["distance_gap_m"].fillna(
            float(result.attrs.get("radius_m", candidate_radius()))
        )
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The classifier has no positive class (1)")
    result["prob"] = np.clip(probabilities[:, classes.index(1)], 1e-12, 1.0)
    return result


def _route_scores(predicted: list[int], reference: list[int]) -> tuple[float, float]:
    predicted_counts = Counter(predicted)
    reference_counts = Counter(reference)
    overlap = sum((predicted_counts & reference_counts).values())
    precision = overlap / sum(predicted_counts.values()) if predicted_counts else 0.0
    recall = overlap / sum(reference_counts.values()) if reference_counts else 0.0
    return float(precision), float(recall)


def compute_metrics(
    trip: dict,
    per_observation_arcs: np.ndarray,
    predicted_route: list[int],
    *,
    gap_threshold_s: float,
) -> dict[str, float | int | None]:
    """Compute evaluation metrics without returning ground-truth labels."""
    track = trip["track"]
    reference_route = trip.get("route")
    metrics: dict[str, float | int | None] = {
        "top1_acc": None,
        "macro_f1": None,
        "route_precision": None,
        "route_recall": None,
        "continuity": None,
        "n_obs": int(len(track)),
    }

    if reference_route:
        projection = TripProjection.from_coordinates(
            track["lon"].to_numpy(), track["lat"].to_numpy()
        )
        labels = derive_point_labels(
            track,
            trip["nodes"],
            trip["arcs"],
            reference_route,
            projection,
        )
        truth = labels["true_arc_id"].to_numpy(dtype=np.int64)
        metrics["top1_acc"] = float(np.mean(per_observation_arcs == truth))
        label_set = np.unique(np.concatenate((truth, per_observation_arcs)))
        metrics["macro_f1"] = float(
            f1_score(
                truth,
                per_observation_arcs,
                labels=label_set,
                average="macro",
                zero_division=0,
            )
        )
        precision, recall = _route_scores(predicted_route, reference_route)
        metrics["route_precision"] = precision
        metrics["route_recall"] = recall

    connected: list[bool] = []
    arcs = np.asarray(trip["arcs"], dtype=np.int64)
    for obs_idx in range(1, len(track)):
        previous_arc = int(per_observation_arcs[obs_idx - 1])
        next_arc = int(per_observation_arcs[obs_idx])
        elapsed = float(track.iloc[obs_idx]["t"] - track.iloc[obs_idx - 1]["t"])
        if elapsed <= gap_threshold_s and previous_arc >= 0 and next_arc >= 0:
            connected.append(
                previous_arc == next_arc
                or int(arcs[previous_arc, 1]) == int(arcs[next_arc, 0])
            )
    if connected:
        metrics["continuity"] = float(np.mean(connected))
    return metrics


def evaluate_prepared_trip(
    trip: dict,
    candidates: pd.DataFrame,
    *,
    config: HMMConfig,
    started_at: float | None = None,
) -> dict[str, Any]:
    """Decode a trip whose candidate probabilities have already been computed."""
    per_observation_arcs = viterbi(trip, candidates, config)
    route_arc_ids = reconstruct_route(
        trip,
        per_observation_arcs,
        gap_threshold_s=config.gap_threshold_s,
        max_search_m=config.max_search_m,
    )
    metrics = compute_metrics(
        trip,
        per_observation_arcs,
        route_arc_ids,
        gap_threshold_s=config.gap_threshold_s,
    )
    if started_at is not None:
        metrics["time_ms"] = round((time.perf_counter() - started_at) * 1000.0, 2)

    track = trip["track"]
    return {
        "trip_id": trip["trip_id"],
        "gps": [
            [float(lat), float(lon)]
            for lat, lon in zip(track["lat"].to_numpy(), track["lon"].to_numpy())
        ],
        "predicted_route": route_coordinates(trip, route_arc_ids),
        "per_obs_arcs": [int(arc_id) for arc_id in per_observation_arcs],
        "metrics": metrics,
    }


def prepare_trip(
    trip: dict,
    model: Any | None,
    feature_columns: list[str],
    *,
    radius_m: float,
    config: HMMConfig,
) -> pd.DataFrame:
    """Build candidates and attach real model scores or distance-only scores."""
    projection = TripProjection.from_coordinates(
        trip["track"]["lon"].to_numpy(), trip["track"]["lat"].to_numpy()
    )
    candidates = build_candidate_features(
        trip,
        radius_m=radius_m,
        projection=projection,
    )
    candidates.attrs["radius_m"] = radius_m
    return score_candidates(
        candidates,
        model,
        feature_columns,
        emission_sigma_m=config.emission_sigma_m,
    )


def predict_trip(
    trip_dir: str | Path,
    *,
    model: Any | None,
    feature_columns: list[str],
    radius_m: float,
    config: HMMConfig,
) -> dict[str, Any]:
    """Run candidate generation, scoring, Viterbi, route reconstruction and metrics."""
    started_at = time.perf_counter()
    trip = load_trip(trip_dir, include_route=True)
    candidates = prepare_trip(
        trip,
        model,
        feature_columns,
        radius_m=radius_m,
        config=config,
    )
    result = evaluate_prepared_trip(
        trip,
        candidates,
        config=config,
        started_at=started_at,
    )
    result["predictor"] = "classifier+hmm" if model is not None else "distance-only"
    return result
