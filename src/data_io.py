"""Load and clean one PathSnap trip without modifying raw files."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRACK_COLUMNS = ("lon", "lat", "t")


def _read_numeric(path: Path, dtype: type) -> np.ndarray:
    values = np.loadtxt(path, dtype=dtype, ndmin=2)
    if values.size == 0:
        return np.empty((0, 0), dtype=dtype)
    return values


def _trip_id(trip_dir: Path) -> str:
    return trip_dir.name


def load_trip(trip_dir: str | Path, *, include_route: bool = True) -> dict[str, Any]:
    """Load one trip directory using the repository's shared data contract.

    The raw files are read-only. Track rows are sorted by timestamp, invalid
    coordinates and duplicate timestamps are removed, and gap metadata is
    retained in ``track.attrs`` for EDA and downstream feature generation.
    """
    trip_dir = Path(trip_dir)
    trip_id = _trip_id(trip_dir)
    if not trip_dir.is_dir():
        raise FileNotFoundError(f"Trip directory does not exist: {trip_dir}")

    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"
    required = (track_path, nodes_path, arcs_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required trip files: {', '.join(missing)}")

    track_values = _read_numeric(track_path, float)
    if track_values.shape[1] != 3:
        raise ValueError(f"{track_path} must contain exactly 3 columns")
    track = pd.DataFrame(track_values, columns=TRACK_COLUMNS)
    track["original_idx"] = np.arange(len(track), dtype=np.int64)
    invalid = (
        ~np.isfinite(track[["lon", "lat", "t"]]).all(axis=1)
        | ~track["lon"].between(-180.0, 180.0)
        | ~track["lat"].between(-90.0, 90.0)
    )
    invalid_count = int(invalid.sum())
    track = track.loc[~invalid].sort_values("t", kind="mergesort")
    duplicate_count = int(track.duplicated("t").sum())
    track = track.drop_duplicates("t", keep="first").reset_index(drop=True)
    track["obs_idx"] = np.arange(len(track), dtype=np.int64)
    time_diffs = track["t"].diff()
    track["time_gap_s"] = time_diffs.fillna(0.0)
    track["is_gap_after"] = track["time_gap_s"].shift(-1, fill_value=0.0).gt(1.5)
    track.attrs.update(
        {
            "trip_id": trip_id,
            "invalid_rows_removed": invalid_count,
            "duplicate_timestamps_removed": duplicate_count,
            "gap_threshold_s": 1.5,
            "gap_count": int(track["is_gap_after"].sum()),
        }
    )

    nodes = _read_numeric(nodes_path, float)
    arcs = _read_numeric(arcs_path, np.int64)
    if nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError(f"{trip_id}: nodes must have 2 columns and arcs must have 2 columns")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"{trip_id}: arcs contain node IDs outside the nodes array")

    route: list[int] | None = None
    if include_route and route_path.is_file():
        route = [int(value) for value in route_path.read_text().split()]
        if any(arc_id < 0 or arc_id >= len(arcs) for arc_id in route):
            raise ValueError(f"{trip_id}: route contains an invalid arc ID")

    return {"trip_id": trip_id, "track": track, "nodes": nodes, "arcs": arcs, "route": route}
