"""Derive monotonic GPS-point labels from a trip reference route."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .geo import TripProjection, point_to_segments


def derive_point_labels(
    track: pd.DataFrame,
    nodes: np.ndarray,
    arcs: np.ndarray,
    route: Sequence[int],
    projection: TripProjection | None = None,
    *,
    lookahead: int = 8,
) -> pd.DataFrame:
    """Assign each observation to a nearby route arc using a monotonic window."""
    if not route:
        raise ValueError("A non-empty route is required to derive labels")
    if lookahead < 0:
        raise ValueError("lookahead must be non-negative")
    if projection is None:
        projection = TripProjection.from_coordinates(
            track["lon"].to_numpy(), track["lat"].to_numpy()
        )

    gps_x, gps_y = projection.project(
        track["lon"].to_numpy(), track["lat"].to_numpy()
    )
    node_x, node_y = projection.project(nodes[:, 0], nodes[:, 1])
    route_array = np.asarray(route, dtype=np.int64)
    starts = np.column_stack((node_x[arcs[route_array, 0]], node_y[arcs[route_array, 0]]))
    ends = np.column_stack((node_x[arcs[route_array, 1]], node_y[arcs[route_array, 1]]))

    route_positions: list[int] = []
    distances: list[float] = []
    nearest_any: list[int] = []  # unconstrained nearest route arc, used only to measure label mismatch
    current = 0
    for point in np.column_stack((gps_x, gps_y)):
        stop = min(len(route_array), current + lookahead + 1)
        candidate_positions = np.arange(current, stop)
        candidate_points = np.repeat(point[None, :], len(candidate_positions), axis=0)
        candidate_distances, _ = point_to_segments(
            candidate_points, starts[candidate_positions], ends[candidate_positions]
        )
        all_distances, _ = point_to_segments(
            np.repeat(point[None, :], len(route_array), axis=0), starts, ends
        )
        nearest_any.append(int(route_array[np.argmin(all_distances)]))
        nearest_index = int(np.argmin(candidate_distances))
        selected = int(candidate_positions[nearest_index])
        current = selected
        route_positions.append(selected)
        distances.append(float(candidate_distances[nearest_index]))

    labels = pd.DataFrame(
        {
            "obs_idx": track["obs_idx"].to_numpy(),
            "true_arc_id": route_array[np.asarray(route_positions)],
            "route_pos": route_positions,
            "label_distance_m": distances,
            "nearest_route_arc": nearest_any,
        }
    )
    labels.attrs["route_length"] = len(route_array)
    labels.attrs["lookahead"] = lookahead
    return labels


def label_mismatch_rate(labels: pd.DataFrame) -> float:
    """Share of points whose monotonic label differs from the plain nearest route arc."""
    return float((labels["true_arc_id"] != labels["nearest_route_arc"]).mean())
