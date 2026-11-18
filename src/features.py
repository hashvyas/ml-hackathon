"""Candidate road segments and non-leaking geometric features."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import TripProjection, point_to_segments


def build_candidate_features(
    trip: dict,
    *,
    radius_m: float = 50.0,
    projection: TripProjection | None = None,
    labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one feature row per GPS observation and nearby directed arc."""
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    track = trip["track"]
    nodes = np.asarray(trip["nodes"])
    arcs = np.asarray(trip["arcs"])
    if projection is None:
        projection = TripProjection.from_coordinates(
            track["lon"].to_numpy(), track["lat"].to_numpy()
        )

    gps_x, gps_y = projection.project(track["lon"].to_numpy(), track["lat"].to_numpy())
    node_x, node_y = projection.project(nodes[:, 0], nodes[:, 1])
    starts = np.column_stack((node_x[arcs[:, 0]], node_y[arcs[:, 0]]))
    ends = np.column_stack((node_x[arcs[:, 1]], node_y[arcs[:, 1]]))
    vectors = ends - starts
    arc_lengths = np.linalg.norm(vectors, axis=1)
    midpoints = (starts + ends) / 2.0
    tree = cKDTree(midpoints)
    max_arc_length = float(arc_lengths.max()) if len(arc_lengths) else 0.0
    midpoint_search_radius = radius_m + max_arc_length / 2.0

    out: list[pd.DataFrame] = []
    degrees = np.bincount(arcs.ravel(), minlength=len(nodes))
    gps_points = np.column_stack((gps_x, gps_y))
    gps_heading = np.degrees(np.arctan2(np.diff(gps_y, prepend=gps_y[0]), np.diff(gps_x, prepend=gps_x[0])))
    dt = track["t"].diff().to_numpy()
    displacement = np.hypot(np.diff(gps_x, prepend=gps_x[0]), np.diff(gps_y, prepend=gps_y[0]))
    speed = np.divide(displacement, dt, out=np.zeros_like(displacement), where=dt > 0)

    for obs_idx, point in enumerate(gps_points):
        candidate_ids = np.asarray(
            tree.query_ball_point(point, midpoint_search_radius), dtype=np.int64
        )
        if len(candidate_ids) == 0:
            continue
        distances, offsets = point_to_segments(
            np.repeat(point[None, :], len(candidate_ids), axis=0),
            starts[candidate_ids],
            ends[candidate_ids],
        )
        within_radius = distances <= radius_m
        candidate_ids = candidate_ids[within_radius]
        distances = distances[within_radius]
        offsets = offsets[within_radius]
        if len(candidate_ids) == 0:
            continue
        order = np.argsort(distances, kind="stable")
        candidate_ids = candidate_ids[order]
        distances = distances[order]
        offsets = offsets[order]
        arc_heading = np.degrees(np.arctan2(vectors[candidate_ids, 1], vectors[candidate_ids, 0]))
        heading_diff = np.abs((gps_heading[obs_idx] - arc_heading + 180.0) % 360.0 - 180.0)
        gap = float(distances[1] - distances[0]) if len(distances) > 1 else np.nan
        out.append(
            pd.DataFrame(
                {
                    "obs_idx": obs_idx,
                    "arc_id": candidate_ids,
                    "dist_m": distances,
                    "distance_rank": np.arange(len(candidate_ids), dtype=np.int64),
                    "heading_diff_deg": heading_diff,
                    "speed_mps": speed[obs_idx],
                    "arc_length_m": arc_lengths[candidate_ids],
                    "from_degree": degrees[arcs[candidate_ids, 0]],
                    "to_degree": degrees[arcs[candidate_ids, 1]],
                    "candidate_count": len(candidate_ids),
                    "distance_gap_m": gap,
                    "offset_fraction": offsets,
                }
            )
        )

    columns = [
        "obs_idx", "arc_id", "dist_m", "distance_rank", "heading_diff_deg",
        "speed_mps", "arc_length_m", "from_degree", "to_degree",
        "candidate_count", "distance_gap_m", "offset_fraction",
    ]
    result = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=columns)
    if labels is not None:
        result = result.merge(
            labels[["obs_idx", "true_arc_id"]],
            on="obs_idx",
            how="left",
            validate="many_to_one",
        )
        result["label"] = (result["arc_id"] == result["true_arc_id"]).astype(np.int8)
    result.attrs["radius_m"] = radius_m
    result.attrs["candidate_miss_rate"] = candidate_miss_rate(result, labels)
    return result


def candidate_miss_rate(
    candidates: pd.DataFrame, labels: pd.DataFrame | None = None
) -> float:
    """Return the fraction of observations whose true arc is not a candidate."""
    if labels is None or labels.empty:
        return 0.0
    candidate_pairs = candidates[["obs_idx", "arc_id"]].drop_duplicates()
    labelled_pairs = labels[["obs_idx", "true_arc_id"]].rename(
        columns={"true_arc_id": "arc_id"}
    )
    covered = labelled_pairs.merge(
        candidate_pairs, on=["obs_idx", "arc_id"], how="inner"
    )["obs_idx"].nunique()
    return float(1.0 - covered / len(labelled_pairs))


if __name__ == "__main__":
    # R sweep on TRAIN trips only: candidate miss rate per radius, then choose R.
    import json
    from pathlib import Path

    from .data_io import load_trip
    from .labels import derive_point_labels, label_mismatch_rate

    radii = (30.0, 50.0, 80.0)
    rows, n_obs, n_invalid, n_dup, mismatches = [], 0, 0, 0, []
    for trip_dir in sorted(Path("data/train").iterdir()):
        trip = load_trip(trip_dir)
        attrs = trip["track"].attrs
        n_obs += len(trip["track"])
        n_invalid += attrs["invalid_rows_removed"]
        n_dup += attrs["duplicate_timestamps_removed"]
        proj = TripProjection.from_coordinates(trip["track"]["lon"].to_numpy(), trip["track"]["lat"].to_numpy())
        labels = derive_point_labels(trip["track"], trip["nodes"], trip["arcs"], trip["route"], proj)
        mismatches.append(label_mismatch_rate(labels) * len(labels))
        cands = build_candidate_features(trip, radius_m=max(radii), projection=proj, labels=labels)
        for r in radii:  # a smaller R is just a distance filter on the largest-R candidate set
            rows.append({"trip_id": trip["trip_id"], "radius_m": r, "n_obs": len(labels),
                         "miss_rate": candidate_miss_rate(cands[cands["dist_m"] <= r], labels),
                         "cands_per_obs": (cands["dist_m"] <= r).sum() / len(labels)})
    df = pd.DataFrame(rows)
    df["missed"] = df["miss_rate"] * df["n_obs"]
    summary = df.groupby("radius_m").apply(
        lambda g: pd.Series({"miss_rate": g["missed"].sum() / g["n_obs"].sum(),
                             "cands_per_obs": (g["cands_per_obs"] * g["n_obs"]).sum() / g["n_obs"].sum()}),
        include_groups=False).reset_index()
    # rule: smallest R whose miss rate is within 0.5 percentage points of the largest R (fewer candidates, same recall)
    best = summary["miss_rate"].min()
    chosen = float(summary.loc[summary["miss_rate"] <= best + 0.005, "radius_m"].min())
    Path("results").mkdir(exist_ok=True)
    summary.to_csv("results/candidate_miss_rate.csv", index=False)
    info = {"radius_m": chosen, "rule": "smallest R with miss_rate <= min + 0.005",
            "train_obs_after_cleaning": n_obs, "invalid_rows_removed": int(n_invalid),
            "duplicate_timestamps_removed": int(n_dup), "label_mismatch_rate": sum(mismatches) / n_obs}
    Path("results/preprocessing.json").write_text(json.dumps(info, indent=2))
    print("PREPROCESSING (train):", f"obs kept={n_obs}, dropped NaN/out-of-range={n_invalid}, dropped duplicate t={n_dup}")
    print(f"LABEL MISMATCH RATE (monotonic label != nearest route arc): {info['label_mismatch_rate']:.4f}")
    print("CANDIDATE MISS RATE per R:\n", summary.to_string(index=False))
    print("CHOSEN R:", chosen, "m")
