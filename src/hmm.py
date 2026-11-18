"""Distance- and score-based hidden Markov map matching."""

from __future__ import annotations

import heapq
from collections import OrderedDict
from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

from .geo import TripProjection


@dataclass(frozen=True)
class HMMConfig:
    emission_sigma_m: float = 25.0
    transition_beta_m: float = 50.0
    gap_threshold_s: float = 5.0
    model_weight: float = 0.75
    max_search_m: float = 3000.0

    def __post_init__(self) -> None:
        if self.emission_sigma_m <= 0 or self.transition_beta_m <= 0:
            raise ValueError("HMM emission sigma and transition beta must be positive")
        if self.gap_threshold_s <= 0 or self.max_search_m <= 0:
            raise ValueError("HMM gap threshold and search distance must be positive")
        if not 0.0 <= self.model_weight <= 1.0:
            raise ValueError("model_weight must be between 0 and 1")


class _RoadGraph:
    """Directed road graph with bounded, cached Dijkstra searches."""

    def __init__(self, trip: dict) -> None:
        self.arcs = np.asarray(trip["arcs"], dtype=np.int64)
        nodes = np.asarray(trip["nodes"], dtype=float)
        track = trip["track"]
        projection = TripProjection.from_coordinates(
            track["lon"].to_numpy(), track["lat"].to_numpy()
        )
        node_x, node_y = projection.project(nodes[:, 0], nodes[:, 1])
        self.node_xy = np.column_stack((node_x, node_y))
        vectors = self.node_xy[self.arcs[:, 1]] - self.node_xy[self.arcs[:, 0]]
        self.arc_lengths = np.linalg.norm(vectors, axis=1)
        self.adjacency: list[list[tuple[int, int, float]]] = [
            [] for _ in range(len(nodes))
        ]
        for arc_id, (from_node, to_node) in enumerate(self.arcs):
            length = float(self.arc_lengths[arc_id])
            self.adjacency[int(from_node)].append((int(to_node), arc_id, length))
        self.distance_cache: OrderedDict[tuple[int, int], dict[int, float]] = OrderedDict()
        self.cache_limit = 256

    @staticmethod
    def _bucket(max_distance: float) -> int:
        return int(ceil(max_distance / 250.0) * 250)

    def distances_from(self, source: int, max_distance: float) -> dict[int, float]:
        bound = self._bucket(max_distance)
        key = (source, bound)
        cached = self.distance_cache.get(key)
        if cached is not None:
            self.distance_cache.move_to_end(key)
            return cached

        distances = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            for neighbor, _, length in self.adjacency[node]:
                next_distance = distance + length
                if next_distance > bound:
                    continue
                if next_distance < distances.get(neighbor, float("inf")):
                    distances[neighbor] = next_distance
                    heapq.heappush(queue, (next_distance, neighbor))

        self.distance_cache[key] = distances
        if len(self.distance_cache) > self.cache_limit:
            self.distance_cache.popitem(last=False)
        return distances

    def path(self, source: int, target: int, max_distance: float) -> list[int] | None:
        if source == target:
            return []
        distances: dict[int, float] = {source: 0.0}
        previous: dict[int, tuple[int, int]] = {}
        queue = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            if node == target:
                break
            for neighbor, arc_id, length in self.adjacency[node]:
                next_distance = distance + length
                if next_distance > max_distance:
                    continue
                if next_distance < distances.get(neighbor, float("inf")):
                    distances[neighbor] = next_distance
                    previous[neighbor] = (node, arc_id)
                    heapq.heappush(queue, (next_distance, neighbor))
        if target not in previous:
            return None

        route: list[int] = []
        node = target
        while node != source:
            node, arc_id = previous[node]
            route.append(arc_id)
        route.reverse()
        return route


def _transition_log_probability(
    graph: _RoadGraph,
    previous_arc: int,
    next_arc: int,
    previous_offset: float,
    next_offset: float,
    gps_distance: float,
    beta_m: float,
    max_search_m: float,
) -> float:
    previous_to = graph.arcs[previous_arc, 1]
    next_from, _ = graph.arcs[next_arc]
    if previous_arc == next_arc:
        route_distance = max(
            0.0,
            (next_offset - previous_offset) * graph.arc_lengths[previous_arc],
        )
    else:
        search_limit = min(max_search_m, max(500.0, gps_distance * 4.0 + 100.0))
        distances = graph.distances_from(int(previous_to), search_limit)
        middle_distance = distances.get(int(next_from))
        if middle_distance is None:
            # Unreachable within the search bound means the road distance exceeds search_limit,
            # so penalise it at least as much as the worst reachable pair (a fixed -20 ignored beta).
            return -(search_limit - gps_distance) / beta_m
        route_distance = (
            (1.0 - previous_offset) * graph.arc_lengths[previous_arc]
            + middle_distance
            + next_offset * graph.arc_lengths[next_arc]
        )
    return -abs(float(route_distance) - gps_distance) / beta_m


def _finish_segment(
    groups: list[dict[str, np.ndarray]],
    backpointers: list[np.ndarray | None],
    final_scores: np.ndarray,
    output: np.ndarray,
) -> None:
    state_idx = int(np.argmax(final_scores))
    for group_idx in range(len(groups) - 1, -1, -1):
        group = groups[group_idx]
        output[int(group["obs_idx"][0])] = int(group["arc_ids"][state_idx])
        if group_idx:
            backpointer = backpointers[group_idx]
            if backpointer is None:
                raise RuntimeError("Missing Viterbi backpointer in a connected segment")
            state_idx = int(backpointer[state_idx])


def viterbi(
    trip: dict,
    candidates: pd.DataFrame,
    config: HMMConfig | None = None,
) -> np.ndarray:
    """Return one selected arc per observation, or -1 when no candidate exists."""
    config = config or HMMConfig()
    n_observations = len(trip["track"])
    output = np.full(n_observations, -1, dtype=np.int64)
    if candidates.empty:
        return output
    required = {"obs_idx", "arc_id", "dist_m"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"Candidate table must include {sorted(required)}")

    graph = _RoadGraph(trip)
    track = trip["track"]
    times = track["t"].to_numpy(dtype=float)
    gps_x, gps_y = TripProjection.from_coordinates(
        track["lon"].to_numpy(), track["lat"].to_numpy()
    ).project(track["lon"].to_numpy(), track["lat"].to_numpy())
    grouped = candidates.sort_values(["obs_idx", "arc_id"], kind="stable").groupby(
        "obs_idx", sort=True
    )
    groups: list[dict[str, np.ndarray]] = []
    for raw_obs_idx, frame in grouped:
        obs_idx = int(raw_obs_idx)
        if obs_idx < 0 or obs_idx >= n_observations:
            raise ValueError(f"Candidate observation index is out of range: {obs_idx}")
        arc_ids = frame["arc_id"].to_numpy(dtype=np.int64)
        if np.any(arc_ids < 0) or np.any(arc_ids >= len(graph.arcs)):
            raise ValueError("Candidate table contains an invalid arc ID")
        distances = frame["dist_m"].to_numpy(dtype=float)
        probabilities = (
            frame["prob"].to_numpy(dtype=float)
            if "prob" in frame
            else np.exp(-0.5 * (distances / config.emission_sigma_m) ** 2)
        )
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        offsets = (
            frame["offset_fraction"].to_numpy(dtype=float)
            if "offset_fraction" in frame
            else np.full(len(frame), 0.5, dtype=float)
        )
        emission = (
            config.model_weight * np.log(probabilities)
            + (1.0 - config.model_weight)
            * (-0.5 * (distances / config.emission_sigma_m) ** 2)
        )
        emission -= np.max(emission)
        groups.append(
            {
                "obs_idx": np.asarray([int(obs_idx)], dtype=np.int64),
                "arc_ids": arc_ids,
                "offsets": np.clip(offsets, 0.0, 1.0),
                "gps": np.asarray([gps_x[int(obs_idx)], gps_y[int(obs_idx)]]),
                "emission": emission,
            }
        )

    segment_groups: list[dict[str, np.ndarray]] = []
    backpointers: list[np.ndarray | None] = []
    scores = np.empty(0, dtype=float)
    previous_group: dict[str, np.ndarray] | None = None
    for group in groups:
        should_split = previous_group is None
        if previous_group is not None:
            previous_obs = int(previous_group["obs_idx"][0])
            obs_idx = int(group["obs_idx"][0])
            elapsed = float(times[obs_idx] - times[previous_obs])
            should_split = elapsed > config.gap_threshold_s
            if should_split:
                _finish_segment(segment_groups, backpointers, scores, output)
                segment_groups = []
                backpointers = []

        if should_split:
            scores = group["emission"].copy()
            backpointers.append(None)
        else:
            previous_obs = int(previous_group["obs_idx"][0])
            obs_idx = int(group["obs_idx"][0])
            gps_distance = float(np.linalg.norm(group["gps"] - previous_group["gps"]))
            transition = np.empty((len(previous_group["arc_ids"]), len(group["arc_ids"])))
            for prev_idx, previous_arc in enumerate(previous_group["arc_ids"]):
                for next_idx, next_arc in enumerate(group["arc_ids"]):
                    transition[prev_idx, next_idx] = _transition_log_probability(
                        graph,
                        int(previous_arc),
                        int(next_arc),
                        float(previous_group["offsets"][prev_idx]),
                        float(group["offsets"][next_idx]),
                        gps_distance,
                        config.transition_beta_m,
                        config.max_search_m,
                    )
            total = scores[:, None] + transition
            backpointer = np.argmax(total, axis=0).astype(np.int64)
            scores = total[backpointer, np.arange(total.shape[1])] + group["emission"]
            backpointers.append(backpointer)

        segment_groups.append(group)
        previous_group = group

    if segment_groups:
        _finish_segment(segment_groups, backpointers, scores, output)
    return output


def reconstruct_route(
    trip: dict,
    per_observation_arcs: np.ndarray,
    *,
    gap_threshold_s: float = 5.0,
    max_search_m: float = 3000.0,
) -> list[int]:
    """Collapse repeated states and fill short connected gaps with shortest paths."""
    graph = _RoadGraph(trip)
    track = trip["track"]
    times = track["t"].to_numpy(dtype=float)
    result: list[int] = []
    previous_arc: int | None = None
    previous_obs: int | None = None
    for obs_idx, value in enumerate(per_observation_arcs):
        arc_id = int(value)
        if arc_id < 0:
            continue
        if arc_id >= len(graph.arcs):
            raise ValueError(f"Invalid decoded arc ID: {arc_id}")
        if previous_arc is None:
            result.append(arc_id)
        elif arc_id != previous_arc:
            elapsed = float(times[obs_idx] - times[previous_obs])
            previous_to = int(graph.arcs[previous_arc, 1])
            next_from = int(graph.arcs[arc_id, 0])
            if elapsed <= gap_threshold_s:
                connector = graph.path(previous_to, next_from, max_search_m)
                if connector is not None:
                    result.extend(connector)
            result.append(arc_id)
        previous_arc = arc_id
        previous_obs = obs_idx

    collapsed: list[int] = []
    for arc_id in result:
        if not collapsed or collapsed[-1] != arc_id:
            collapsed.append(arc_id)
    return collapsed


def route_coordinates(trip: dict, route_arc_ids: list[int]) -> list[list[float]]:
    """Convert ordered route arc IDs to Leaflet-style [lat, lon] coordinates."""
    nodes = np.asarray(trip["nodes"], dtype=float)
    arcs = np.asarray(trip["arcs"], dtype=np.int64)
    coordinates: list[list[float]] = []
    for arc_id in route_arc_ids:
        start, end = arcs[arc_id]
        for node_id in (int(start), int(end)):
            lon, lat = nodes[node_id]
            point = [float(lat), float(lon)]
            if not coordinates or coordinates[-1] != point:
                coordinates.append(point)
    return coordinates
