import unittest

import numpy as np
import pandas as pd

from src.hmm import HMMConfig, reconstruct_route, viterbi


def small_trip() -> dict:
    return {
        "trip_id": "tiny",
        "track": pd.DataFrame(
            {
                "lon": [0.0005, 0.0015],
                "lat": [0.0, 0.0],
                "t": [1.0, 2.0],
                "obs_idx": [0, 1],
            }
        ),
        "nodes": np.asarray(
            [[0.0, 0.0], [0.001, 0.0], [0.002, 0.0]],
            dtype=float,
        ),
        "arcs": np.asarray([[0, 1], [1, 2], [1, 0], [2, 1]], dtype=np.int64),
        "route": [0, 1],
    }


class ViterbiTests(unittest.TestCase):
    def test_prefers_high_score_connected_route(self) -> None:
        candidates = pd.DataFrame(
            {
                "obs_idx": [0, 0, 1, 1],
                "arc_id": [0, 2, 1, 3],
                "dist_m": [0.0, 0.0, 0.0, 0.0],
                "offset_fraction": [0.5, 0.5, 0.5, 0.5],
                "prob": [0.99, 0.01, 0.99, 0.01],
            }
        )

        decoded = viterbi(small_trip(), candidates)

        np.testing.assert_array_equal(decoded, [0, 1])

    def test_returns_minus_one_for_observations_without_candidates(self) -> None:
        trip = small_trip()
        decoded = viterbi(trip, pd.DataFrame(columns=["obs_idx", "arc_id", "dist_m"]))

        np.testing.assert_array_equal(decoded, [-1, -1])

    def test_route_reconstruction_collapses_repeated_arcs(self) -> None:
        trip = small_trip()

        route = reconstruct_route(trip, np.asarray([0, 1]))

        self.assertEqual(route, [0, 1])

    def test_large_time_gap_starts_a_new_chain(self) -> None:
        trip = small_trip()
        trip["track"].loc[1, "t"] = 20.0
        candidates = pd.DataFrame(
            {
                "obs_idx": [0, 1],
                "arc_id": [0, 3],
                "dist_m": [0.0, 0.0],
                "prob": [0.1, 0.9],
            }
        )

        decoded = viterbi(
            trip,
            candidates,
            HMMConfig(gap_threshold_s=5.0),
        )

        np.testing.assert_array_equal(decoded, [0, 3])


if __name__ == "__main__":
    unittest.main()
