"""Generate processed candidate tables for train or student_test trips."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_io import load_trip
from src.features import build_candidate_features
from src.geo import TripProjection
from src.labels import derive_point_labels


def process_partition(data_root: Path, partition: str, output_root: Path, radius_m: float) -> None:
    source = data_root / partition
    destination = output_root / partition
    destination.mkdir(parents=True, exist_ok=True)
    for trip_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        trip = load_trip(trip_dir)
        projection = TripProjection.from_coordinates(
            trip["track"]["lon"].to_numpy(), trip["track"]["lat"].to_numpy()
        )
        labels = None
        if trip["route"] is not None:
            labels = derive_point_labels(
                trip["track"], trip["nodes"], trip["arcs"], trip["route"], projection
            )
        candidates = build_candidate_features(
            trip, radius_m=radius_m, projection=projection, labels=labels
        )
        output_path = destination / f"{trip['trip_id']}_candidates.csv"
        candidates.to_csv(output_path, index=False)
        print(
            f"{partition}/{trip['trip_id']}: observations={len(trip['track'])} "
            f"candidates={len(candidates)} miss_rate={candidates.attrs['candidate_miss_rate']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--partition", choices=("train", "student_test", "all"), default="all")
    parser.add_argument("--radius-m", type=float, default=50.0)
    args = parser.parse_args()
    partitions = ("train", "student_test") if args.partition == "all" else (args.partition,)
    for partition in partitions:
        process_partition(args.data_root, partition, args.output_root, args.radius_m)


if __name__ == "__main__":
    main()
