"""Fit the selected classifier on train candidate tables and export it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.pipeline import FEATURE_COLUMNS, candidate_radius


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=Path("data/processed/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--max-rows-per-trip", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_rows_per_trip <= 0:
        parser.error("--max-rows-per-trip must be positive")

    files = sorted(args.candidate_dir.glob("*_candidates.csv"))
    if not files:
        raise FileNotFoundError(
            f"No train candidate tables in {args.candidate_dir}. "
            "Run training/preprocess_dataset.py --partition train first."
        )

    radius_m = candidate_radius()
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        if "label" not in frame or any(column not in frame for column in FEATURE_COLUMNS):
            raise ValueError(f"{path} does not match the candidate feature contract")
        if len(frame) > args.max_rows_per_trip:
            frame = frame.sample(n=args.max_rows_per_trip, random_state=args.seed)
        frame = frame[FEATURE_COLUMNS + ["label"]].copy()
        frame["distance_gap_m"] = frame["distance_gap_m"].fillna(radius_m)
        frames.append(frame)
        print(f"Loaded {path.name}: {len(frame)} training rows", flush=True)

    training = pd.concat(frames, ignore_index=True)
    if training["label"].nunique() != 2:
        raise ValueError("Training candidates must contain both label classes")
    model = HistGradientBoostingClassifier(random_state=args.seed)
    model.fit(training[FEATURE_COLUMNS], training["label"].astype("int8"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "final_model.joblib"
    columns_path = args.output_dir / "feature_cols.json"
    joblib.dump(model, model_path)
    columns_path.write_text(json.dumps(FEATURE_COLUMNS, indent=2) + "\n", encoding="utf-8")
    print(
        f"Exported {model_path} from {len(training)} rows across {len(files)} train trips; "
        f"feature schema: {columns_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
