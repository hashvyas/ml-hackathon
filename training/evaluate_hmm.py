"""Run the prediction pipeline on every trip in a labelled evaluation partition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.hmm import HMMConfig
from src.pipeline import (
    candidate_radius,
    load_model_artifacts,
    predict_trip,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("student_test",), default="student_test")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/hmm_student_test.csv"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model, feature_columns, predictor = load_model_artifacts(
        root / "models" / "final_model.joblib",
        root / "models" / "feature_cols.json",
    )
    params_path = root / "models" / "hmm_params.json"
    params = {}
    if params_path.is_file():
        params = json.loads(params_path.read_text(encoding="utf-8"))
    config = HMMConfig(
        emission_sigma_m=float(params.get("emission_sigma_m", 25.0)),
        transition_beta_m=float(params.get("transition_beta_m", 50.0)),
        gap_threshold_s=float(params.get("gap_threshold_s", 5.0)),
        model_weight=float(params.get("model_weight", 0.75)),
        max_search_m=float(params.get("max_search_m", 3000.0)),
    )

    trip_dirs = sorted(
        path for path in (args.data_root / args.partition).iterdir() if path.is_dir()
    )
    if not trip_dirs:
        raise FileNotFoundError(f"No trips found in {args.data_root / args.partition}")
    radius = candidate_radius()
    rows = []
    for trip_dir in trip_dirs:
        result = predict_trip(
            trip_dir,
            model=model,
            feature_columns=feature_columns,
            radius_m=radius,
            config=config,
        )
        rows.append({"trip_id": result["trip_id"], **result["metrics"]})
        print(
            f"{result['trip_id']}: top1={result['metrics']['top1_acc']} "
            f"route_f1={result['metrics']['macro_f1']} "
            f"time_ms={result['metrics']['time_ms']}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Evaluated {len(rows)} trips using {predictor}; results saved to {args.output}")


if __name__ == "__main__":
    main()
