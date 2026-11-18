"""Tune HMM sigma and beta on labelled train or student_test trips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_io import load_trip
from src.hmm import HMMConfig
from src.pipeline import (
    candidate_radius,
    evaluate_prepared_trip,
    load_model_artifacts,
    prepare_trip,
)


def _values(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",")]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be comma-separated positive numbers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("train", "student_test"), default="student_test")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--sigma-m", type=_values, default=_values("15,25,40"))
    parser.add_argument("--beta-m", type=_values, default=_values("20,50,100"))
    parser.add_argument("--gap-threshold-s", type=float, default=5.0)
    parser.add_argument("--max-trips", type=int, default=0, help="0 processes the whole partition")
    parser.add_argument("--output", type=Path, default=Path("results/hmm_tuning.csv"))
    parser.add_argument("--params-output", type=Path, default=Path("models/hmm_params.json"))
    args = parser.parse_args()

    trip_dirs = sorted(path for path in (args.data_root / args.partition).iterdir() if path.is_dir())
    if args.max_trips:
        trip_dirs = trip_dirs[: args.max_trips]
    if not trip_dirs:
        raise FileNotFoundError(f"No trips found in {args.data_root / args.partition}")

    root = Path(__file__).resolve().parents[1]
    model, feature_columns, _ = load_model_artifacts(
        root / "models" / "final_model.joblib",
        root / "models" / "feature_cols.json",
    )
    radius = candidate_radius()
    prepared = []
    for trip_dir in trip_dirs:
        trip = load_trip(trip_dir, include_route=True)
        if not trip["route"]:
            raise ValueError(f"{args.partition}/{trip['trip_id']} has no route for tuning")
        base_config = HMMConfig(
            emission_sigma_m=args.sigma_m[0],
            transition_beta_m=args.beta_m[0],
            gap_threshold_s=args.gap_threshold_s,
        )
        candidates = prepare_trip(
            trip,
            model,
            feature_columns,
            radius_m=radius,
            config=base_config,
        )
        prepared.append((trip, candidates))
        print(
            f"Prepared {args.partition}/{trip['trip_id']}: "
            f"{len(trip['track'])} observations, {len(candidates)} candidates",
            flush=True,
        )

    rows = []
    for sigma in args.sigma_m:
        for beta in args.beta_m:
            config = HMMConfig(
                emission_sigma_m=sigma,
                transition_beta_m=beta,
                gap_threshold_s=args.gap_threshold_s,
            )
            metrics = []
            for trip, original_candidates in prepared:
                candidates = original_candidates.copy()
                if model is None and not candidates.empty:
                    distances = candidates["dist_m"].to_numpy(dtype=float)
                    candidates["prob"] = np.exp(-0.5 * (distances / sigma) ** 2)
                result = evaluate_prepared_trip(trip, candidates, config=config)
                metrics.append(result["metrics"])
            rows.append(
                {
                    "sigma_m": sigma,
                    "beta_m": beta,
                    "top1_acc": float(np.mean([m["top1_acc"] for m in metrics])),
                    "macro_f1": float(np.mean([m["macro_f1"] for m in metrics])),
                    "route_precision": float(np.mean([m["route_precision"] for m in metrics])),
                    "route_recall": float(np.mean([m["route_recall"] for m in metrics])),
                    "continuity": float(np.mean([m["continuity"] or 0.0 for m in metrics])),
                    "n_trips": len(metrics),
                }
            )
            print(f"Evaluated sigma={sigma:g} beta={beta:g}", flush=True)

    results = pd.DataFrame(rows).sort_values(
        ["top1_acc", "macro_f1"], ascending=False, kind="stable"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    best = results.iloc[0]
    params = {
        "emission_sigma_m": float(best["sigma_m"]),
        "transition_beta_m": float(best["beta_m"]),
        "gap_threshold_s": args.gap_threshold_s,
        "model_weight": 0.75,
        "max_search_m": 3000.0,
        "tuned_on": args.partition,
        "n_trips": int(best["n_trips"]),
    }
    args.params_output.parent.mkdir(parents=True, exist_ok=True)
    args.params_output.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    print(f"\nSelected HMM parameters: {json.dumps(params)}")
    print(f"Saved grid results to {args.output}")


if __name__ == "__main__":
    main()
