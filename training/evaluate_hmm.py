"""Compare classifier-only with classifier + HMM on the 10 student_test trips.

Classifier-only: for each GPS point take the candidate with the highest probability.
Classifier + HMM: Viterbi over the same candidates and probabilities.
Both go through the same route reconstruction and the same metrics (src/pipeline.compute_metrics).
HMM parameters come from models/hmm_params.json (tuned on train trips by training/tune_hmm.py).
student_test is used here for the final comparison only.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_io import load_trip
from src.hmm import HMMConfig, reconstruct_route, viterbi
from src.labels import derive_point_labels
from src.maps import make_map
from src.pipeline import candidate_radius, compute_metrics, load_model_artifacts, prepare_trip

RESULTS = Path("results")


def classifier_only(trip, candidates):
    """Argmax-probability candidate per observation; -1 when a point has no candidate."""
    arcs = np.full(len(trip["track"]), -1, dtype=np.int64)
    best = candidates.sort_values("prob", ascending=False).drop_duplicates("obs_idx")
    arcs[best["obs_idx"].to_numpy()] = best["arc_id"].to_numpy()
    return arcs


def likely_cause(facts):
    """One-line cause from measured trip facts (checked in order of how much they limit any decoder)."""
    if facts["hmm_only_errors"] and facts["reverse_twin_share"] > 0.5:
        return (f"{facts['reverse_twin_share']:.0%} of the {facts['hmm_only_errors']} points the HMM gets wrong (classifier right) "
                f"are the reverse-direction twin arc: U-turns are not penalised")
    if facts["miss_rate"] > 0.02:
        return f"true arc not among candidates for {facts['miss_rate']:.1%} of points"
    if facts["gaps_over_60s"] > 0:
        return f"{facts['gaps_over_60s']} GPS gaps over 60 s split the HMM chain"
    if facts["label_mismatch"] > 0.10:
        return f"reference labels uncertain: {facts['label_mismatch']:.1%} label mismatch"
    return f"classifier already strong (Top-1 {facts['clf_top1']:.3f}); HMM smoothing moved some correct points"


def main():
    model, feature_columns, predictor = load_model_artifacts("models/final_model.joblib", "models/feature_cols.json")
    params = json.loads(Path("models/hmm_params.json").read_text())
    config = HMMConfig(**{k: params[k] for k in ("emission_sigma_m", "transition_beta_m", "gap_threshold_s",
                                                  "model_weight", "max_search_m")})
    print(f"predictor={predictor}, HMM params={params}")
    radius = candidate_radius()

    rows, facts, cache = [], {}, {}
    for trip_dir in sorted(Path("data/student_test").iterdir()):
        trip = load_trip(trip_dir)
        t0 = time.perf_counter()
        candidates = prepare_trip(trip, model, feature_columns, radius_m=radius, config=config)
        prep_s = time.perf_counter() - t0

        outputs = {}
        for method in ("classifier_only", "classifier_hmm"):
            t0 = time.perf_counter()
            arcs = classifier_only(trip, candidates) if method == "classifier_only" else viterbi(trip, candidates, config)
            route = reconstruct_route(trip, arcs, gap_threshold_s=config.gap_threshold_s, max_search_m=config.max_search_m)
            decode_s = time.perf_counter() - t0
            metrics = compute_metrics(trip, arcs, route, gap_threshold_s=config.gap_threshold_s)
            rows.append({"trip_id": trip["trip_id"], "method": method, **metrics, "time_s": prep_s + decode_s})
            outputs[method] = route

        labels = derive_point_labels(trip["track"], trip["nodes"], trip["arcs"], trip["route"])
        pairs = set(zip(candidates["obs_idx"], candidates["arc_id"]))
        facts[trip["trip_id"]] = {
            "miss_rate": 1 - np.mean([(o, a) in pairs for o, a in zip(labels["obs_idx"], labels["true_arc_id"])]),
            "gaps_over_60s": int((trip["track"]["t"].diff() > 60).sum()),
            "label_mismatch": float((labels["true_arc_id"] != labels["nearest_route_arc"]).mean()),
            "clf_top1": rows[-2]["top1_acc"],
        }
        clf_arcs, hmm_arcs, truth = classifier_only(trip, candidates), viterbi(trip, candidates, config), labels["true_arc_id"].to_numpy()
        hmm_only = np.flatnonzero((hmm_arcs != truth) & (clf_arcs == truth) & (hmm_arcs >= 0))
        arcs = trip["arcs"]
        facts[trip["trip_id"]]["hmm_only_errors"] = len(hmm_only)
        facts[trip["trip_id"]]["reverse_twin_share"] = float(np.mean([(arcs[hmm_arcs[i]] == arcs[truth[i]][::-1]).all()
                                                                      for i in hmm_only])) if len(hmm_only) else 0.0
        cache[trip["trip_id"]] = (trip, outputs)
        print(f"{trip['trip_id']}: classifier top1={rows[-2]['top1_acc']:.4f}  hmm top1={rows[-1]['top1_acc']:.4f}", flush=True)

    table = pd.DataFrame(rows)
    means = table.groupby("method", as_index=False).mean(numeric_only=True).assign(trip_id="MEAN")
    table = pd.concat([table, means], ignore_index=True)
    table.to_csv(RESULTS / "hmm_comparison.csv", index=False)

    wide = table[table.trip_id != "MEAN"].pivot(index="trip_id", columns="method", values="top1_acc")
    wide["hmm_gain"] = wide["classifier_hmm"] - wide["classifier_only"]
    failures = wide[wide["hmm_gain"] <= 0].reset_index()
    failures["likely_cause"] = [likely_cause(facts[t]) for t in failures["trip_id"]]
    failures = failures.assign(**{k: [facts[t][k] for t in failures["trip_id"]] for k in ("miss_rate", "gaps_over_60s", "label_mismatch", "hmm_only_errors", "reverse_twin_share")})
    failures.to_csv(RESULTS / "hmm_failures.csv", index=False)

    helps, hurts = wide["hmm_gain"].idxmax(), wide["hmm_gain"].idxmin()
    for tag, tid in (("hmm_helps", helps), ("hmm_not_helping", hurts)):
        trip, outputs = cache[tid]
        out = RESULTS / "maps" / f"student_test_{tid}_{tag}.html"
        make_map(trip, predicted_arcs=outputs["classifier_hmm"], classifier_arcs=outputs["classifier_only"],
                 predicted_name="Classifier + HMM route", out_path=out)
        print(f"map {out}: Top-1 gain {wide.loc[tid, 'hmm_gain']:+.4f}")

    pd.set_option("display.width", 250)
    print("\nstudent_test (final comparison; HMM parameters tuned on train trips):")
    print(table[table.trip_id == "MEAN"].drop(columns="trip_id").round(4).to_string(index=False))
    print("\nTrips where the HMM does not improve Top-1:")
    print(failures.round(4).to_string(index=False) if len(failures) else "none")


if __name__ == "__main__":
    main()
