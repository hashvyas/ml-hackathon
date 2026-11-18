"""Static map views for the PDF (same layers as the Folium maps, drawn with matplotlib in UTM metres).

python report/make_figures.py
Writes report/figures/network_train_00000001.png and one PNG per trip listed in results/hmm_failures.csv /
the two HMM comparison maps.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_io import load_trip
from src.geo import TripProjection
from src.hmm import HMMConfig, reconstruct_route, viterbi
from src.pipeline import candidate_radius, load_model_artifacts, prepare_trip
from training.evaluate_hmm import classifier_only

OUT = Path("report/figures")


def draw(trip, title, out, routes=()):
    tr = trip["track"]
    proj = TripProjection.from_coordinates(tr.lon.values, tr.lat.values)
    gx, gy = proj.project(tr.lon.values, tr.lat.values)
    nx_, ny_ = proj.project(trip["nodes"][:, 0], trip["nodes"][:, 1])
    x0, y0 = gx.min(), gy.min()  # plot in km from the track's south-west corner
    gx, gy, nx_, ny_ = (gx - x0) / 1000, (gy - y0) / 1000, (nx_ - x0) / 1000, (ny_ - y0) / 1000
    m = 0.15
    inside = (nx_ > gx.min() - m) & (nx_ < gx.max() + m) & (ny_ > gy.min() - m) & (ny_ < gy.max() + m)
    arcs = trip["arcs"]
    near = arcs[inside[arcs[:, 0]] & inside[arcs[:, 1]]]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(np.c_[nx_[near[:, 0]], nx_[near[:, 1]]].T, np.c_[ny_[near[:, 0]], ny_[near[:, 1]]].T, color="#bbbbbb", lw=0.5)
    for arc_ids, color, width, label in [(trip["route"], "green", 5, "reference route")] + list(routes):
        a = arcs[np.asarray([x for x in arc_ids if x >= 0])]
        ax.plot(np.c_[nx_[a[:, 0]], nx_[a[:, 1]]].T, np.c_[ny_[a[:, 0]], ny_[a[:, 1]]].T, color=color, lw=width, alpha=0.7)
        ax.plot([], [], color=color, lw=width, label=label)
    ax.scatter(gx, gy, s=1, color="blue", alpha=0.35, label="GPS points", zorder=5)
    ax.plot([], [], color="#bbbbbb", label="road network")
    ax.set_aspect("equal"); ax.set_xlabel("x (km, UTM)"); ax.set_ylabel("y (km, UTM)")
    ax.set_title(title); ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print("wrote", out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    draw(load_trip("data/train/00000001"), "train 00000001: GPS, road network, reference route",
         OUT / "network_train_00000001.png")

    model, cols, _ = load_model_artifacts("models/final_model.joblib", "models/feature_cols.json")
    p = json.loads(Path("models/hmm_params.json").read_text())
    config = HMMConfig(**{k: p[k] for k in ("emission_sigma_m", "transition_beta_m", "gap_threshold_s", "model_weight", "max_search_m")})
    for html in sorted(Path("results/maps").glob("student_test_*_hmm_*.html")):
        tid, tag = html.stem.split("_")[2], html.stem.split("_", 3)[3]
        trip = load_trip(f"data/student_test/{tid}")
        cands = prepare_trip(trip, model, cols, radius_m=candidate_radius(), config=config)
        routes = []
        for arcs, color, label in [(classifier_only(trip, cands), "orange", "classifier-only route"),
                                   (viterbi(trip, cands, config), "red", "classifier + HMM route")]:
            routes.append((reconstruct_route(trip, arcs, gap_threshold_s=config.gap_threshold_s), color, 2, label))
        draw(trip, f"student_test {tid} ({tag.replace('_', ' ')})", OUT / f"{html.stem}.png", routes)


if __name__ == "__main__":
    main()
