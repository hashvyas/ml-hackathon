"""Folium map of one trip: GPS points, nearby road network, reference and predicted routes."""

from pathlib import Path

import folium
import numpy as np

from .geo import TripProjection

MARGIN_M = 150.0  # how far beyond the track bounding box we still draw roads


def _arc_segments(trip, arc_ids):
    """List of [[lat, lon], [lat, lon]] segments for the given arc ids (data stores lon, lat)."""
    nodes, arcs = trip["nodes"], trip["arcs"]
    return [[[nodes[a, 1], nodes[a, 0]], [nodes[b, 1], nodes[b, 0]]] for a, b in arcs[list(arc_ids)]]


def make_map(trip, predicted_arcs=None, out_path=None, classifier_arcs=None, predicted_name="Predicted route"):
    track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    lon, lat = track["lon"].to_numpy(), track["lat"].to_numpy()

    # Nearby network: arcs with both end nodes inside the track bounding box + margin, measured in metres.
    proj = TripProjection.from_coordinates(lon, lat)
    gx, gy = proj.project(lon, lat)
    nx_, ny_ = proj.project(nodes[:, 0], nodes[:, 1])
    inside = ((nx_ >= gx.min() - MARGIN_M) & (nx_ <= gx.max() + MARGIN_M)
              & (ny_ >= gy.min() - MARGIN_M) & (ny_ <= gy.max() + MARGIN_M))
    keep = inside[arcs[:, 0]] & inside[arcs[:, 1]]
    nearby = np.flatnonzero(keep)
    # draw each two-way road once: a->b and b->a share the unordered pair {a, b}
    _, first = np.unique(np.sort(arcs[nearby], axis=1), axis=0, return_index=True)
    nearby = nearby[first]

    m = folium.Map(location=[lat.mean(), lon.mean()])
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    roads = folium.FeatureGroup(name=f"Road network ({len(nearby)} arcs)")
    folium.PolyLine(_arc_segments(trip, nearby), color="#888888", weight=1, opacity=0.6).add_to(roads)
    roads.add_to(m)

    if trip.get("route"):
        ref = folium.FeatureGroup(name="Reference route")
        folium.PolyLine(_arc_segments(trip, trip["route"]), color="green", weight=5, opacity=0.7).add_to(ref)
        ref.add_to(m)

    if classifier_arcs is not None:
        clf = folium.FeatureGroup(name="Classifier-only route")
        folium.PolyLine(_arc_segments(trip, [a for a in classifier_arcs if a >= 0]),
                        color="orange", weight=3, opacity=0.9).add_to(clf)
        clf.add_to(m)

    if predicted_arcs is not None:
        pred_ids = [a for a in predicted_arcs if a >= 0]  # -1 means "no candidate"
        pred = folium.FeatureGroup(name=predicted_name)
        folium.PolyLine(_arc_segments(trip, pred_ids), color="red", weight=3, opacity=0.9).add_to(pred)
        pred.add_to(m)

    gps = folium.FeatureGroup(name=f"GPS points ({len(track)})")
    for la, lo, t in zip(lat, lon, track["t"]):
        folium.CircleMarker([la, lo], radius=2, color="blue", fill=True, weight=1, tooltip=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)

    folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = ("<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;"
              "border:1px solid #999;font-size:13px'>"
              f"<b>Trip {trip['trip_id']}</b><br>"
              "<span style='color:blue'>&#9679;</span> GPS points<br>"
              "<span style='color:#888'>&#9473;</span> Road network<br>"
              "<span style='color:green'>&#9473;</span> Reference route<br>"
              + ("<span style='color:orange'>&#9473;</span> Classifier-only route<br>" if classifier_arcs is not None else "")
              + f"<span style='color:red'>&#9473;</span> {predicted_name}</div>")
    m.get_root().html.add_child(folium.Element(legend))

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_path))
    return m


if __name__ == "__main__":
    from .data_io import load_trip

    for partition, trip_id in [("train", "00000001"), ("student_test", "00000007")]:
        trip = load_trip(Path("data") / partition / trip_id)
        out = Path("results/maps") / f"{partition}_{trip_id}.html"
        make_map(trip, out_path=out)
        html = out.read_text(encoding="utf-8")
        assert "Reference route" in html and "leaflet" in html.lower()
        print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
