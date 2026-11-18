"""Check a running API: /health, /trips, /predict for every trip, with timing.

Start the API first:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Then run:             python tests/api_smoke.py [http://127.0.0.1:8000]
Writes results/api_timing.csv. Not collected by pytest because it needs a live server.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
CONTRACT = {"trip_id", "gps", "predicted_route", "per_obs_arcs", "metrics", "predictor"}
METRICS = {"top1_acc", "macro_f1", "route_precision", "route_recall", "continuity", "n_obs", "time_ms"}


def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json",
                                                                  "Origin": "http://laptop2:5500"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read()), resp.headers


health, _ = call("/health")
print("GET /health ->", health)
trips, headers = call("/trips")
ids = [t["trip_id"] for t in trips["trips"]]
print(f"GET /trips -> {len(ids)} trips; CORS header: {headers.get('access-control-allow-origin')}")

rows = ["order,trip_id,n_obs,client_wall_s,server_time_ms,top1_acc,macro_f1,route_precision,route_recall,continuity"]
for order, trip_id in enumerate(ids, 1):
    t0 = time.perf_counter()
    result, _ = call("/predict", {"trip_id": trip_id})
    wall = time.perf_counter() - t0
    assert set(result) == CONTRACT, f"unexpected keys {set(result) ^ CONTRACT}"
    assert set(result["metrics"]) == METRICS, f"unexpected metric keys {set(result['metrics']) ^ METRICS}"
    assert len(result["per_obs_arcs"]) == len(result["gps"]) == result["metrics"]["n_obs"]
    m = result["metrics"]
    rows.append(f"{order},{trip_id},{m['n_obs']},{wall:.2f},{m['time_ms']},{m['top1_acc']:.4f},{m['macro_f1']:.4f},"
                f"{m['route_precision']:.4f},{m['route_recall']:.4f},{m['continuity']:.4f}")
    print(f"POST /predict {trip_id}: {wall:.2f} s, top1={m['top1_acc']:.4f}")

Path("results").mkdir(exist_ok=True)
Path("results/api_timing.csv").write_text("\n".join(rows) + "\n")
print("contract OK for all trips; saved results/api_timing.csv")
