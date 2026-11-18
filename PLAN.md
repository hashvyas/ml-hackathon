# PathSnap Hackathon: Team Plan (3 people, 1 day)

## Context
PES UE24CS352A hackathon "PathSnap". The task is to map-match noisy 1 Hz GPS tracks to a road network. Per GPS point, a classifier scores candidate road segments (arcs). HMM/Viterbi then turns the scores into a connected route. The demo runs on two laptops: Laptop 1 runs FastAPI with the model, HMM and server-side ground truth; Laptop 2 runs the UI, which lists the 10 evaluator trips, sends the chosen trip ID, and shows the route and accuracy.

What has to be submitted: a public GitHub repo (code for every stage, model weights, README) and one PDF report (sections 2.2 to 7 of Deliverables.pdf). There is a phase-1 check before lunch, and its contents aren't known yet.

### What the data looks like
`data/{train (80), student_test (10)}/<id>/<id>.{track,nodes,arcs,route}` (hackathon PDFs are in `docs/`)
- `.track`: lon, lat, t (s), about 1 Hz, with gaps. Example: trip 1 has 1315 points.
- `.nodes`: lon, lat. The row index is the node_id. Each trip has its own local network (88k nodes in trip 1).
- `.arcs`: from_node, to_node. The row index is the arc_id. Arcs are directed: both 0→1 and 1→0 appear.
- `.route`: the ground-truth arc_ids in order. It has **no mapping from GPS point to arc**. For trip 1 that's 525 arcs for 1315 points, so point labels have to be derived.
- Train trips are around lon 8.58 / lat 49.86 (Germany); student_test trip 0 is around lon 54 / lat 56. **Regions differ**, so distances must be in metres from a per-trip projection (UTM via pyproj), never in degrees.
- The 10 evaluator trips are **not in the folder**. We assume they arrive at the event in the same folder format, so the API reads them from a configurable `EVAL_DIR`.
- Rules: no hardcoded outputs, labels never reach the UI, show API logs on Laptop 1, and every member must be able to explain all of it in the viva.

## Data rules (everyone)
- **Raw data is read-only.** No code writes into `data/train/` or `data/student_test/`. Anything generated (labels, candidates, features, predictions) goes to `data/processed/` or `results/`.
- **Fit only on train.** Models, scalers, encoders and Bayesian opt use the 80 train trips with GroupKFold. student_test is loaded only inside the final evaluation function in `train_models.py`, so nothing can be fit on it by accident.
- **student_test labels are for scoring only.** They are derived from `.route` the same way as train labels and never go into training.
- **HMM σ/β** may be tuned on train + student_test (the guidelines allow this).
- **Demo trips:** the docs describe 10 reserved evaluator trips that are separate from student_test, but the team understands the demo may use student_test. **Confirm with the organisers.** `EVAL_DIR` stays configurable either way. If the demo uses student_test, say in the report that the demo accuracy is optimistic because those trips were also used for model comparison.

## Repo layout (agree on this first, 10 min)
```
ml-hackathon/   (repo root)
  data/train, data/student_test   raw trips (92 MB, committed, read-only)
  data/processed/  generated files (gitignored)
  docs/            hackathon PDFs
  PLAN.md          this plan
  src/
    data_io.py      load_trip(dir) -> {track_df, nodes(np), arcs(np), route|None}   [A]
    geo.py          per-trip UTM projection, point-to-segment distance            [A]
    labels.py       per-GPS-point true arc from .route                           [ME]
    features.py     candidates + features per GPS point                          [ME]
    hmm.py          viterbi(cands_with_prob, trip, params) -> arc per obs        [B]
    pipeline.py     predict_trip(dir) -> obs_arcs, route geometry, metrics       [B]
    maps.py         folium map (GPS, network, reference, predicted, layers)     [A]
  notebooks/ 01_eda.ipynb [A]   02_models.ipynb [ME]   03_hmm_eval.ipynb [B]
  training/  train_models.py, cv.py, bayes_opt.py (Optuna)                       [ME]
  models/    final_model.joblib, feature_cols.json, hmm_params.json
  results/   csv logs, plots, optuna trials, folium html
  api/       main.py (FastAPI)                                                    [B]
  ui/        index.html, app.js, style.css (Leaflet)                              [B]
  requirements.txt, README.md                                                     [B]
  report/    report.pdf                                                           [all; B assembles]
```

## Interface contracts (so nobody blocks on anyone)
1. `load_trip(path)` returns `track` as a DataFrame[lon, lat, t], `nodes` as an (N,2) array, `arcs` as an (M,2) array, and `route` as a list or None. A writes this in the **first 30 min** and pushes it.
2. The candidate table (from ME) has one row per (obs_idx, arc_id) with `dist_m` and the feature columns, plus `label` (0/1) when a route exists. Label = the arc this point was on.
3. Model artifact (from ME): `models/final_model.joblib` plus `feature_cols.json`; `predict_proba` gives P(candidate is true arc).
4. HMM input (B) is the candidate table plus a `prob` column. Output is one arc_id per obs, or -1 if the point has no candidate. Until the model is ready, B can test with `prob = exp(-dist²/2σ²)`.
5. API response (B): `{trip_id, gps:[[lat,lon]], predicted_route:[[lat,lon]...], per_obs_arcs, metrics:{top1_acc, macro_f1, route_precision, route_recall, continuity, n_obs, time_ms}}`. Ground truth only ever goes into `metrics`.

---

## ME (you): training and testing, including features and Bayesian opt
1. **Labels** (`labels.py`): give each GPS point the route arc it lies on. Walk the route in order and assign each point the nearest arc among the next few route arcs (monotonic, so a point can't jump backwards). Report the label mismatch rate as a "reference route limitation".
2. **Candidates and features** (`features.py`): per point, take the arcs within R m (e.g. 30–50 m, via a KD-tree on arc midpoints/segments). Measure the **candidate miss rate**, i.e. how often the true arc isn't in the set, and tune R with it. Features: perpendicular distance, rank by distance, heading difference between GPS motion and arc bearing, speed, arc length, node degree, candidate count, the distance gap to the nearest candidate, and offset along the arc. If you add a "connects to previous road" feature, compute it against **any of the previous point's candidates**, never the true previous arc, which leaks the label. Otherwise leave connectivity to the HMM.
3. **5+ models** from different families: LogisticRegression, GaussianNB, KNN, RandomForest/ExtraTrees, HistGB/LightGBM/XGBoost, MLP (loss curve allowed), and optionally an SVM on a subsample. **GroupKFold by trip** on the 80 train trips.
4. **Metrics per model**: CV macro F1, student_test macro F1, log loss, observation-level Top-1 accuracy (argmax per point), training/inference time. Save confusion matrices. Write one CSV row per model to `results/model_comparison.csv`, with seeds fixed and configs logged.
5. **Bayesian opt** (Optuna TPE, GroupKFold, objective = CV macro F1 or Top-1) on the best model: about 50 trials, `optuna_trials.csv`, then refit and compare tuned vs untuned on the same metrics.
6. Export `final_model.joblib` and `feature_cols.json`, and give B the probabilities for student_test.
7. Write report §3 (per-model questions A–C, comparison table, plots) and §4.

## Teammate A: data, EDA, maps
1. **H0 (first 30 min):** `data_io.py` and `geo.py`, pushed so ME and B can start. `load_trip` also **cleans** the track (pipeline step 2): sort by timestamp, drop duplicate timestamps and NaN/out-of-range coordinates, and flag time gaps. Every cleaning rule is logged so EDA can report it.
2. **EDA** (`01_eda.ipynb`, run on train and student_test, labelled by partition): trip inventory, point counts, file sizes, dtypes, missing/invalid values, timestamp gaps (the share of 1 s steps, big gaps), geographic extents, network size, connectivity (networkx: components, isolated nodes), route structure (is it a connected arc chain?), anomalies.
3. **`maps.py`**: one folium map function with layers for GPS points, the nearby road network, the reference route and the predicted route, plus a legend and LayerControl. EDA uses it now; ME and B reuse it later.
4. Write report §2.2 (EDA) and §6 (folium screenshots).

## Teammate B: HMM/Viterbi, route reconstruction, API, UI, deployment, report assembly
1. **H0:** create the GitHub repo and the layout above, `.gitignore`, `requirements.txt`, and invite everyone.
2. **HMM/Viterbi** (`hmm.py`): hidden states = candidate arcs per point. Emission = classifier prob (blended with Gaussian distance). Transition = exp(-|route distance − GPS distance|/β), with route distance from Dijkstra on the arc graph (bounded search). Unreachable pairs get a tiny probability. A point with no candidates is skipped and bridged. A large time gap means the chain is split or β is relaxed. Use log-space Viterbi.
3. **Route reconstruction** (`hmm.reconstruct_route`, pipeline step 6): take the per-point arcs, merge consecutive repeats, and fill each break between non-adjacent arcs with the shortest path, giving one ordered arc sequence. Convert it to lat/lon for the map. Route precision/recall are computed on this sequence against `.route`.
4. **Decoder tuning** (σ, β, gap threshold) on train/student_test only. Compare **classifier-only vs classifier+HMM**: Top-1, route precision/recall, continuity (the share of consecutive predicted arcs that are connected or equal). Include failure cases.
5. Write report §5 (HMM and Viterbi), using A's `maps.py` for the classifier-only vs HMM maps.
6. **FastAPI** (`api/main.py`): `GET /trips` returns the trip IDs from `EVAL_DIR` (and student_test for dev). `POST /predict {trip_id}` runs `pipeline.predict_trip`, computes metrics against `.route` server-side, and logs every request to the console. The evaluator watches this log. Enable CORS and run with `uvicorn --host 0.0.0.0`. Load the model once at startup.
7. **`pipeline.py`** chains `load_trip → features → model.predict_proba → hmm.viterbi → metrics`, built from ME's, A's and B's own functions. B can start with a dummy nearest-arc predictor so the API and UI work before the model exists.
8. **UI** (`ui/`, plain HTML + JS + Leaflet): an input for the API base URL (Laptop 1's LAN IP), the trip list from `/trips`, and a button that POSTs. The map shows raw GPS (blue dots) and the predicted route (green line), with a metrics panel (Top-1 acc, macro F1, route precision/recall, continuity, inference time) and the list of predicted segments. **No ground truth in the UI: never draw the reference route.** Only the server computes metrics from it. Reference-route maps belong in the report's folium maps only. Serve it with `python -m http.server` on Laptop 2.
9. **Two-laptop test:** same Wi-Fi/hotspot, allow port 8000 through Windows Firewall, run all 10 student_test trips end to end, and time each one (the first request may be slow because of Dijkstra). Hotspot fallback if the venue Wi-Fi blocks clients from seeing each other.
10. Write the README (install, data placement and the "raw data read-only, generated files go to data/processed/" rule, training commands, running API/UI, two-laptop steps) and report §7 (UI screenshots, API log screenshots). **Assemble the full PDF.**

---

## Timeline (H = hours from start; lunch around H4)
| | ME | A | B |
|---|---|---|---|
| H0–0.5 | read data, plan features | data_io + geo, push | repo, layout, requirements |
| H0.5–2 | labels + candidates, miss rate | EDA notebook + folium | FastAPI skeleton + UI skeleton with dummy predictor |
| H2–4 | features + 5 models + CV | network connectivity, folium maps for the report | HMM v1 (distance emission), API ↔ UI over LAN |
| **Phase-1 check** | model comparison table | EDA + maps | live demo with dummy |
| H4–6 | Bayesian opt, export model | data-quality write-up, help B test trips | HMM with real probs, tune, comparison, wire pipeline, test all 10 trips |
| H6–7.5 | report §3–4 | report §2.2, §6 | report §5, README, screenshots, §7, assemble PDF |
| H7.5–8 | **everyone:** run the full demo once, make the repo public, each member rehearses explaining all parts | | |

## Risks
- **Point labels are derived**, so every metric depends on `labels.py`. ME shares it early and A and B reuse it as-is.
- **Dijkstra on 88k-node networks** is slow. Bound the search radius and cache per trip; B times it in the API.
- **Regions differ** between train and test, so no raw lat/lon features (they would leak location). Use only relative and geometric features.
- **Evaluator trip format unknown**: keep `EVAL_DIR` configurable and confirm the format with the organisers early.

## Verification
- `python training/train_models.py` regenerates `results/model_comparison.csv` and the plots.
- `notebooks/03_hmm_eval.ipynb` shows classifier-only vs HMM on student_test.
- Laptop 1 runs `uvicorn api.main:app --host 0.0.0.0 --port 8000`. On Laptop 2, open the UI and run all 10 student_test trips: each shows a route and metrics, and the request appears in the Laptop 1 log.
- grep the repo for hardcoded trip IDs or route arrays in the api/ and ui/ folders.
