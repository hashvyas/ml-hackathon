# PathSnap

PathSnap maps noisy GPS observations to a directed road network. The prediction service finds nearby road-segment candidates, scores them with the exported classifier (or an explicit distance-only baseline), decodes a connected sequence with a log-space HMM/Viterbi pass, and reconstructs a route. The browser UI connects to the API over a LAN.

Team: V Vyas, V Kushal, Vaishnavi B (PES UE24CS352A hackathon). Report: [`report/report.pdf`](report/report.pdf).

## Repository structure

| Component | Where |
|---|---|
| Data processing (read, inspect, clean) | `src/data_io.py` (load and clean a trip), `src/geo.py` (per-trip UTM projection), `notebooks/01_eda.ipynb` (EDA), `training/preprocess_dataset.py` (build candidate tables) |
| Labels | `src/labels.py` (point labels derived from `.route`) |
| Feature engineering | `src/features.py` (candidates within R m and their features; `python -m src.features` runs the R sweep) |
| Model training | `training/train_models.py` (five models, GroupKFold by trip), `training/train_scores.py` (train-set scores) |
| Model evaluation | `training/train_models.py` (metrics, comparison plots, confusion matrices) -> `results/model_comparison.*`, `results/confusion_*.png` |
| Hyperparameter optimisation | `training/bayes_opt.py` (Optuna TPE), config and results in `results/optuna_*`, `results/tuned_vs_untuned.*` |
| HMM and Viterbi | `src/hmm.py` (Viterbi, route reconstruction), `training/tune_hmm.py`, `training/evaluate_hmm.py` -> `results/hmm_*.csv` |
| End-to-end pipeline | `src/pipeline.py` |
| API | `api/main.py` (FastAPI endpoints, model loaded once at startup) |
| UI | `ui/index.html`, `ui/app.js`, `ui/style.css` (Leaflet) |
| Model weights and configuration | `models/final_model.joblib`, `models/feature_cols.json`, `models/hmm_params.json`, `models/README_model.txt` |
| Visualisations | `src/maps.py` (Folium maps -> `results/maps/`), `report/make_figures.py`, `report/screenshot_maps.py` |
| Tests | `tests/test_hmm.py` (unit tests), `tests/api_smoke.py` (checks a running API) |
| Supporting results | `results/` (CSV results, logs, configs, plots), `results/eda/`, `results/round1_preprocessing_summary.md` |
| Report | `report/report.md`, `report/report.pdf`, `report/build_pdf.py` |
| Raw data (read-only) | `data/train/` (80 trips), `data/student_test/` (10 trips) |
| Hackathon documents | `docs/` |

## Requirements

- Python 3.11 through 3.13 (results produced with 3.12; `requirements.txt` pins the versions used)
- The committed trip files in `data/train/` and `data/student_test/`
- Internet access for the UI's Leaflet, OpenStreetMap tiles, and web fonts

Install dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Raw trip files are read-only. Generated candidate tables belong in `data/processed/`, which is git-ignored.

## Train and export classifier

Build the candidate tables (R = 80 m, chosen on train trips by `python -m src.features`), compare the five models, then tune and export the chosen one:

```powershell
python training/preprocess_dataset.py --radius-m 80   # data/processed/{train,student_test}
python training/train_models.py                       # 5 models, 5-fold GroupKFold by trip -> results/model_comparison.csv
python training/bayes_opt.py                          # Optuna TPE, 50 trials (~20 min) -> results/optuna_*, models/final_model.joblib
```

`models/final_model.joblib` is the **tuned HistGradientBoosting** model: parameters from `results/optuna_best.json`, fitted on one fixed 100,000-row subsample (seed 42) of the 80 train trips. To re-export it without rerunning the search:

```powershell
python training/export_model.py
```

Both commands write `models/final_model.joblib` and `models/feature_cols.json` and give the same model (see `models/README_model.txt`). Do not train on `student_test` or evaluator trips. If the classifier artifact is missing, the API reports `distance-only` mode and uses a Gaussian distance score.

## Tune and evaluate the HMM

Tune sigma, beta and the gap threshold on train trips (default grid 3 x 3 x 3, first 20 train trips):

```powershell
python training/tune_hmm.py --max-trips 20
```

This writes `results/hmm_tuning.csv` and `models/hmm_params.json`, selected by mean observation-level Top-1 (macro F1 breaks ties). The classifier was fitted on train trips, so these train scores are optimistic development numbers.

Compare classifier-only with classifier + HMM on the ten student_test trips (final comparison only):

```powershell
python training/evaluate_hmm.py
```

This writes `results/hmm_comparison.csv`, `results/hmm_failures.csv` and two maps in `results/maps/` (one trip where the HMM helps, one where it does not).

## EDA and report

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb   # EDA tables and plots -> results/eda/
python -m src.maps                     # Folium maps for one train and one student_test trip -> results/maps/
python training/train_scores.py        # train-set scores of the five models -> results/model_train_scores.csv
python report/make_figures.py          # static figures -> report/figures/
python -m playwright install chromium  # once, for the map screenshots
python report/screenshot_maps.py       # Folium map screenshots -> report/figures/folium_*.png
python report/build_pdf.py             # report/report.md -> report/report.pdf
```

## Run the API

From the repository root on the laptop hosting predictions:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API loads the classifier and HMM settings once at process startup. Endpoints:

- `GET /health` reports service and predictor mode.
- `GET /trips` returns available trip IDs.
- `POST /predict` with `{"trip_id":"00000007"}` returns GPS coordinates, the predicted route, per-observation predicted arc IDs, and metrics.

The `.route` reference is read and evaluated on the server only. Reference labels and reference geometry are never returned to the browser. Every prediction request and completion/error is logged to the API console.

Useful environment variables (set before starting Uvicorn):

| Variable | Default | Purpose |
|---|---|---|
| `EVAL_DIR` | `data/eval` if it exists, otherwise `data/student_test` | Directory containing trip subfolders |
| `MODEL_PATH` | `models/final_model.joblib` | Classifier artifact |
| `FEATURE_COLUMNS_PATH` | `models/feature_cols.json` | Classifier feature schema |
| `CANDIDATE_RADIUS_M` | `results/preprocessing.json` value, otherwise `80` | Candidate search radius |
| `HMM_SIGMA_M` | Tuned file value, otherwise `25` | Gaussian emission distance scale |
| `HMM_BETA_M` | Tuned file value, otherwise `50` | Transition-distance scale |
| `HMM_GAP_THRESHOLD_S` | Tuned file value, otherwise `5` | Start a new Viterbi chain after this time gap |
| `HMM_MAX_SEARCH_M` | Tuned file value, otherwise `3000` | Bound road-graph shortest-path searches |
| `HMM_MODEL_WEIGHT` | Tuned file value, otherwise `0.75` | Blend classifier and distance emissions |
| `CORS_ORIGINS` | `*` | Comma-separated browser origins allowed by CORS |

Check a running API end to end (all trips, contract, timing -> `results/api_timing.csv`):

```powershell
python tests/api_smoke.py http://127.0.0.1:8000
```

## Run the UI

On the laptop displaying the map, from the repository root:

```powershell
python -m http.server 5500 --directory ui
```

Open `http://localhost:5500`, set **API base URL** to `http://<API-laptop-LAN-IP>:8000`, then load trips and predict. The map shows GPS observations in blue and the predicted route in green; metrics and decoded road arc IDs appear beside it.

The trip list is discovered from the API on every refresh. Teammate code changes reach this UI when they are merged into the shared branch and pulled on the UI laptop; refresh the page after updating the checkout. The UI does not need hardcoded trip IDs.

## Two-laptop run

**Laptop 1 (API):**

1. Find its LAN address: `ipconfig` -> "IPv4 Address" of the Wi-Fi adapter, e.g. `192.168.1.23`.
2. Allow port 8000 once (PowerShell as Administrator):
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
   ```
3. Start the API and keep this console visible for the evaluator (every request is logged):
   ```powershell
   $env:EVAL_DIR = "evaluator_test"   # folder with the 10 evaluator trips; leave unset for data/student_test
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

**Laptop 2 (UI):**

1. `python -m http.server 5500 --directory ui`
2. Open `http://localhost:5500`, set **API base URL** to `http://192.168.1.23:8000` (Laptop 1's address), click **Load trips**, pick a trip, click **Predict route**.
3. Quick check from Laptop 2 first: open `http://192.168.1.23:8000/health` in the browser.

**Hotspot fallback:** if the venue Wi-Fi blocks laptop-to-laptop traffic (health check times out), turn on the phone or Laptop 1 mobile hotspot, join both laptops to it, rerun `ipconfig` on Laptop 1 and use the new address.
