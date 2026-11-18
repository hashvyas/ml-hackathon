# PathSnap

PathSnap maps noisy GPS observations to a directed road network. The prediction service finds nearby road-segment candidates, scores them with the exported classifier (or an explicit distance-only baseline), decodes a connected sequence with a log-space HMM/Viterbi pass, and reconstructs a route. The browser UI connects to the API over a LAN.

## Requirements

- Python 3.10 through 3.13
- The committed trip files in `data/train/` and `data/student_test/`
- Internet access for the UI's Leaflet, OpenStreetMap tiles, and web fonts

Install dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Raw trip files are read-only. Generated candidate tables belong in `data/processed/`, which is git-ignored.

## Train and export classifier

The checked-in round-one comparison selected `HistGradientBoosting` by train-only GroupKFold macro F1. Generate train candidates and export the selected classifier:

```powershell
python training/preprocess_dataset.py --partition train --radius-m 80
python training/export_model.py
```

The exporter uses only labelled `data/processed/train/` tables, takes at most 5,000 candidate rows from each trip with a fixed seed, and writes:

- `models/final_model.joblib`
- `models/feature_cols.json`

Do not train on `student_test` or evaluator trips. If the classifier artifact is missing, the API reports `distance-only` mode and uses a Gaussian distance score; it does not pretend a model is loaded.

## Tune HMM parameters

After exporting the classifier, tune the emission sigma and transition beta using the labelled development partition:

```powershell
python training/tune_hmm.py --partition student_test
```

This evaluates the default 3-by-3 sigma/beta grid on all ten student-test trips, selects by mean observation-level Top-1 accuracy (macro F1 breaks ties), and writes:

- `results/hmm_tuning.csv`
- `models/hmm_params.json`

The student-test set is used for model comparison and decoder tuning, so those scores are development results, not an unbiased final estimate. Leave `EVAL_DIR` unset to use `data/student_test/`; for reserved evaluator trips, point `EVAL_DIR` at their trip-folder directory.

Run the end-to-end pipeline once on each of the ten student-test trips and save the per-trip metrics:

```powershell
python training/evaluate_hmm.py
```

This writes `results/hmm_student_test.csv`; it uses the exported model and saved HMM parameters when present, and reports when it is using the distance-only baseline.

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

Use the laptop's LAN IPv4 address in the UI, not `127.0.0.1`. On Windows, allow inbound TCP port 8000 through Windows Firewall. Keep both laptops on the same Wi-Fi or hotspot.

## Run the UI

On the laptop displaying the map, from the repository root:

```powershell
python -m http.server 5500 --directory ui
```

Open `http://localhost:5500`, set **API base URL** to `http://<API-laptop-LAN-IP>:8000`, then load trips and predict. The map shows GPS observations in blue and the predicted route in green; metrics and decoded road arc IDs appear beside it.

The trip list is discovered from the API on every refresh. Teammate code changes reach this UI when they are merged into the shared branch and pulled on the UI laptop; refresh the page after updating the checkout. The UI does not need hardcoded trip IDs.

## Team workflow

Keep personal work on `kushal` and push it for team review:

```powershell
git switch kushal
git pull --ff-only origin kushal
git add <your-files>
git commit -m "Describe your change"
git push origin kushal
```

The team lead can merge the branch into `main`. To include teammates' merged changes in your checkout:

```powershell
git switch main
git pull --ff-only origin main
git switch kushal
git merge --ff-only main
```

Never write generated labels, candidates, predictions, or other outputs into the raw `data/train/` or `data/student_test/` folders.
