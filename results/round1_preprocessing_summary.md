# Round 1: preprocessing and feature engineering summary

All numbers are copied from files produced by the code. Nothing is estimated.
Sources: `results/preprocessing.json` and `results/candidate_miss_rate.csv`, both written by `python -m src.features` on the train partition. `results/preprocess_log.txt` is written by `python training/preprocess_dataset.py --radius-m 80` and covers both partitions. `results/eda/*.csv` comes from `notebooks/01_eda.ipynb`.

## 1. Cleaning rules (`src/data_io.load_trip`) and rows dropped

| Rule | train rows dropped | student_test rows dropped |
|---|---|---|
| Drop rows with NaN/inf lon, lat or t, or lon outside [-180, 180], or lat outside [-90, 90] | 0 | 0 |
| Sort by t (stable mergesort) | 0 trips were out of order | 0 trips were out of order |
| Drop duplicate timestamps (keep first) | 0 | 0 |
| Flag time gaps > 1.5 s (`is_gap_after`; rows are kept) | 7,273 gaps flagged | 264 gaps flagged |
| Points kept | 208,417 | 19,160 |

Source: `results/preprocessing.json` (train) and `results/eda/missing_invalid.csv` and `results/eda/timestamps.csv` (both partitions).

## 2. Labels (`src/labels.derive_point_labels`)

- Method: walk `.route` in order. Each GPS point gets the nearest route arc among the current position and the next 8 (`lookahead=8`). The position never moves backwards.
- **Label mismatch rate (train): 0.0856** (8.56%). This is the share of points whose monotonic label differs from the unconstrained nearest route arc (`labels.label_mismatch_rate`).
- **Label mismatch rate (student_test): 0.0903** (19,160 points). It uses the same `derive_point_labels` and `label_mismatch_rate` functions as train, is reported only, and was not used for any choice.

## 3. Candidate miss rate per radius (train, `src/features.py` sweep)

Miss = the point's labelled arc is not among its candidates. Candidates are the arcs whose segment lies within R metres of the point, found with a KD-tree on arc midpoints and then the exact point-to-segment distance.

| R (m) | miss rate | candidates per point |
|---|---|---|
| 30 | 0.1441 | 5.27 |
| 50 | 0.1293 | 8.96 |
| 80 | 0.1202 | 16.23 |

- **Chosen R = 80 m.** Rule: the smallest R whose miss rate is within 0.005 of the best. Only 80 m qualifies.
- student_test at R = 80 m: miss rate **0.0045**, with 20.32 candidates per point. This is the point-weighted mean of the 10 per-trip rates in `preprocess_log.txt`. It is reported only and was not used to choose R.
- Where the train misses come from: they are concentrated in a few trips. At R = 80 m, per-trip miss rates are 0.8847 for trip 00000054 (19,313 points), 0.64 for 00000093 and 0.5128 for 00000033 (`preprocess_log.txt`). In those trips the GPS track lies far from the reference route, which is a limitation of the reference route rather than of the candidate search.

## 4. Features (one row per (obs_idx, arc_id), `src/features.build_candidate_features`)

| Feature | Why |
|---|---|
| `dist_m` | Perpendicular distance from the GPS point to the arc in metres. The true road is usually the closest one. |
| `distance_rank` | Rank of the arc by distance among this point's candidates, so it is relative to the neighbours and robust to noise level. |
| `heading_diff_deg` | Angle between the GPS motion direction and the arc bearing. It separates the two directions of a road and parallel roads. |
| `speed_mps` | GPS speed from the previous point. At low speed the heading is unreliable, so the model can down-weight it. |
| `arc_length_m` | Long arcs collect more nearby points, so this gives the model a prior. |
| `from_degree`, `to_degree` | Node degree at the arc's ends, which marks junction arcs where confusion is likely. |
| `candidate_count` | Number of arcs within R, a measure of how ambiguous the point is (dense network vs a single road). |
| `distance_gap_m` | Gap between the nearest and second-nearest candidate. A large gap means a confident match. It is NaN when there is one candidate and is filled with R in `training/train_models.py`. |
| `offset_fraction` | Position of the closest point along the arc (0 to 1). Values at 0 or 1 mean the point projects onto an arc end, so it is probably on a neighbouring arc. |
| `label` | 1 if `arc_id` equals the labelled arc of the point, else 0. This is the target, not a feature. |

No raw lat/lon is used because train and student_test sit in different regions. No feature uses the true previous arc, which would leak the label.
