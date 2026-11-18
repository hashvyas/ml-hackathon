models/final_model.joblib
-------------------------
Model: HistGradientBoostingClassifier, tuned by Optuna TPE (training/bayes_opt.py, 50 trials,
objective = mean 5-fold GroupKFold macro F1 on the 80 train trips).
Parameters: results/optuna_best.json
Fit data: the 80 train trips only, one fixed 100,000-row subsample (seed 42) of the candidate
tables in data/processed/train (R = 80 m). No student_test or evaluator data.
Chosen over the untuned model by mean CV macro F1 (0.8788 vs 0.8767, results/tuned_vs_untuned.csv).
The gain is smaller than the fold standard deviation (~0.03).

How it is produced:
  python training/bayes_opt.py     # search + evaluation + export (about 20 min)
  python training/export_model.py  # re-export the same model from optuna_best.json (under 1 min)
Both give the same model (checked: identical predict_proba on student_test trip 00000007).

models/feature_cols.json: the ten feature names in the order the model expects.
Missing distance_gap_m (single candidate) is filled with R = 80 m in training and in src/pipeline.py.
