"""Bayesian optimisation (Optuna TPE) of HistGradientBoosting, then tuned vs untuned evaluation.

Search uses train trips only (same subsample, features and GroupKFold as train_models.py).
student_test is loaded only in evaluate(), after the search has finished.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.train_models import FEATURES, N_FOLDS, RESULTS, SEED, TRAIN_SUBSAMPLE, load_partition, subsample, top1

N_TRIALS = 50
MODELS_DIR = Path("models")
SPACE = {  # name: (low, high, log scale?)
    "learning_rate": (0.01, 0.3, True),
    "max_iter": (100, 600, False),
    "max_leaf_nodes": (15, 127, False),
    "min_samples_leaf": (10, 200, False),
    "l2_regularization": (1e-6, 10.0, True),
    "max_bins": (64, 255, False),
}


def make_model(params):
    # HGB needs no scaling (tree splits are scale-free), so there is no preprocessing to fit
    return HistGradientBoostingClassifier(random_state=SEED, **params)


def cross_validate(params, train):
    """Same protocol as train_models.py: GroupKFold by trip, fixed subsample of each training fold."""
    oof, fold_f1 = np.zeros(len(train)), []
    for tr_idx, va_idx in GroupKFold(n_splits=N_FOLDS).split(train, groups=train["trip_id"]):
        fold = subsample(train.iloc[tr_idx])
        model = make_model(params).fit(fold[FEATURES], fold["label"])
        oof[va_idx] = model.predict_proba(train.iloc[va_idx][FEATURES])[:, 1]
        fold_f1.append(f1_score(train["label"].iloc[va_idx], oof[va_idx] >= 0.5, average="macro"))
    return oof, fold_f1


def search(train):
    def objective(trial):
        params = {}
        for name, (low, high, log) in SPACE.items():
            if isinstance(low, int):
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_float(name, low, high, log=log)
        return float(np.mean(cross_validate(params, train)[1]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)

    trials = study.trials_dataframe()[["number", "value", "duration"] + [f"params_{n}" for n in SPACE]]
    trials["duration_s"] = trials.pop("duration").dt.total_seconds()
    trials.to_csv(RESULTS / "optuna_trials.csv", index=False)
    (RESULTS / "optuna_best.json").write_text(json.dumps(
        {"best_value_cv_macro_f1": study.best_value, "best_trial": study.best_trial.number, "params": study.best_params}, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trials["number"], trials["value"], "o", alpha=0.6, label="trial value")
    ax.plot(trials["number"], trials["value"].cummax(), "r-", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("mean CV macro F1 (train, 5-fold GroupKFold)")
    ax.set_title("Optuna TPE history: HistGradientBoosting"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "optuna_history.png", dpi=120); plt.close(fig)

    importance = optuna.importance.get_param_importances(study)  # default: fANOVA
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(importance)[::-1], list(importance.values())[::-1], color="#4C78A8")
    ax.set_xlabel("importance (fANOVA)"); ax.set_title("Optuna parameter importance (train CV)")
    fig.tight_layout(); fig.savefig(RESULTS / "optuna_param_importance.png", dpi=120); plt.close(fig)
    return study.best_params


def evaluate(name, params, train):
    """CV, train-subsample and student_test metrics for one parameter set. student_test is loaded HERE only."""
    oof, fold_f1 = cross_validate(params, train)
    fit_rows = subsample(train)
    t0 = time.perf_counter()
    model = make_model(params).fit(fit_rows[FEATURES], fit_rows["label"])
    train_s = time.perf_counter() - t0
    fit_prob = model.predict_proba(fit_rows[FEATURES])[:, 1]

    test = load_partition("student_test")
    t0 = time.perf_counter()
    prob = model.predict_proba(test[FEATURES])[:, 1]
    infer_s = time.perf_counter() - t0
    pred = prob >= 0.5
    cm = confusion_matrix(test["label"], pred)
    row = {"model": name,
           "cv_macro_f1_mean": np.mean(fold_f1), "cv_macro_f1_std": np.std(fold_f1),
           "cv_log_loss": log_loss(train["label"], oof, labels=[0, 1]), "cv_top1": top1(train, oof),
           "train_macro_f1": f1_score(fit_rows["label"], fit_prob >= 0.5, average="macro"),
           "train_log_loss": log_loss(fit_rows["label"], fit_prob, labels=[0, 1]),
           "test_macro_f1": f1_score(test["label"], pred, average="macro"), "test_top1": top1(test, prob),
           "test_log_loss": log_loss(test["label"], prob, labels=[0, 1]),
           "test_precision": precision_score(test["label"], pred), "test_recall": recall_score(test["label"], pred),
           "test_tn_fp_fn_tp": " ".join(map(str, cm.ravel())),
           "train_time_s": train_s, "infer_time_s": infer_s}
    return row, model, cm


def main():
    RESULTS.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    train = load_partition("train")
    (RESULTS / "optuna_config.json").write_text(json.dumps(
        {"model": "HistGradientBoostingClassifier", "sampler": "TPESampler", "seed": SEED, "n_trials": N_TRIALS,
         "objective": "mean CV macro F1 at threshold 0.5", "cv": f"GroupKFold({N_FOLDS}) by trip_id, train trips only",
         "train_subsample_rows": TRAIN_SUBSAMPLE, "features": FEATURES,
         "search_space": {n: {"low": lo, "high": hi, "log": lg} for n, (lo, hi, lg) in SPACE.items()}}, indent=2))

    best = search(train)  # student_test is not loaded in here
    print("BEST PARAMS:", best)

    rows, models, cms = [], {}, {}
    for name, params in [("HGB_untuned", {}), ("HGB_tuned", best)]:
        row, models[name], cms[name] = evaluate(name, params, train)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "tuned_vs_untuned.csv", index=False)

    ConfusionMatrixDisplay(cms["HGB_tuned"]).plot(colorbar=False)
    plt.title("HistGradientBoosting tuned: student_test")
    plt.tight_layout(); plt.savefig(RESULTS / "confusion_HistGradientBoosting_tuned.png", dpi=120); plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, col in zip(axes, ["cv_macro_f1_mean", "test_macro_f1", "test_top1", "test_log_loss"]):
        ax.bar(table["model"], table[col], color=["#999999", "#4C78A8"])
        ax.set_title(col)
        for i, v in enumerate(table[col]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(RESULTS / "tuned_vs_untuned.png", dpi=120); plt.close(fig)

    pd.set_option("display.width", 250)
    print(table.round(4).T.to_string())
    for _, r in table.iterrows():
        print(f"{r['model']}: train macro F1 {r['train_macro_f1']:.4f} vs CV {r['cv_macro_f1_mean']:.4f} "
              f"(gap {r['train_macro_f1'] - r['cv_macro_f1_mean']:+.4f}); train log loss {r['train_log_loss']:.4f} "
              f"vs CV {r['cv_log_loss']:.4f}")

    winner = table.loc[table["cv_macro_f1_mean"].idxmax(), "model"]  # chosen on train CV only
    joblib.dump(models[winner], MODELS_DIR / "final_model.joblib")
    (MODELS_DIR / "feature_cols.json").write_text(json.dumps(FEATURES, indent=2))
    gain = table.set_index("model").loc["HGB_tuned", "cv_macro_f1_mean"] - table.set_index("model").loc["HGB_untuned", "cv_macro_f1_mean"]
    print(f"EXPORTED: {winner} -> models/final_model.joblib (higher mean CV macro F1 on train trips; "
          f"tuned minus untuned CV gain = {gain:+.4f})")


if __name__ == "__main__":
    main()
