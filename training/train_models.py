"""Round 1: compare 5 model families on the candidate tables (GroupKFold by trip)."""

import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
# env overrides exist only for the smoke test (few trips, few rows, 2 folds)
N_FOLDS = int(os.environ.get("N_FOLDS", 5))
TRAIN_SUBSAMPLE = int(os.environ.get("SUBSAMPLE", 100_000))  # shortcut: cut from 200k to 100k because KNN alone took >10 min at 200k; raise if time allows
N_TRAIN_TRIPS = int(os.environ.get("N_TRIPS", 0))  # 0 = all train trips
ONLY = os.environ.get("ONLY")  # run a single model (timing check)
FEATURES = ["dist_m", "distance_rank", "heading_diff_deg", "speed_mps", "arc_length_m",
            "from_degree", "to_degree", "candidate_count", "distance_gap_m", "offset_fraction"]
PROCESSED = Path("data/processed")
RESULTS = Path("results")
RADIUS = json.loads((RESULTS / "preprocessing.json").read_text())["radius_m"]

MODELS = {  # scaling only for distance/gradient-based models; inside the pipeline so it is fit on train folds only
    "LogisticRegression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    "GaussianNB": lambda: GaussianNB(),
    "KNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, n_jobs=-1)),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(random_state=SEED),
}


def load_partition(name):
    files = sorted((PROCESSED / name).glob("*_candidates.csv"))
    if name == "train" and N_TRAIN_TRIPS:
        files = files[:N_TRAIN_TRIPS]
    frames = [pd.read_csv(f).assign(trip_id=f.stem.split("_")[0]) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["distance_gap_m"] = df["distance_gap_m"].fillna(RADIUS)  # single candidate: no competitor within R
    return df


def subsample(df):
    return df.sample(n=min(TRAIN_SUBSAMPLE, len(df)), random_state=SEED)


def top1(df, prob):
    """Share of GPS points whose highest-probability candidate is the true arc."""
    best = df.assign(prob=prob).sort_values("prob", ascending=False).drop_duplicates(["trip_id", "obs_idx"])
    return float(best["label"].mean())


def scores(df, prob):
    return {"macro_f1": f1_score(df["label"], prob >= 0.5, average="macro"),
            "log_loss": log_loss(df["label"], prob, labels=[0, 1]),
            "top1": top1(df, prob)}


def cross_validate(name, train):
    oof, fold_f1 = np.zeros(len(train)), []
    for tr_idx, va_idx in GroupKFold(n_splits=N_FOLDS).split(train, groups=train["trip_id"]):
        fold = subsample(train.iloc[tr_idx])
        model = MODELS[name]().fit(fold[FEATURES], fold["label"])
        oof[va_idx] = model.predict_proba(train.iloc[va_idx][FEATURES])[:, 1]
        fold_f1.append(f1_score(train["label"].iloc[va_idx], oof[va_idx] >= 0.5, average="macro"))
    return oof, fold_f1


def final_evaluation(name, train):
    """Fit on (subsampled) train; student_test is loaded HERE only, for scoring."""
    test = load_partition("student_test")
    fit_rows = subsample(train)
    t0 = time.perf_counter()
    model = MODELS[name]().fit(fit_rows[FEATURES], fit_rows["label"])
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    prob = model.predict_proba(test[FEATURES])[:, 1]
    infer_s = time.perf_counter() - t0
    return test, prob, train_s, infer_s


def main():
    RESULTS.mkdir(exist_ok=True)
    train = load_partition("train")
    config = {"seed": SEED, "n_folds": N_FOLDS, "cv": "GroupKFold by trip_id", "train_subsample_rows": TRAIN_SUBSAMPLE,
              "train_rows_total": len(train), "radius_m": RADIUS, "features": FEATURES, "threshold": 0.5,
              "models": {n: str(MODELS[n]()) for n in MODELS}}
    (RESULTS / "train_config.json").write_text(json.dumps(config, indent=2))
    print(f"NOTE: each fit uses a fixed random subsample of {TRAIN_SUBSAMPLE} rows "
          f"(of {len(train)} train rows, seed {SEED}); validation/test use all rows.")

    rows = []
    for name in [ONLY] if ONLY else MODELS:
        start = time.perf_counter()
        oof, fold_f1 = cross_validate(name, train)
        cv = scores(train, oof)
        test, prob, train_s, infer_s = final_evaluation(name, train)
        te = scores(test, prob)
        rows.append({"model": name, "cv_macro_f1_mean": np.mean(fold_f1), "cv_macro_f1_std": np.std(fold_f1),
                     "cv_macro_f1_oof": cv["macro_f1"], "cv_log_loss": cv["log_loss"], "cv_top1": cv["top1"],
                     "test_macro_f1": te["macro_f1"], "test_log_loss": te["log_loss"], "test_top1": te["top1"],
                     "train_time_s": train_s, "infer_time_s": infer_s})
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, (title, y, p) in zip(axes, [("CV out-of-fold", train["label"], oof), ("student_test", test["label"], prob)]):
            ConfusionMatrixDisplay(confusion_matrix(y, p >= 0.5)).plot(ax=ax, colorbar=False)
            ax.set_title(f"{name}: {title}")
        fig.tight_layout()
        fig.savefig(RESULTS / f"confusion_{name}.png", dpi=120)
        plt.close(fig)
        print(f"{name} done in {time.perf_counter() - start:.1f} s", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "model_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, col in zip(axes, ["cv_macro_f1_mean", "test_macro_f1", "test_top1", "test_log_loss"]):
        ax.barh(table["model"], table[col], color="#4C78A8")
        ax.set_title(col)
        for i, v in enumerate(table[col]):
            ax.text(v, i, f" {v:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(RESULTS / "model_comparison.png", dpi=120)

    print(table.round(4).to_string(index=False))
    best = table.sort_values("cv_macro_f1_mean", ascending=False).iloc[0]
    print(f"\nCHOSEN: {best['model']} - highest CV macro F1 ({best["cv_macro_f1_mean"]:.4f} +- {best["cv_macro_f1_std"]:.4f}, train trips only). "
          f"student_test check: macro F1 {best['test_macro_f1']:.4f}, Top-1 {best['test_top1']:.4f}, "
          f"log loss {best['test_log_loss']:.4f}. Best test Top-1 overall: "
          f"{table.loc[table['test_top1'].idxmax(), 'model']}.")


if __name__ == "__main__":
    main()
