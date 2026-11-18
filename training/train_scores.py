"""Train-set macro F1 and log loss for the five round-1 models, to compare with CV (over/underfitting).

Same models, same 100,000-row train subsample (seed 42) as train_models.py. Train trips only.
python training/train_scores.py -> results/model_train_scores.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import f1_score, log_loss

from training.train_models import FEATURES, MODELS, RESULTS, load_partition, subsample

rows = subsample(load_partition("train"))
out = []
for name, make in MODELS.items():
    prob = make().fit(rows[FEATURES], rows["label"]).predict_proba(rows[FEATURES])[:, 1]
    out.append({"model": name, "train_macro_f1": f1_score(rows["label"], prob >= 0.5, average="macro"),
                "train_log_loss": log_loss(rows["label"], prob, labels=[0, 1])})
    print(out[-1], flush=True)
pd.DataFrame(out).to_csv(RESULTS / "model_train_scores.csv", index=False)
