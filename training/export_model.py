"""Re-export the tuned classifier without rerunning the Optuna search.

Uses exactly what training/bayes_opt.py used for its final fit: the 80 train trips,
the same 100,000-row subsample (seed 42), the ten features and the best parameters
in results/optuna_best.json. The result is the same model bayes_opt.py exports.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from training.bayes_opt import make_model
from training.train_models import FEATURES, RESULTS, load_partition, subsample


def main() -> None:
    best = json.loads((RESULTS / "optuna_best.json").read_text())["params"]
    rows = subsample(load_partition("train"))  # train trips only
    model = make_model(best).fit(rows[FEATURES], rows["label"])
    out = Path("models")
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "final_model.joblib")
    (out / "feature_cols.json").write_text(json.dumps(FEATURES, indent=2))
    print(f"Exported tuned HistGradientBoosting {best} fitted on {len(rows)} train rows -> models/final_model.joblib")


if __name__ == "__main__":
    main()
