#!/usr/bin/env python
"""
Grid-search ensemble weights for LightGBM, CatBoost, and XGBoost using saved OOF predictions.
Outputs:
- Best weight combination and corresponding AUC
- Updated ensemble predictions/submission
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS = ["lightgbm", "catboost", "xgboost"]
STEP = 0.05


def load_predictions(prefix: str) -> dict[str, np.ndarray]:
    preds = {}
    for name in MODELS:
        path = OUTPUT_DIR / f"task1_{name}_{prefix}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file: {path}")
        preds[name] = pd.read_csv(path)["probability"].to_numpy()
    return preds


def load_target() -> pd.Series:
    train = pd.read_csv(OUTPUT_DIR / "train_features.csv")
    return (train["Class"] == "NG").astype(int)


def grid_search_weights(oof_preds: dict[str, np.ndarray], target: pd.Series) -> tuple[dict[str, float], float]:
    weights = np.arange(0.0, 1.0 + 1e-9, STEP)
    best_auc = -1.0
    best_w = {}
    lgb = oof_preds["lightgbm"]
    cat = oof_preds["catboost"]
    xgb = oof_preds["xgboost"]
    for w1 in weights:
        for w2 in weights:
            if w1 + w2 > 1.0:
                continue
            w3 = 1.0 - w1 - w2
            ensemble = w1 * lgb + w2 * cat + w3 * xgb
            auc = roc_auc_score(target, ensemble)
            if auc > best_auc:
                best_auc = auc
                best_w = {"lightgbm": w1, "catboost": w2, "xgboost": w3}
    return best_w, best_auc


def apply_weights(test_preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weights[name] * preds for name, preds in test_preds.items())


def main() -> None:
    target = load_target()
    oof_preds = load_predictions("oof")
    best_w, best_auc = grid_search_weights(oof_preds, target)
    test_preds = load_predictions("test_pred")
    ensemble_test = apply_weights(test_preds, best_w)
    pd.DataFrame({"row_id": np.arange(len(ensemble_test)), "probability": ensemble_test}).to_csv(
        OUTPUT_DIR / "task1_reweighted_test_pred.csv", index=False
    )
    pd.DataFrame({"row_id": np.arange(len(target)), "probability": apply_weights(oof_preds, best_w)}).to_csv(
        OUTPUT_DIR / "task1_reweighted_oof.csv", index=False
    )
    submission = pd.read_csv(OUTPUT_DIR / "task1_ensemble_submission.csv")
    submission["probability"] = np.repeat(ensemble_test, 2)[: len(submission)]
    submission.to_csv(OUTPUT_DIR / "task1_reweighted_submission.csv", index=False)
    with (OUTPUT_DIR / "ensemble_reweight_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump({"best_weights": best_w, "auc": best_auc}, fp, indent=2)
    print(f"Best weights: {best_w}, AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
