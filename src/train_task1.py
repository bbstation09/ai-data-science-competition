#!/usr/bin/env python
"""
Task1 LightGBM training script using engineered features.
- Loads outputs/train_features.csv & outputs/test_features.csv
- Executes Stratified K-Fold LightGBM training
- Saves metrics, feature importances, SHAP summary, and prediction files
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
SEED = 42


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(OUTPUT_DIR / "train_features.csv")
    test = pd.read_csv(OUTPUT_DIR / "test_features.csv")
    return train, test


def train_lightgbm(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    feature_cols = [c for c in train.columns if c != "Class"]
    target = (train["Class"] == "NG").astype(int)
    categorical_cols = [c for c in feature_cols if train[c].dtype == "object"]
    for df in [train, test]:
        for col in categorical_cols:
            df[col] = df[col].astype("category")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    params = {
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 25,
        "reg_alpha": 0.1,
        "reg_lambda": 0.3,
        "objective": "binary",
        "random_state": SEED,
        "n_jobs": -1,
        "scale_pos_weight": (target == 0).sum() / (target == 1).sum(),
    }

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    importances = np.zeros((len(feature_cols), folds.n_splits))
    for fold, (tr_idx, val_idx) in enumerate(folds.split(train, target), 1):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            train.iloc[tr_idx][feature_cols],
            target.iloc[tr_idx],
            eval_set=[(train.iloc[val_idx][feature_cols], target.iloc[val_idx])],
            eval_metric="auc",
            categorical_feature=categorical_cols,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        val_pred = model.predict_proba(train.iloc[val_idx][feature_cols])[:, 1]
        oof[val_idx] = val_pred
        test_pred += model.predict_proba(test[feature_cols])[:, 1] / folds.n_splits
        importances[:, fold - 1] = model.feature_importances_

    oof_auc = roc_auc_score(target, oof)
    fold_metrics = []
    for fold, (_, val_idx) in enumerate(folds.split(train, target), 1):
        fold_auc = roc_auc_score(target.iloc[val_idx], oof[val_idx])
        fold_metrics.append({"fold": fold, "auc": fold_auc})

    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importances.mean(axis=1)}
    ).sort_values("importance", ascending=False)

    final_model = lgb.LGBMClassifier(**params).fit(
        train[feature_cols], target, categorical_feature=categorical_cols
    )

    return {
        "oof": oof,
        "test_pred": test_pred,
        "overall_auc": oof_auc,
        "fold_metrics": fold_metrics,
        "importance": importance_df,
        "feature_cols": feature_cols,
        "target": target,
        "model": final_model,
    }


def save_artifacts(results: dict, test: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "overall_auc": results["overall_auc"],
        "folds": results["fold_metrics"],
    }
    with (OUTPUT_DIR / "task1_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    results["importance"].head(100).to_csv(
        OUTPUT_DIR / "task1_feature_importances.csv", index=False
    )

    preds = pd.DataFrame(
        {"row_id": np.arange(len(results["test_pred"])), "probability": results["test_pred"]}
    )
    preds.to_csv(OUTPUT_DIR / "task1_test_pred.csv", index=False)

    submission = pd.DataFrame({"ID": test.index, "probability": results["test_pred"]})
    submission["decision"] = False
    submission.to_csv(OUTPUT_DIR / "task1_submission_stub.csv", index=False)

    thresholds = np.linspace(0, 1, 101)
    scores = []
    target = results["target"]
    oof = results["oof"]
    for thr in thresholds:
        preds_bin = (oof >= thr).astype(int)
        tp = ((preds_bin == 1) & (target == 1)).sum()
        fp = ((preds_bin == 1) & (target == 0)).sum()
        tn = ((preds_bin == 0) & (target == 0)).sum()
        fn = ((preds_bin == 0) & (target == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        scores.append((thr, f1))
    pd.DataFrame(scores, columns=["threshold", "f1"]).to_csv(
        OUTPUT_DIR / "task1_threshold_search.csv", index=False
    )


def plot_artifacts(results: dict, train: pd.DataFrame) -> None:
    importance = results["importance"].head(30)
    plt.figure(figsize=(8, 10))
    plt.barh(importance["feature"], importance["importance"])
    plt.title("LightGBM Feature Importance (Top 30)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task1_feature_importance.png", dpi=200)
    plt.close()

    explainer = shap.TreeExplainer(results["model"])
    sample = train[results["feature_cols"]].sample(
        n=min(400, len(train)), random_state=SEED
    )
    shap_values = explainer.shap_values(sample)
    shap_values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
    plt.figure()
    shap.summary_plot(shap_values_to_plot, sample, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task1_shap_summary.png", dpi=200)
    plt.close()


def main() -> None:
    train, test = load_features()
    results = train_lightgbm(train, test)
    save_artifacts(results, test)
    plot_artifacts(results, train)
    print(f"Task1 LightGBM AUC: {results['overall_auc']:.4f}")


if __name__ == "__main__":
    main()
