#!/usr/bin/env python
"""
Optuna-based hyperparameter tuning for LightGBM, CatBoost, and XGBoost.
Results are stored in outputs/best_params.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SEED = 42


def load_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train = pd.read_csv(OUTPUT_DIR / "train_features.csv")
    target = (train["Class"] == "NG").astype(int)
    features = train.drop(columns=["Class"])
    return features, target


def tune_lightgbm(features: pd.DataFrame, target: pd.Series, n_trials: int = 12) -> dict:
    categorical_cols = [c for c in features.columns if features[c].dtype == "object"]
    for col in categorical_cols:
        features[col] = features[col].astype("category")
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "random_state": SEED,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08),
            "num_leaves": trial.suggest_int("num_leaves", 32, 256),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "n_estimators": 2000,
        }
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        oof = np.zeros(len(features))
        for tr_idx, val_idx in folds.split(features, target):
            model = lgb.LGBMClassifier(**params)
            model.fit(
                features.iloc[tr_idx],
                target.iloc[tr_idx],
                eval_set=[(features.iloc[val_idx], target.iloc[val_idx])],
                eval_metric="auc",
                categorical_feature=categorical_cols,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )
            oof[val_idx] = model.predict_proba(features.iloc[val_idx])[:, 1]
        return roc_auc_score(target, oof)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_catboost(features: pd.DataFrame, target: pd.Series, n_trials: int = 2) -> dict:
    features, target = sample_data(features, target)
    categorical_cols = [c for c in features.columns if features[c].dtype == "object"]
    cat_indices = [features.columns.get_loc(col) for col in categorical_cols]
    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": 600,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 4, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_seed": SEED,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": False,
            "od_type": "Iter",
            "od_wait": 200,
            "class_weights": [1.0, (target == 0).sum() / (target == 1).sum()],
        }
        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        oof = np.zeros(len(features))
        for tr_idx, val_idx in folds.split(features, target):
            model = CatBoostClassifier(**params)
            model.fit(
                features.iloc[tr_idx],
                target.iloc[tr_idx],
                eval_set=(features.iloc[val_idx], target.iloc[val_idx]),
                cat_features=cat_indices,
            )
            oof[val_idx] = model.predict_proba(features.iloc[val_idx])[:, 1]
        return roc_auc_score(target, oof)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgboost(features: pd.DataFrame, target: pd.Series, n_trials: int = 10) -> dict:
    features, target = sample_data(features, target)
    processed = features.copy()
    categorical_cols = [c for c in processed.columns if processed[c].dtype == "object"]
    for col in categorical_cols:
        processed[col] = processed[col].astype("category").cat.codes

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "scale_pos_weight": (target == 0).sum() / (target == 1).sum(),
            "seed": SEED,
        }
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        oof = np.zeros(len(processed))
        for tr_idx, val_idx in folds.split(processed, target):
            dtrain = xgb.DMatrix(processed.iloc[tr_idx], label=target.iloc[tr_idx])
            dval = xgb.DMatrix(processed.iloc[val_idx], label=target.iloc[val_idx])
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=4000,
                evals=[(dval, "valid")],
                early_stopping_rounds=200,
                verbose_eval=False,
            )
            oof[val_idx] = model.predict(xgb.DMatrix(processed.iloc[val_idx]), iteration_range=(0, model.best_iteration + 1))
        return roc_auc_score(target, oof)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def main(models: List[str]) -> None:
    features, target = load_features()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_params = {}
    params_path = OUTPUT_DIR / "best_params.json"
    if params_path.exists():
        best_params = json.loads(params_path.read_text())
    if "lightgbm" in models:
        best_params["lightgbm"] = tune_lightgbm(features.copy(), target)
    if "catboost" in models:
        best_params["catboost"] = tune_catboost(features.copy(), target)
    if "xgboost" in models:
        best_params["xgboost"] = tune_xgboost(features.copy(), target)
    with params_path.open("w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2)
    print("Saved best params:", best_params)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lightgbm", "catboost", "xgboost"],
        choices=["lightgbm", "catboost", "xgboost"],
    )
    args = parser.parse_args()
    main(args.models)
