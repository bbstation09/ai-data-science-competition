#!/usr/bin/env python
"""
고도화된 Task1 학습 파이프라인.
구성 요소:
- LightGBM/XGBoost/CatBoost/ExtraTrees/TabNet 5개 모델 학습
- 10-Fold CV × 5개 시드 반복 → 안정적 OOF/TEST 예측
- Optuna 하이퍼 파라미터 튜닝 (기본 400 trials)
- Level-2 Stacking (Logistic Regression by default)
- Platt/Isotonic/Temperature calibration 적용 및 비교

Output artifacts:
- task1_level1_{model}_{oof/test}.csv
- task1_meta_{oof/test}.csv (stacked)
- task1_calibrated_{method}.csv (raw + calibrated 테스트 확률)
- task1_ensemble_metrics.json, task1_calibration_metrics.json
- task1_ensemble_submission.csv (최종 보정 확률 기반)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency 안내
    lgb = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception:  # pragma: no cover
    TabNetClassifier = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
DEFAULT_SEEDS = [111, 222, 333, 444, 555]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced Task1 training")
    parser.add_argument("--n_splits", type=int, default=10, help="CV fold 수")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="comma로 구분된 CV seeds",
    )
    parser.add_argument("--n_trials", type=int, default=400, help="Optuna trial 수")
    parser.add_argument("--skip_tuning", action="store_true", help="Optuna 생략 시 기존 best_params.json 사용")
    parser.add_argument("--stacker", type=str, default="logistic", choices=["logistic", "lightgbm"], help="Level-2 meta 모델")
    parser.add_argument("--enable_tabnet", action="store_true", help="TabNet 학습 포함 (설치 필요)")
    parser.add_argument("--enable_cnn", action="store_true", help="FEM 2D CNN Level-1 모델 포함")
    parser.add_argument("--calibration_metric", type=str, default="auc", choices=["auc", "brier"], help="보정 기법 선택 기준")
    parser.add_argument(
        "--models",
        type=str,
        default="lightgbm,xgboost,catboost,extratrees",
        help="쉼표 구분 Level-1 모델 목록",
    )
    parser.add_argument(
        "--blend_modes",
        type=str,
        default="rank,geometric",
        help="추가 블렌딩 모드 (rank, geometric 등)",
    )
    parser.add_argument(
        "--reuse_predictions",
        action="store_true",
        help="현재 저장된 level1/meta 예측을 사용하고 재학습을 생략",
    )
    return parser.parse_args()


def load_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(OUTPUT_DIR / "train_features.csv")
    test = pd.read_csv(OUTPUT_DIR / "test_features.csv")
    return train, test


def feature_metadata(train: pd.DataFrame) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in train.columns if c != "Class"]
    categorical_cols = [c for c in feature_cols if train[c].dtype == "object"]
    return feature_cols, categorical_cols


def stratified_folds(n_splits: int, seed: int, X: pd.DataFrame, y: pd.Series):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, y))


def average_oof_test(n_samples: int, test_len: int, total_models: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    oof = np.zeros(n_samples)
    counts = np.zeros(n_samples)
    test_pred = np.zeros(test_len)
    return oof, counts, test_pred


def update_predictions(oof, counts, test_pred, val_idx, val_pred, fold_test_pred):
    oof[val_idx] += val_pred
    counts[val_idx] += 1
    test_pred += fold_test_pred


def finalize_predictions(oof, counts, test_pred, total_models):
    oof = np.divide(oof, counts, out=np.zeros_like(oof), where=counts > 0)
    test_pred = test_pred / total_models
    return oof, test_pred


def suggest_params(model: str, trial: optuna.Trial) -> Dict:
    if model == "lightgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        }
    if model == "xgboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 1.0, log=True),
        }
    if model == "catboost":
        return {
            "depth": trial.suggest_int("depth", 5, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
    if model == "extratrees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
            "max_depth": trial.suggest_int("max_depth", 8, 40),
            "max_features": trial.suggest_float("max_features", 0.3, 0.9),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
    if model == "tabnet":
        return {
            "n_d": trial.suggest_int("n_d", 8, 32),
            "n_a": trial.suggest_int("n_a", 8, 32),
            "n_steps": trial.suggest_int("n_steps", 3, 7),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-3, log=True),
        }
    if model == "fem_cnn":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 48, 64, 96]),
            "epochs": trial.suggest_int("epochs", 40, 90),
            "patience": trial.suggest_int("patience", 6, 15),
        }
    raise ValueError(f"Unsupported model {model}")


def tune_model(
    model_name: str,
    train: pd.DataFrame,
    target: pd.Series,
    n_trials: int,
    n_splits: int,
) -> Dict:
    feature_cols, categorical_cols = feature_metadata(train)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(model_name, trial)
        folds = StratifiedKFold(n_splits=min(n_splits, 5), shuffle=True, random_state=trial.number + 7)
        scores = []
        train_lgb = None
        enc_train = None
        if model_name == "lightgbm":
            train_lgb = train[feature_cols].copy()
            for col in categorical_cols:
                train_lgb[col] = train_lgb[col].astype("category")
        elif model_name == "xgboost":
            enc_train, _ = encode_categoricals(train[feature_cols], train[feature_cols], categorical_cols)
        for tr_idx, val_idx in folds.split(train, target):
            if model_name == "lightgbm":
                if lgb is None:
                    raise RuntimeError("LightGBM not installed")
                model = lgb.LGBMClassifier(
                    n_estimators=1200,
                    objective="binary",
                    random_state=trial.number,
                    n_jobs=-1,
                    **params,
                )
                model.fit(train_lgb.iloc[tr_idx], target.iloc[tr_idx])
                preds = model.predict_proba(train_lgb.iloc[val_idx])[:, 1]
            elif model_name == "xgboost":
                if xgb is None:
                    raise RuntimeError("xgboost not installed")
                dtrain = xgb.DMatrix(enc_train.iloc[tr_idx], label=target.iloc[tr_idx])
                dval = xgb.DMatrix(enc_train.iloc[val_idx], label=target.iloc[val_idx])
                watch = [(dval, "valid")]
                booster = xgb.train(
                    {
                        "objective": "binary:logistic",
                        "eval_metric": "auc",
                        "eta": params["learning_rate"],
                        "max_depth": params["max_depth"],
                        "subsample": params["subsample"],
                        "colsample_bytree": params["colsample_bytree"],
                        "min_child_weight": params["min_child_weight"],
                        "lambda": params["lambda"],
                        "alpha": params["alpha"],
                        "verbosity": 0,
                    },
                    dtrain,
                    num_boost_round=600,
                    evals=watch,
                    early_stopping_rounds=50,
                    verbose_eval=False,
                )
                preds = booster.predict(xgb.DMatrix(enc_train.iloc[val_idx]))
            elif model_name == "catboost":
                if CatBoostClassifier is None:
                    raise RuntimeError("CatBoost not installed")
                cat_indices = [feature_cols.index(col) for col in categorical_cols]
                model = CatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="AUC",
                    iterations=2000,
                    random_seed=trial.number,
                    verbose=False,
                    **params,
                )
                model.fit(
                    train.iloc[tr_idx][feature_cols],
                    target.iloc[tr_idx],
                    eval_set=(train.iloc[val_idx][feature_cols], target.iloc[val_idx]),
                    cat_features=cat_indices,
                    use_best_model=True,
                )
                preds = model.predict_proba(train.iloc[val_idx][feature_cols])[:, 1]
            elif model_name == "extratrees":
                model = ExtraTreesClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    max_features=params["max_features"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=trial.number,
                    n_jobs=-1,
                )
                model.fit(train.iloc[tr_idx][feature_cols], target.iloc[tr_idx])
                preds = model.predict_proba(train.iloc[val_idx][feature_cols])[:, 1]
            elif model_name == "tabnet":
                if TabNetClassifier is None:
                    raise RuntimeError("TabNet not installed")
                clf = TabNetClassifier(
                    n_d=params["n_d"],
                    n_a=params["n_a"],
                    n_steps=params["n_steps"],
                    gamma=params["gamma"],
                    lambda_sparse=params["lambda_sparse"],
                    seed=trial.number,
                    verbose=0,
                )
                clf.fit(
                    train.iloc[tr_idx][feature_cols].values,
                    target.iloc[tr_idx].values,
                    eval_set=[(train.iloc[val_idx][feature_cols].values, target.iloc[val_idx].values)],
                    patience=30,
                    max_epochs=200,
                    batch_size=1024,
                    virtual_batch_size=128,
                    num_workers=0,
                    drop_last=False,
                )
                preds = clf.predict_proba(train.iloc[val_idx][feature_cols].values)[:, 1]
            else:
                raise ValueError(model_name)
            scores.append(roc_auc_score(target.iloc[val_idx], preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    encoded_train = train.copy()
    encoded_test = test.copy()
    for col in categorical_cols:
        codes, uniques = pd.factorize(pd.concat([encoded_train[col], encoded_test[col]], axis=0))
        encoded_train[col] = codes[: len(encoded_train)]
        encoded_test[col] = codes[len(encoded_train) :]
    return encoded_train, encoded_test


def train_lightgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: pd.Series,
    params: Dict,
    seeds: List[int],
    n_splits: int,
) -> Dict:
    if lgb is None:
        raise RuntimeError("LightGBM not installed")
    feature_cols, categorical_cols = feature_metadata(train)
    for df in [train, test]:
        for col in categorical_cols:
            df[col] = df[col].astype("category")
    base_params = {
        "n_estimators": 4000,
        "objective": "binary",
        "n_jobs": -1,
        "boosting_type": "gbdt",
    }
    base_params.update(params)
    oof, counts, test_pred = average_oof_test(len(train), len(test), len(seeds) * n_splits)
    total_models = 0
    models = []
    for seed in seeds:
        folds = stratified_folds(n_splits, seed, train, target)
        for fold_id, (tr_idx, val_idx) in enumerate(folds):
            model = lgb.LGBMClassifier(**base_params, random_state=seed)
            model.fit(
                train.iloc[tr_idx][feature_cols],
                target.iloc[tr_idx],
                eval_set=[(train.iloc[val_idx][feature_cols], target.iloc[val_idx])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
            )
            val_pred = model.predict_proba(train.iloc[val_idx][feature_cols])[:, 1]
            fold_test_pred = model.predict_proba(test[feature_cols])[:, 1]
            update_predictions(oof, counts, test_pred, val_idx, val_pred, fold_test_pred)
            models.append(model)
            total_models += 1
    oof, test_pred = finalize_predictions(oof, counts, test_pred, total_models)
    return {"name": "lightgbm", "oof": oof, "test": test_pred, "models": models, "params": base_params}


def train_xgboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: pd.Series,
    params: Dict,
    seeds: List[int],
    n_splits: int,
) -> Dict:
    if xgb is None:
        raise RuntimeError("xgboost not installed")
    feature_cols, categorical_cols = feature_metadata(train)
    encoded_train, encoded_test = encode_categoricals(train[feature_cols], test[feature_cols], categorical_cols)
    dtest = xgb.DMatrix(encoded_test)
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": params.get("learning_rate", 0.02),
        "max_depth": params.get("max_depth", 7),
        "subsample": params.get("subsample", 0.8),
        "colsample_bytree": params.get("colsample_bytree", 0.7),
        "min_child_weight": params.get("min_child_weight", 5.0),
        "lambda": params.get("lambda", 0.3),
        "alpha": params.get("alpha", 0.1),
    }
    oof, counts, test_pred = average_oof_test(len(train), len(test), len(seeds) * n_splits)
    total_models = 0
    boosters = []
    for seed in seeds:
        folds = stratified_folds(n_splits, seed, train, target)
        for fold_id, (tr_idx, val_idx) in enumerate(folds):
            dtrain = xgb.DMatrix(encoded_train.iloc[tr_idx], label=target.iloc[tr_idx])
            dval = xgb.DMatrix(encoded_train.iloc[val_idx], label=target.iloc[val_idx])
            booster = xgb.train(
                dict(base_params, seed=seed),
                dtrain,
                num_boost_round=6000,
                evals=[(dval, "valid")],
                early_stopping_rounds=200,
                verbose_eval=False,
            )
            val_pred = booster.predict(xgb.DMatrix(encoded_train.iloc[val_idx]), iteration_range=(0, booster.best_iteration + 1))
            fold_test_pred = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
            update_predictions(oof, counts, test_pred, val_idx, val_pred, fold_test_pred)
            boosters.append(booster)
            total_models += 1
    oof, test_pred = finalize_predictions(oof, counts, test_pred, total_models)
    return {"name": "xgboost", "oof": oof, "test": test_pred, "models": boosters, "params": base_params}


def train_catboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: pd.Series,
    params: Dict,
    seeds: List[int],
    n_splits: int,
) -> Dict:
    if CatBoostClassifier is None:
        raise RuntimeError("CatBoost not installed")
    feature_cols, categorical_cols = feature_metadata(train)
    cat_indices = [feature_cols.index(col) for col in categorical_cols]
    base_params = {
        "iterations": 4000,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": False,
    }
    base_params.update(params)
    oof, counts, test_pred = average_oof_test(len(train), len(test), len(seeds) * n_splits)
    total_models = 0
    models = []
    for seed in seeds:
        folds = stratified_folds(n_splits, seed, train, target)
        for fold_id, (tr_idx, val_idx) in enumerate(folds):
            model = CatBoostClassifier(**base_params, random_seed=seed)
            model.fit(
                train.iloc[tr_idx][feature_cols],
                target.iloc[tr_idx],
                eval_set=(train.iloc[val_idx][feature_cols], target.iloc[val_idx]),
                cat_features=cat_indices,
                use_best_model=True,
            )
            val_pred = model.predict_proba(train.iloc[val_idx][feature_cols])[:, 1]
            fold_test_pred = model.predict_proba(test[feature_cols])[:, 1]
            update_predictions(oof, counts, test_pred, val_idx, val_pred, fold_test_pred)
            models.append(model)
            total_models += 1
    oof, test_pred = finalize_predictions(oof, counts, test_pred, total_models)
    return {"name": "catboost", "oof": oof, "test": test_pred, "models": models, "params": base_params}


def train_extratrees(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: pd.Series,
    params: Dict,
    seeds: List[int],
    n_splits: int,
) -> Dict:
    feature_cols, categorical_cols = feature_metadata(train)
    enc_train, enc_test = encode_categoricals(train[feature_cols], test[feature_cols], categorical_cols)
    base_params = {
        "n_estimators": params.get("n_estimators", 1000),
        "max_depth": params.get("max_depth", None),
        "max_features": params.get("max_features", 0.8),
        "min_samples_leaf": params.get("min_samples_leaf", 2),
        "n_jobs": -1,
    }
    oof, counts, test_pred = average_oof_test(len(train), len(test), len(seeds) * n_splits)
    total_models = 0
    models = []
    for seed in seeds:
        folds = stratified_folds(n_splits, seed, train, target)
        for fold_id, (tr_idx, val_idx) in enumerate(folds):
            model = ExtraTreesClassifier(**base_params, random_state=seed + fold_id)
            model.fit(enc_train.iloc[tr_idx], target.iloc[tr_idx])
            val_pred = model.predict_proba(enc_train.iloc[val_idx])[:, 1]
            fold_test_pred = model.predict_proba(enc_test)[:, 1]
            update_predictions(oof, counts, test_pred, val_idx, val_pred, fold_test_pred)
            models.append(model)
            total_models += 1
    oof, test_pred = finalize_predictions(oof, counts, test_pred, total_models)
    return {"name": "extratrees", "oof": oof, "test": test_pred, "models": models, "params": base_params}


def train_tabnet(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: pd.Series,
    params: Dict,
    seeds: List[int],
    n_splits: int,
) -> Dict:
    if TabNetClassifier is None:
        raise RuntimeError("TabNet not installed")
    feature_cols, _ = feature_metadata(train)
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])
    X_train = scaler.transform(train[feature_cols])
    X_test = scaler.transform(test[feature_cols])
    base_params = {
        "n_d": params.get("n_d", 24),
        "n_a": params.get("n_a", 24),
        "n_steps": params.get("n_steps", 5),
        "gamma": params.get("gamma", 1.2),
        "lambda_sparse": params.get("lambda_sparse", 1e-4),
    }
    oof, counts, test_pred = average_oof_test(len(train), len(test), len(seeds) * n_splits)
    total_models = 0
    models = []
    for seed in seeds:
        folds = stratified_folds(n_splits, seed, train, target)
        for fold_id, (tr_idx, val_idx) in enumerate(folds):
            clf = TabNetClassifier(**base_params, seed=seed + fold_id, verbose=0)
            clf.fit(
                X_train[tr_idx],
                target.iloc[tr_idx].values,
                eval_set=[(X_train[val_idx], target.iloc[val_idx].values)],
                patience=30,
                max_epochs=400,
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False,
            )
            val_pred = clf.predict_proba(X_train[val_idx])[:, 1]
            fold_test_pred = clf.predict_proba(X_test)[:, 1]
            update_predictions(oof, counts, test_pred, val_idx, val_pred, fold_test_pred)
            models.append(clf)
            total_models += 1
    oof, test_pred = finalize_predictions(oof, counts, test_pred, total_models)
    return {"name": "tabnet", "oof": oof, "test": test_pred, "models": models, "params": base_params}


class FEMCNN(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x).squeeze(1)


def numpy_to_loader(x: np.ndarray, y: np.ndarray | None, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_x = torch.from_numpy(x)
    if y is not None:
        tensor_y = torch.from_numpy(y.astype(np.float32))
        dataset = TensorDataset(tensor_x, tensor_y)
    else:
        dataset = TensorDataset(tensor_x)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_fem_cnn(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: pd.Series,
    params: Dict,
    seeds: List[int],
    n_splits: int,
) -> Dict:
    p_cols = [c for c in train.columns if c.startswith("p")]
    if len(p_cols) != 256:
        raise RuntimeError("FEM CNN expects 256 pressure columns (p0~p255)")
    p_cols = sorted(p_cols, key=lambda x: int(x[1:]))
    height = width = 16
    train_arr = train[p_cols].to_numpy(dtype=np.float32).reshape(-1, 1, height, width)
    test_arr = test[p_cols].to_numpy(dtype=np.float32).reshape(-1, 1, height, width)
    mean = train_arr.mean()
    std = train_arr.std() + 1e-6
    train_arr = (train_arr - mean) / std
    test_arr = (test_arr - mean) / std
    max_epochs = params.get("epochs", 60)
    batch_size = params.get("batch_size", 64)
    learning_rate = params.get("learning_rate", 1e-3)
    weight_decay = params.get("weight_decay", 1e-4)
    patience = params.get("patience", 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof, counts, test_pred = average_oof_test(len(train), len(test), len(seeds) * n_splits)
    total_models = 0
    target_arr = target.to_numpy(dtype=np.float32)
    criterion = nn.BCEWithLogitsLoss()
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        folds = stratified_folds(n_splits, seed, train, target)
        for fold_id, (tr_idx, val_idx) in enumerate(folds):
            model = FEMCNN(dropout=params.get("dropout", 0.3)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            train_loader = numpy_to_loader(train_arr[tr_idx], target_arr[tr_idx], batch_size, shuffle=True)
            val_loader = numpy_to_loader(train_arr[val_idx], target_arr[val_idx], batch_size, shuffle=False)
            test_loader = numpy_to_loader(test_arr, None, batch_size, shuffle=False)
            best_auc = 0.0
            best_state = None
            patience_counter = 0
            for epoch in range(max_epochs):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                model.eval()
                val_logits = []
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, labels = batch[0].to(device), batch[1].to(device)
                        logits = model(inputs)
                        val_logits.append(torch.sigmoid(logits).cpu().numpy())
                val_pred = np.concatenate(val_logits)
                val_auc = roc_auc_score(target_arr[val_idx], val_pred)
                if val_auc > best_auc + 1e-4:
                    best_auc = val_auc
                    best_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                val_preds = []
                for batch in val_loader:
                    inputs = batch[0].to(device)
                    logits = model(inputs)
                    val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_pred = np.concatenate(val_preds)
            fold_test_pred = []
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch[0].to(device)
                    logits = model(inputs)
                    fold_test_pred.append(torch.sigmoid(logits).cpu().numpy())
            fold_test_pred = np.concatenate(fold_test_pred)
            update_predictions(oof, counts, test_pred, val_idx, val_pred, fold_test_pred)
            total_models += 1
    oof, test_pred = finalize_predictions(oof, counts, test_pred, total_models)
    return {"name": "fem_cnn", "oof": oof, "test": test_pred, "params": {"mean": float(mean), "std": float(std)}}


MODEL_FUNCS = {
    "lightgbm": train_lightgbm,
    "xgboost": train_xgboost,
    "catboost": train_catboost,
    "extratrees": train_extratrees,
    "tabnet": train_tabnet,
    "fem_cnn": train_fem_cnn,
}


def train_base_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: pd.Series,
    seeds: List[int],
    n_splits: int,
    n_trials: int,
    skip_tuning: bool,
    enable_tabnet: bool,
    enable_cnn: bool,
    models: List[str],
) -> List[Dict]:
    best_params: Dict[str, Dict] = {}
    best_param_path = OUTPUT_DIR / "best_params.json"
    if best_param_path.exists():
        best_params = json.loads(best_param_path.read_text())

    models_to_train = [m for m in models if m in MODEL_FUNCS]
    if enable_tabnet and "tabnet" not in models_to_train:
        models_to_train.append("tabnet")
    if enable_cnn and "fem_cnn" not in models_to_train:
        models_to_train.append("fem_cnn")

    trained_models = []
    for name in models_to_train:
        if not skip_tuning or name not in best_params:
            print(f"[Optuna] tuning {name} ({n_trials} trials)...")
            best_params[name] = tune_model(name, train, target, n_trials, n_splits)
            best_param_path.write_text(json.dumps(best_params, indent=2))
        params = best_params.get(name, {})
        print(f"Training {name} with params: {params}")
        trainer = MODEL_FUNCS[name]
        result = trainer(train.copy(), test.copy(), target, params, seeds, n_splits)
        trained_models.append(result)
        save_level1_predictions(result)
    return trained_models


def save_level1_predictions(result: Dict) -> None:
    name = result["name"]
    pd.DataFrame({"row_id": np.arange(len(result["oof"])), "probability": result["oof"]}).to_csv(
        OUTPUT_DIR / f"task1_level1_{name}_oof.csv", index=False
    )
    pd.DataFrame({"row_id": np.arange(len(result["test"])), "probability": result["test"]}).to_csv(
        OUTPUT_DIR / f"task1_level1_{name}_test.csv", index=False
    )


def stack_meta_model(
    base_models: List[Dict],
    target: pd.Series,
    seeds: List[int],
    n_splits: int,
    stacker: str,
) -> Dict:
    meta_train = pd.DataFrame({m["name"]: m["oof"] for m in base_models})
    meta_test = pd.DataFrame({m["name"]: m["test"] for m in base_models})
    oof = np.zeros(len(target))
    counts = np.zeros(len(target))
    test_pred = np.zeros(len(meta_test))
    total_models = 0
    feature_cols = meta_train.columns.tolist()
    models = []
    for seed in seeds:
        folds = stratified_folds(n_splits, seed, meta_train, target)
        for fold_id, (tr_idx, val_idx) in enumerate(folds):
            if stacker == "logistic":
                model = LogisticRegression(max_iter=500)
            else:
                if lgb is None:
                    raise RuntimeError("LightGBM not installed for stacker")
                model = lgb.LGBMClassifier(
                    n_estimators=1500,
                    learning_rate=0.03,
                    num_leaves=32,
                    subsample=0.9,
                    colsample_bytree=0.9,
                )
            model.fit(meta_train.iloc[tr_idx][feature_cols], target.iloc[tr_idx])
            val_pred = model.predict_proba(meta_train.iloc[val_idx][feature_cols])[:, 1]
            fold_test_pred = model.predict_proba(meta_test[feature_cols])[:, 1]
            update_predictions(oof, counts, test_pred, val_idx, val_pred, fold_test_pred)
            models.append(model)
            total_models += 1
    oof, test_pred = finalize_predictions(oof, counts, test_pred, total_models)
    pd.DataFrame({"row_id": np.arange(len(oof)), "probability": oof}).to_csv(
        OUTPUT_DIR / "task1_meta_oof.csv", index=False
    )
    pd.DataFrame({"row_id": np.arange(len(test_pred)), "probability": test_pred}).to_csv(
        OUTPUT_DIR / "task1_meta_test.csv", index=False
    )
    return {"name": "stacker", "oof": oof, "test": test_pred, "models": models}


def build_blend_results(base_models: List[Dict], modes: List[str]) -> List[Dict]:
    modes = [mode.strip().lower() for mode in modes if mode.strip()]
    if len(base_models) < 2 or not modes:
        return []
    data_oof = np.column_stack([m["oof"] for m in base_models])
    data_test = np.column_stack([m["test"] for m in base_models])
    results: List[Dict] = []
    n_samples = data_oof.shape[0]
    denom = max(n_samples - 1, 1)
    for mode in modes:
        if mode == "rank":
            ranks_oof = np.zeros_like(data_oof)
            ranks_test = np.zeros_like(data_test)
            for idx in range(data_oof.shape[1]):
                ranks_oof[:, idx] = rankdata(data_oof[:, idx], method="average")
                ranks_test[:, idx] = rankdata(data_test[:, idx], method="average")
            blend_oof = ((ranks_oof - 1) / denom).mean(axis=1)
            blend_test = ((ranks_test - 1) / max(data_test.shape[0] - 1, 1)).mean(axis=1)
            results.append({"name": "blend_rank", "oof": blend_oof, "test": blend_test})
        elif mode == "geometric":
            eps = 1e-9
            oof_geo = np.exp(np.mean(np.log(np.clip(data_oof, eps, 1 - eps)), axis=1))
            test_geo = np.exp(np.mean(np.log(np.clip(data_test, eps, 1 - eps)), axis=1))
            results.append({"name": "blend_geometric", "oof": oof_geo, "test": test_geo})
    return results


def calibration_platt(probs: np.ndarray, target: np.ndarray, test_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model = LogisticRegression(max_iter=200)
    probs = np.clip(probs, 1e-5, 1 - 1e-5)
    model.fit(probs.reshape(-1, 1), target)
    calibrated = model.predict_proba(probs.reshape(-1, 1))[:, 1]
    test_cal = model.predict_proba(test_probs.reshape(-1, 1))[:, 1]
    return calibrated, test_cal


def calibration_isotonic(probs: np.ndarray, target: np.ndarray, test_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, target)
    return iso.transform(probs), iso.transform(test_probs)


def calibration_temperature(probs: np.ndarray, target: np.ndarray, test_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eps = 1e-5
    logits = np.log(probs + eps) - np.log(1 - probs + eps)

    def nll(temp: float) -> float:
        scaled = logits / temp
        pred = 1 / (1 + np.exp(-scaled))
        return log_loss(target, pred)

    temps = np.linspace(0.5, 3.0, 50)
    best_temp = temps[np.argmin([nll(t) for t in temps])]
    calibrated = 1 / (1 + np.exp(-(logits / best_temp)))
    test_logits = np.log(test_probs + eps) - np.log(1 - test_probs + eps)
    test_calibrated = 1 / (1 + np.exp(-(test_logits / best_temp)))
    return calibrated, test_calibrated


CALIBRATION_FUNCS = {
    "platt": calibration_platt,
    "isotonic": calibration_isotonic,
    "temperature": calibration_temperature,
}


def apply_calibrations(oof: np.ndarray, test: np.ndarray, target: np.ndarray, metric: str) -> Dict:
    metrics = {}
    cal_predictions = {}
    cal_oof_store = {}
    for name, func in CALIBRATION_FUNCS.items():
        cal_oof, cal_test = func(oof.copy(), target.values, test.copy())
        auc = roc_auc_score(target, cal_oof)
        brier = brier_score_loss(target, cal_oof)
        metrics[name] = {"auc": auc, "brier": brier}
        cal_predictions[name] = cal_test
        cal_oof_store[name] = cal_oof
        pd.DataFrame({"row_id": np.arange(len(cal_oof)), "probability": cal_oof}).to_csv(
            OUTPUT_DIR / f"task1_calibrated_{name}_oof.csv", index=False
        )
        pd.DataFrame({"row_id": np.arange(len(cal_test)), "probability": cal_test}).to_csv(
            OUTPUT_DIR / f"task1_calibrated_{name}_test.csv", index=False
        )
    if len(cal_predictions) > 1:
        stack_names = list(cal_predictions.keys())
        oof_stack = np.vstack([cal_oof_store[n] for n in stack_names])
        test_stack = np.vstack([cal_predictions[n] for n in stack_names])
        ens_oof = oof_stack.mean(axis=0)
        ens_test = test_stack.mean(axis=0)
        ens_auc = roc_auc_score(target, ens_oof)
        ens_brier = brier_score_loss(target, ens_oof)
        metrics["mean_ensemble"] = {"auc": ens_auc, "brier": ens_brier}
        cal_predictions["mean_ensemble"] = ens_test
        cal_oof_store["mean_ensemble"] = ens_oof
    if metric == "auc":
        best_name = max(metrics.items(), key=lambda x: x[1]["auc"])[0]
    else:
        best_name = min(metrics.items(), key=lambda x: x[1]["brier"])[0]
    best_test = cal_predictions[best_name]
    best_oof = cal_oof_store[best_name]
    pd.DataFrame({"row_id": np.arange(len(best_test)), "probability": best_test}).to_csv(
        OUTPUT_DIR / "task1_ensemble_test_pred.csv", index=False
    )
    pd.DataFrame({"row_id": np.arange(len(best_oof)), "probability": best_oof}).to_csv(
        OUTPUT_DIR / "task1_ensemble_oof.csv", index=False
    )
    pd.DataFrame({"row_id": np.arange(len(oof)), "probability": oof}).to_csv(
        OUTPUT_DIR / "task1_ensemble_oof_raw.csv", index=False
    )
    metrics["selected"] = best_name
    with (OUTPUT_DIR / "task1_calibration_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    return {"best_method": best_name, "test_pred": best_test, "oof_pred": best_oof}


def save_submission(test_pred: np.ndarray, test_len: int) -> None:
    submission = pd.DataFrame({"ID": np.arange(test_len), "probability": test_pred})
    submission["decision"] = False
    submission.to_csv(OUTPUT_DIR / "task1_ensemble_submission.csv", index=False)


def save_metrics(
    base_models: List[Dict], stack_result: Dict, extra_results: List[Dict], target: pd.Series, selected_name: str
) -> None:
    metrics = {}
    for model in base_models:
        metrics[model["name"]] = float(roc_auc_score(target, model["oof"]))
    metrics["stacker"] = float(roc_auc_score(target, stack_result["oof"]))
    for result in extra_results:
        metrics[result["name"]] = float(roc_auc_score(target, result["oof"]))
    metrics["selected_ensemble"] = selected_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "task1_ensemble_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


def load_saved_predictions(models: List[str]) -> tuple[List[Dict], Dict]:
    if not models:
        return [], {}
    base_models = []
    for name in models:
        oof_path = OUTPUT_DIR / f"task1_level1_{name}_oof.csv"
        test_path = OUTPUT_DIR / f"task1_level1_{name}_test.csv"
        if not oof_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Missing level1 prediction files for {name}")
        oof = pd.read_csv(oof_path)["probability"].to_numpy()
        test = pd.read_csv(test_path)["probability"].to_numpy()
        base_models.append({"name": name, "oof": oof, "test": test})
    meta = {}
    meta_oof_path = OUTPUT_DIR / "task1_meta_oof.csv"
    meta_test_path = OUTPUT_DIR / "task1_meta_test.csv"
    if meta_oof_path.exists() and meta_test_path.exists():
        meta = {
            "name": "stacker",
            "oof": pd.read_csv(meta_oof_path)["probability"].to_numpy(),
            "test": pd.read_csv(meta_test_path)["probability"].to_numpy(),
        }
    return base_models, meta


def run(args: argparse.Namespace | None = None) -> Dict:
    args = args or parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    blend_modes = [m.strip() for m in args.blend_modes.split(",") if m.strip()]
    train, test = load_features()
    target = (train["Class"] == "NG").astype(int)

    if args.reuse_predictions:
        base_models, stack_result = load_saved_predictions(model_list)
        if not base_models or not stack_result:
            raise RuntimeError("Missing saved predictions. Run without --reuse_predictions first.")
    else:
        base_models = train_base_models(
            train,
            test,
            target,
            seeds=seeds,
            n_splits=args.n_splits,
            n_trials=args.n_trials,
            skip_tuning=args.skip_tuning,
            enable_tabnet=args.enable_tabnet,
            enable_cnn=args.enable_cnn,
            models=model_list,
        )
        stack_result = stack_meta_model(base_models, target, seeds, args.n_splits, args.stacker)
    blend_results = build_blend_results(base_models, blend_modes)
    best_candidate = stack_result
    best_name = stack_result["name"]
    best_auc = roc_auc_score(target, best_candidate["oof"])
    for result in blend_results:
        result_auc = roc_auc_score(target, result["oof"])
        result["auc"] = result_auc
        if result_auc > best_auc:
            best_auc = result_auc
            best_candidate = result
            best_name = result["name"]
    save_metrics(base_models, stack_result, blend_results, target, best_name)
    calibration_info = apply_calibrations(best_candidate["oof"], best_candidate["test"], target, args.calibration_metric)
    calibration_info["ensemble_source"] = best_name
    save_submission(calibration_info["test_pred"], len(test))
    return {
        "base_models": base_models,
        "stack_result": stack_result,
        "blend_results": blend_results,
        "calibration": calibration_info,
        "seeds": seeds,
        "selected_ensemble": best_name,
    }


def main() -> None:
    run()


if __name__ == "__main__":
    main()
