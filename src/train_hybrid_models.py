#!/usr/bin/env python
"""
Custom hybrid training script:
- LightGBM/XGBoost/CatBoost + TabNet + FEM CNN
- Multi-seed Stratified KFold CV
- Logistic stacking + Isotonic calibration
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception:
    TabNetClassifier = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
BEST_PARAM_PATH = OUTPUT_DIR / "best_params.json"


def load_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(OUTPUT_DIR / "train_features.csv")
    test = pd.read_csv(OUTPUT_DIR / "test_features.csv")
    return train, test


def stratified_indices(n_splits: int, seed: int, X: pd.DataFrame, y: pd.Series):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(X, y))


def average_predictions(n_samples: int, test_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    oof = np.zeros(n_samples)
    counts = np.zeros(n_samples)
    test_pred = np.zeros(test_len)
    return oof, counts, test_pred


def finalize_predictions(oof: np.ndarray, counts: np.ndarray, test_pred: np.ndarray, total_models: int) -> Tuple[np.ndarray, np.ndarray]:
    oof = np.divide(oof, counts, out=np.zeros_like(oof), where=counts > 0)
    test_pred = test_pred / max(total_models, 1)
    return oof, test_pred


def load_best_params() -> Dict[str, Dict]:
    if BEST_PARAM_PATH.exists():
        return json.loads(BEST_PARAM_PATH.read_text())
    return {}


def train_lightgbm_cv(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series, seeds: List[int], n_splits: int) -> Dict:
    if lgb is None:
        raise RuntimeError("LightGBM not available")
    params = load_best_params().get(
        "lightgbm",
        {
            "learning_rate": 0.03,
            "num_leaves": 96,
            "min_child_samples": 40,
            "subsample": 0.85,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.05,
            "reg_lambda": 0.1,
        },
    )
    feature_cols = [c for c in train.columns if c != "Class"]
    categorical_cols = [c for c in feature_cols if train[c].dtype == "object"]
    for df in [train, test]:
        for col in categorical_cols:
            df[col] = df[col].astype("category")
    base_params = dict(
        objective="binary",
        n_estimators=4000,
        n_jobs=-1,
        boosting_type="gbdt",
    )
    base_params.update(params)
    oof, counts, test_pred = average_predictions(len(train), len(test))
    total = 0
    for seed in seeds:
        folds = stratified_indices(n_splits, seed, train, target)
        for tr_idx, val_idx in folds:
            model = lgb.LGBMClassifier(**base_params, random_state=seed)
            model.fit(
                train.iloc[tr_idx][feature_cols],
                target.iloc[tr_idx],
                eval_set=[(train.iloc[val_idx][feature_cols], target.iloc[val_idx])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
            )
            val_pred = model.predict_proba(train.iloc[val_idx][feature_cols])[:, 1]
            fold_test = model.predict_proba(test[feature_cols])[:, 1]
            oof[val_idx] += val_pred
            counts[val_idx] += 1
            test_pred += fold_test
            total += 1
    oof, test_pred = finalize_predictions(oof, counts, test_pred, total)
    return {"name": "lightgbm", "oof": oof, "test": test_pred}


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_enc = train.copy()
    test_enc = test.copy()
    for col in categorical_cols:
        codes, uniques = pd.factorize(pd.concat([train_enc[col], test_enc[col]], axis=0))
        train_enc[col] = codes[: len(train_enc)]
        test_enc[col] = codes[len(train_enc) :]
    return train_enc, test_enc


def train_xgboost_cv(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series, seeds: List[int], n_splits: int) -> Dict:
    if xgb is None:
        raise RuntimeError("xgboost not available")
    params = load_best_params().get(
        "xgboost",
        {
            "learning_rate": 0.02,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 5.0,
            "lambda": 0.2,
            "alpha": 0.1,
        },
    )
    feature_cols = [c for c in train.columns if c != "Class"]
    categorical_cols = [c for c in feature_cols if train[c].dtype == "object"]
    enc_train, enc_test = encode_categoricals(train[feature_cols], test[feature_cols], categorical_cols)
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "min_child_weight": params["min_child_weight"],
        "lambda": params["lambda"],
        "alpha": params["alpha"],
    }
    oof, counts, test_pred = average_predictions(len(train), len(test))
    dtest = xgb.DMatrix(enc_test)
    total = 0
    for seed in seeds:
        folds = stratified_indices(n_splits, seed, train, target)
        for tr_idx, val_idx in folds:
            dtrain = xgb.DMatrix(enc_train.iloc[tr_idx], label=target.iloc[tr_idx])
            dval = xgb.DMatrix(enc_train.iloc[val_idx], label=target.iloc[val_idx])
            booster = xgb.train(
                dict(base_params, seed=seed),
                dtrain,
                num_boost_round=4000,
                evals=[(dval, "valid")],
                verbose_eval=False,
                early_stopping_rounds=200,
            )
            val_pred = booster.predict(xgb.DMatrix(enc_train.iloc[val_idx]), iteration_range=(0, booster.best_iteration + 1))
            fold_test = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
            oof[val_idx] += val_pred
            counts[val_idx] += 1
            test_pred += fold_test
            total += 1
    oof, test_pred = finalize_predictions(oof, counts, test_pred, total)
    return {"name": "xgboost", "oof": oof, "test": test_pred}


def train_catboost_cv(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series, seeds: List[int], n_splits: int) -> Dict:
    if CatBoostClassifier is None:
        raise RuntimeError("CatBoost not available")
    params = load_best_params().get(
        "catboost",
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0},
    )
    feature_cols = [c for c in train.columns if c != "Class"]
    categorical_cols = [c for c in feature_cols if train[c].dtype == "object"]
    cat_indices = [feature_cols.index(col) for col in categorical_cols]
    base_params = dict(iterations=4000, loss_function="Logloss", eval_metric="AUC", verbose=False)
    base_params.update(params)
    oof, counts, test_pred = average_predictions(len(train), len(test))
    total = 0
    for seed in seeds:
        folds = stratified_indices(n_splits, seed, train, target)
        for tr_idx, val_idx in folds:
            model = CatBoostClassifier(**base_params, random_seed=seed)
            model.fit(
                train.iloc[tr_idx][feature_cols],
                target.iloc[tr_idx],
                eval_set=(train.iloc[val_idx][feature_cols], target.iloc[val_idx]),
                cat_features=cat_indices,
                use_best_model=True,
            )
            val_pred = model.predict_proba(train.iloc[val_idx][feature_cols])[:, 1]
            fold_test = model.predict_proba(test[feature_cols])[:, 1]
            oof[val_idx] += val_pred
            counts[val_idx] += 1
            test_pred += fold_test
            total += 1
    oof, test_pred = finalize_predictions(oof, counts, test_pred, total)
    return {"name": "catboost", "oof": oof, "test": test_pred}


def train_tabnet_cv(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series, seeds: List[int], n_splits: int) -> Dict:
    if TabNetClassifier is None:
        raise RuntimeError("TabNet not available")
    feature_cols = [c for c in train.columns if c != "Class"]
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])
    X_train = scaler.transform(train[feature_cols])
    X_test = scaler.transform(test[feature_cols])
    base_params = dict(n_d=24, n_a=24, n_steps=5, gamma=1.2, lambda_sparse=1e-4)
    oof, counts, test_pred = average_predictions(len(train), len(test))
    total = 0
    for seed in seeds:
        folds = stratified_indices(n_splits, seed, train, target)
        for fold_id, (tr_idx, val_idx) in enumerate(folds):
            clf = TabNetClassifier(**base_params, seed=seed + fold_id, verbose=0)
            clf.fit(
                X_train[tr_idx],
                target.iloc[tr_idx].values,
                eval_set=[(X_train[val_idx], target.iloc[val_idx].values)],
                patience=20,
                max_epochs=300,
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
            )
            val_pred = clf.predict_proba(X_train[val_idx])[:, 1]
            fold_test = clf.predict_proba(X_test)[:, 1]
            oof[val_idx] += val_pred
            counts[val_idx] += 1
            test_pred += fold_test
            total += 1
    oof, test_pred = finalize_predictions(oof, counts, test_pred, total)
    return {"name": "tabnet", "oof": oof, "test": test_pred}


class FEMCNN(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
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


def numpy_loader(x: np.ndarray, y: np.ndarray | None, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_x = torch.from_numpy(x)
    if y is not None:
        tensor_y = torch.from_numpy(y.astype(np.float32))
        dataset = TensorDataset(tensor_x, tensor_y)
    else:
        dataset = TensorDataset(tensor_x)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_fem_cnn_cv(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series, seeds: List[int], n_splits: int) -> Dict:
    p_cols = sorted([c for c in train.columns if c.startswith("p")], key=lambda x: int(x[1:]))
    if len(p_cols) != 256:
        raise RuntimeError("Expected 256 FEM columns (p0~p255)")
    H = W = 16
    train_arr = train[p_cols].to_numpy(dtype=np.float32).reshape(-1, 1, H, W)
    test_arr = test[p_cols].to_numpy(dtype=np.float32).reshape(-1, 1, H, W)
    mean = train_arr.mean()
    std = train_arr.std() + 1e-6
    train_arr = (train_arr - mean) / std
    test_arr = (test_arr - mean) / std
    target_arr = target.to_numpy(dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs = 60
    batch_size = 64
    patience = 10
    oof, counts, test_pred = average_predictions(len(train), len(test))
    total = 0
    criterion = nn.BCEWithLogitsLoss()
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        folds = stratified_indices(n_splits, seed, train, target)
        for fold_id, (tr_idx, val_idx) in enumerate(folds):
            model = FEMCNN(dropout=0.3).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            train_loader = numpy_loader(train_arr[tr_idx], target_arr[tr_idx], batch_size, shuffle=True)
            val_loader = numpy_loader(train_arr[val_idx], target_arr[val_idx], batch_size, shuffle=False)
            test_loader = numpy_loader(test_arr, None, batch_size, shuffle=False)
            best_auc = 0.0
            best_state = None
            no_improve = 0
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
                val_preds = []
                with torch.no_grad():
                    for batch in val_loader:
                        inputs = batch[0].to(device)
                        logits = model(inputs)
                        val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_pred = np.concatenate(val_preds)
                val_auc = roc_auc_score(target_arr[val_idx], val_pred)
                if val_auc > best_auc + 1e-4:
                    best_auc = val_auc
                    best_state = model.state_dict()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            val_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0].to(device)
                    logits = model(inputs)
                    val_preds.append(torch.sigmoid(logits).cpu().numpy())
            val_pred = np.concatenate(val_preds)
            test_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch[0].to(device)
                    logits = model(inputs)
                    test_preds.append(torch.sigmoid(logits).cpu().numpy())
            fold_test = np.concatenate(test_preds)
            oof[val_idx] += val_pred
            counts[val_idx] += 1
            test_pred += fold_test
            total += 1
    oof, test_pred = finalize_predictions(oof, counts, test_pred, total)
    return {"name": "fem_cnn", "oof": oof, "test": test_pred}


def stack_predictions(base_models: List[Dict], target: pd.Series) -> Dict:
    meta_train = pd.DataFrame({m["name"]: m["oof"] for m in base_models})
    meta_test = pd.DataFrame({m["name"]: m["test"] for m in base_models})
    oof = np.zeros(len(target))
    counts = np.zeros(len(target))
    test_pred = np.zeros(len(meta_test))
    seeds = [111, 222]
    n_splits = 5
    for seed in seeds:
        folds = stratified_indices(n_splits, seed, meta_train, target)
        for tr_idx, val_idx in folds:
            model = LogisticRegression(max_iter=500)
            model.fit(meta_train.iloc[tr_idx], target.iloc[tr_idx])
            val_pred = model.predict_proba(meta_train.iloc[val_idx])[:, 1]
            fold_test = model.predict_proba(meta_test)[:, 1]
            oof[val_idx] += val_pred
            counts[val_idx] += 1
            test_pred += fold_test
    oof, test_pred = finalize_predictions(oof, counts, test_pred, len(seeds) * n_splits)
    return {"name": "stacker", "oof": oof, "test": test_pred}


def apply_calibration(oof: np.ndarray, test_pred: np.ndarray, target: pd.Series) -> Dict:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, target.values)
    iso_oof = iso.transform(oof)
    iso_test = iso.transform(test_pred)
    metrics = {
        "isotonic": {"auc": roc_auc_score(target, iso_oof), "brier": brier_score_loss(target, iso_oof)},
        "stack_raw": {"auc": roc_auc_score(target, oof), "brier": brier_score_loss(target, oof)},
    }
    selected = "isotonic"
    best_pred = iso_test
    pd.DataFrame({"row_id": np.arange(len(best_pred)), "probability": best_pred}).to_csv(
        OUTPUT_DIR / "task1_ensemble_test_pred.csv", index=False
    )
    pd.DataFrame({"row_id": np.arange(len(iso_oof)), "probability": iso_oof}).to_csv(
        OUTPUT_DIR / "task1_ensemble_oof.csv", index=False
    )
    pd.DataFrame({"row_id": np.arange(len(oof)), "probability": oof}).to_csv(
        OUTPUT_DIR / "task1_ensemble_oof_raw.csv", index=False
    )
    with (OUTPUT_DIR / "task1_calibration_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump({"selected": selected, **metrics}, fp, indent=2)
    return {"oof": iso_oof, "test": iso_test, "selected": selected}


def build_submission(test_pred: np.ndarray, test_len: int) -> None:
    submission = pd.DataFrame({"ID": np.arange(test_len), "probability": test_pred})
    submission["decision"] = False
    submission.to_csv(OUTPUT_DIR / "task1_ensemble_submission.csv", index=False)


def save_level1(base_models: List[Dict]) -> None:
    for model in base_models:
        pd.DataFrame({"row_id": np.arange(len(model["oof"])), "probability": model["oof"]}).to_csv(
            OUTPUT_DIR / f"task1_level1_{model['name']}_oof.csv", index=False
        )
        pd.DataFrame({"row_id": np.arange(len(model["test"])), "probability": model["test"]}).to_csv(
            OUTPUT_DIR / f"task1_level1_{model['name']}_test.csv", index=False
        )


def main() -> None:
    print("[Hybrid] Starting run...", flush=True)
    train, test = load_features()
    target = (train["Class"] == "NG").astype(int)
    seeds = [111, 222]
    n_splits = 5
    base_models = []
    print("[Hybrid] Training LightGBM...", flush=True)
    base_models.append(train_lightgbm_cv(train.copy(), test.copy(), target, seeds, n_splits))
    print("[Hybrid] Training XGBoost...", flush=True)
    base_models.append(train_xgboost_cv(train.copy(), test.copy(), target, seeds, n_splits))
    print("[Hybrid] Training CatBoost...", flush=True)
    base_models.append(train_catboost_cv(train.copy(), test.copy(), target, seeds, n_splits))
    if TabNetClassifier is not None:
        print("[Hybrid] Training TabNet...", flush=True)
        base_models.append(train_tabnet_cv(train.copy(), test.copy(), target, seeds, n_splits))
    print("[Hybrid] Training FEM CNN...", flush=True)
    base_models.append(train_fem_cnn_cv(train.copy(), test.copy(), target, seeds, n_splits))
    save_level1(base_models)
    print("[Hybrid] Stacking meta model...", flush=True)
    stack_result = stack_predictions(base_models, target)
    all_models = base_models + [stack_result]
    metrics = {m["name"]: float(roc_auc_score(target, m["oof"])) for m in base_models}
    metrics["stacker"] = float(roc_auc_score(target, stack_result["oof"]))
    with (OUTPUT_DIR / "task1_ensemble_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print("[Hybrid] Calibration...", flush=True)
    cal = apply_calibration(stack_result["oof"], stack_result["test"], target)
    build_submission(cal["test"], len(test))
    print("[Hybrid] Completed.", flush=True)


if __name__ == "__main__":
    main()
