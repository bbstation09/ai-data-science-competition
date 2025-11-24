"""
final_ver22_safe_ensemble.py

Target: Safety First (Remove Overfitting Risk)
Strategy:
1. [DROP] XGBoost: Persistent overfitting (Train 1.0) -> Removed from ensemble.
2. [KEEP] CatBoost, ExtraTrees, RandomForest: More robust models.
3. [Pseudo] Adaptive Labeling: Still used for performance boost.
4. [Output] Correct Format (TRUE/FALSE)

- 리더보드 성능 : 0.318
- Selected (TRUE) : 106 / 466
- Best Ensemble AUC: 0.81805
- Max Train Profit: 4800
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from scipy.stats import skew, kurtosis
from scipy.fft import fft

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA 
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ================================================================
# Config
# ================================================================
DATASET_DIR = "dataset"
MODEL_DIR = "model_safe"
os.makedirs(MODEL_DIR, exist_ok=True)

N_FOLDS = 5
SEED = 42
USE_PSEUDO = True 
NUM_FEATS = 120

# ================================================================
# 1. Feature Engineering (Same)
# ================================================================
def get_derivative_features(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    cols = sorted(cols, key=lambda x: int(x[len(prefix):]))
    if len(cols) < 2: return df
    values = df[cols].values
    diff1 = np.diff(values, axis=1)
    df[f'{prefix}_diff_mean'] = np.nanmean(np.abs(diff1), axis=1)
    df[f'{prefix}_diff_max'] = np.nanmax(np.abs(diff1), axis=1)
    df[f'{prefix}_diff_std'] = np.nanstd(diff1, axis=1)
    threshold = np.nanmean(np.abs(diff1)) + 2 * np.nanstd(np.abs(diff1))
    df[f'{prefix}_jumps'] = np.sum(np.abs(diff1) > threshold, axis=1)
    return df

def extract_fft_features(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    if len(cols) < 10: return df
    values = df[cols].values
    fft_vals = np.abs(fft(values, axis=1))
    fft_vals = fft_vals[:, 1:] 
    df[f'{prefix}_fft_mean'] = np.nanmean(fft_vals, axis=1)
    df[f'{prefix}_fft_max'] = np.nanmax(fft_vals, axis=1)
    df[f'{prefix}_fft_energy'] = np.sum(fft_vals ** 2, axis=1)
    return df

def preprocess_data(df_raw):
    df = df_raw.copy()
    drop_cols = ["ID", "Class", "label"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    for col in ['Plant', 'Proc_Param6']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r'(\d+)').astype(float)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def add_features(df):
    df = df.copy()
    df = get_derivative_features(df, "x")
    df = get_derivative_features(df, "p")
    df = extract_fft_features(df, "x")
    df = extract_fft_features(df, "p")
    for prefix in ['x', 'p']:
        cols = [c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
        if len(cols) > 0:
            v = df[cols].values
            df[f'{prefix}_mean'] = np.nanmean(v, axis=1)
            df[f'{prefix}_std'] = np.nanstd(v, axis=1)
            df[f'{prefix}_range'] = np.nanmax(v, axis=1) - np.nanmin(v, axis=1)
            df[f'{prefix}_skew'] = np.nan_to_num(skew(v, axis=1, nan_policy='omit'))
            df[f'{prefix}_kurt'] = np.nan_to_num(kurtosis(v, axis=1, nan_policy='omit'))
    x_cols = [f'X{i}' for i in range(1, 6)]
    y_cols = [f'Y{i}' for i in range(1, 6)]
    if all(c in df.columns for c in x_cols + y_cols):
        df['CX'] = df[x_cols].mean(axis=1)
        df['CY'] = df[y_cols].mean(axis=1)
    return df

def add_pca_features(X_train, X_test):
    imp = SimpleImputer(strategy='median')
    X_tr_imp = imp.fit_transform(X_train)
    X_te_imp = imp.transform(X_test)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_imp)
    X_te_sc = scaler.transform(X_te_imp)
    pca = PCA(n_components=5, random_state=SEED)
    tr_pca = pca.fit_transform(X_tr_sc)
    te_pca = pca.transform(X_te_sc)
    for i in range(5):
        X_train[f'pca_{i}'] = tr_pca[:, i]
        X_test[f'pca_{i}'] = te_pca[:, i]
    return X_train, X_test

# ================================================================
# 2. Models (XGB REMOVED)
# ================================================================
def get_safe_models():
    # [Change] XGBoost removed due to overfitting risk
    models = {
        "Cat": CatBoostClassifier(iterations=2000, learning_rate=0.01, depth=5, # Shallow
                                  l2_leaf_reg=7, # Regularized
                                  subsample=0.8,
                                  auto_class_weights="Balanced", verbose=0, random_seed=SEED),
        
        "Extra": ExtraTreesClassifier(n_estimators=700, class_weight='balanced', 
                                      max_depth=10, # Limited Depth
                                      min_samples_leaf=5, 
                                      random_state=SEED, n_jobs=-1),
        
        "RF": RandomForestClassifier(n_estimators=700, class_weight='balanced', 
                                     max_depth=10, # Limited Depth
                                     min_samples_leaf=5, 
                                     random_state=SEED, n_jobs=-1)
    }
    return models

# ================================================================
# 3. Training
# ================================================================
def train_loop(X, y, X_test, desc="Round 1"):
    models = get_safe_models()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_dict = {}
    pred_dict = {}
    scores = {}
    
    print(f"\n>>> {desc}: Training Start")
    for name, model in models.items():
        oof = np.zeros(len(X))
        pred = np.zeros(len(X_test))
        train_aucs, valid_aucs = [], []
        
        for tr_i, va_i in skf.split(X, y):
            xt, xv = X.iloc[tr_i], X.iloc[va_i]
            yt, yv = y[tr_i], y[va_i]
            
            clf = model.__class__(**model.get_params())
            if name == "Cat": clf.fit(xt, yt, eval_set=(xv, yv), early_stopping_rounds=50, verbose=False)
            else: clf.fit(xt, yt)
            
            tr_p = clf.predict_proba(xt)[:, 1]
            va_p = clf.predict_proba(xv)[:, 1]
            
            train_aucs.append(roc_auc_score(yt, tr_p))
            valid_aucs.append(roc_auc_score(yv, va_p))
            
            oof[va_i] = va_p
            pred += clf.predict_proba(X_test)[:, 1] / N_FOLDS
            
        gap = np.mean(train_aucs) - np.mean(valid_aucs)
        print(f"    [{name}] Train: {np.mean(train_aucs):.4f} | Valid: {np.mean(valid_aucs):.4f} | Gap: {gap:.4f}")
        
        oof_dict[name] = oof
        pred_dict[name] = pred
        scores[name] = np.mean(valid_aucs)
        
    return oof_dict, pred_dict, scores

def smart_ensemble(oof_dict, pred_dict, scores, y):
    print("\n>>> Smart Ensemble Optimization")
    models = list(oof_dict.keys())
    best_ens_auc = 0
    best_weights = {}
    
    for _ in range(3000):
        w = np.random.dirichlet(np.ones(len(models)))
        for i, m in enumerate(models):
            if scores[m] < 0.75: w[i] *= 0.1
        w = w / w.sum()
        combined_oof = np.zeros(len(y))
        for i, m in enumerate(models):
            combined_oof += oof_dict[m] * w[i]
        auc = roc_auc_score(y, combined_oof)
        if auc > best_ens_auc:
            best_ens_auc = auc
            best_weights = {m: w[i] for i, m in enumerate(models)}
            
    print(f"    Best Ensemble AUC: {best_ens_auc:.5f}")
    print(f"    Weights: { {k: round(v, 3) for k, v in best_weights.items()} }")
    
    final_oof = np.zeros(len(y))
    final_pred = np.zeros(len(pred_dict[models[0]]))
    for m, w in best_weights.items():
        final_oof += oof_dict[m] * w
        final_pred += pred_dict[m] * w
    return final_oof, final_pred

# ================================================================
# Main Pipeline
# ================================================================
def run_safe_pipeline():
    print("========================================")
    print("   Project Safe: No XGBoost")
    print("========================================\n")
    
    train_raw = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
    test_raw = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))
    y = (train_raw["Class"] == "NG").astype(int).values
    
    print(">>> Step 1: Feature Engineering")
    X_train = add_features(preprocess_data(train_raw))
    X_test = add_features(preprocess_data(test_raw))
    common = sorted(list(set(X_train.columns) & set(X_test.columns)))
    X_train, X_test = X_train[common], X_test[common]
    
    nan_cols = [c for c in X_train.columns if X_train[c].isna().all()]
    X_train.drop(columns=nan_cols, inplace=True)
    X_test.drop(columns=nan_cols, errors='ignore', inplace=True)
    
    X_train, X_test = add_pca_features(X_train, X_test)
    imp = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)
    
    # Select
    print(f">>> Step 2: Feature Selection (Best {NUM_FEATS})")
    sel = ExtraTreesClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    sel.fit(X_train, y)
    imp_s = pd.Series(sel.feature_importances_, index=X_train.columns)
    top_cols = imp_s.sort_values(ascending=False).head(NUM_FEATS).index.tolist()
    X_train, X_test = X_train[top_cols], X_test[top_cols]
    
    spw = np.sum(y==0) / np.sum(y==1)

    # Round 1
    oof_r1, pred_r1, scores_r1 = train_loop(X_train, y, X_test, "Round 1")
    final_oof_r1, final_pred_r1 = smart_ensemble(oof_r1, pred_r1, scores_r1, y)
    
    # Round 2
    final_test_prob = final_pred_r1
    if USE_PSEUDO:
        print("\n>>> Step 3: Adaptive Pseudo Labeling")
        th_ng = np.percentile(final_pred_r1, 95)
        th_good = np.percentile(final_pred_r1, 20)
        high_conf_ng = np.where(final_pred_r1 >= th_ng)[0]
        high_conf_good = np.where(final_pred_r1 <= th_good)[0]
        print(f"    Adding {len(high_conf_ng)} NG + {len(high_conf_good)} Good")
        
        X_pseudo = X_test.iloc[np.concatenate([high_conf_ng, high_conf_good])]
        y_pseudo = np.concatenate([np.ones(len(high_conf_ng)), np.zeros(len(high_conf_good))])
        X_train_aug = pd.concat([X_train, X_pseudo], axis=0).reset_index(drop=True)
        y_train_aug = np.concatenate([y, y_pseudo])
        
        oof_r2, pred_r2, scores_r2 = train_loop(X_train_aug, y_train_aug, X_test, "Round 2")
        _, final_pred_r2 = smart_ensemble(oof_r2, pred_r2, scores_r2, y_train_aug)
        final_test_prob = final_pred_r2

    # Submission
    print("\n>>> Step 4: Submission Generation (TRUE=Select)")
    sorted_idx = np.argsort(final_oof_r1)
    sorted_y = y[sorted_idx]
    profits = (np.cumsum(sorted_y==0)*100) - (np.cumsum(sorted_y==1)*2000)
    best_idx = np.argmax(profits)
    
    # Safety Margin
    max_profit = profits[best_idx]
    safe_indices = np.where(profits >= max_profit * 0.98)[0]
    best_th = final_oof_r1[sorted_idx[safe_indices[0]]] if len(safe_indices) > 0 else final_oof_r1[sorted_idx[best_idx]]
    
    print(f"    Max Train Profit: {max_profit}")
    print(f"    Optimal Threshold: {best_th:.5f}")
    
    decision_numeric = (final_test_prob <= best_th).astype(int)
    decision_str = np.where(decision_numeric == 1, "TRUE", "FALSE")
    print(f"    Selected (TRUE): {decision_numeric.sum()} / {len(decision_numeric)}")
    
    ids = test_raw["ID"].values
    sub_L = pd.DataFrame({"ID": [f"{i}_L" for i in ids], "probability": final_test_prob, "decision": decision_str})
    sub_P = pd.DataFrame({"ID": [f"{i}_P" for i in ids], "probability": final_test_prob, "decision": decision_str})
    
    pd.concat([sub_L, sub_P]).to_csv("submission_safe.csv", index=False)
    print(">>> Saved 'submission_safe.csv'.")

if __name__ == "__main__":
    run_safe_pipeline()