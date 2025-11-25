"""
================================================================================
>>> Final Solution: Profit Maximization (Threshold vs Top-N)
================================================================================
[Step 2] Feature Engineering...

[Step 3] Training with Fixed Best Parameters...

>>> Final OOF AUC: 0.80397

[Step 4] Comparing Optimization Strategies...
   [Strategy 1] Threshold Optimization
     Best Threshold (P_Good): 0.8686
     Max Profit: 14,900 Won
     Selected: 149 (Good: 149, NG: 0)

   [Strategy 2] Top-N Rank Optimization
     Best N (Count): 149
     Max Profit: 14,900 Won
     Selected: 149 (Good: 149, NG: 0)

[Step 4] Comparing Optimization Strategies...
   [Strategy 1] Threshold Optimization
     Best Threshold (P_Good): 0.8686
     Max Profit: 14,900 Won
     Selected: 149 (Good: 149, NG: 0)

   [Strategy 2] Top-N Rank Optimization
     Best N (Count): 149
     Max Profit: 14,900 Won
     Selected: 149 (Good: 149, NG: 0)

[Step 5] Generating Submission based on Best Strategy...
   >>> Applying Top-N Strategy with Safety Margin
   >>> Optimal Count: 149 -> Safe Count: 146
   Final Test Selection Count: 146 / 200
"""


import pandas as pd
import numpy as np
import warnings
import os
import random
import lightgbm as lgb
import matplotlib.pyplot as plt

from scipy.fft import fft
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

# ============================================================================
# [Configuration]
# ============================================================================
SEEDS = [42, 2024, 91, 777, 1004]  # 5-Seed Ensemble
N_FOLDS = 5
FIXED_SEED = 42
PROFIT_GOOD = 100      # 정상 선택 시 이익
LOSS_NG = -2000        # 불량 선택 시 손실
MAX_SELECTION = 200    # 최대 선택 개수

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(FIXED_SEED)

print("="*80)
print(f">>> Final Solution: Profit Maximization (Threshold vs Top-N)")
print("="*80)

# 1. Data Loading
try:
    if os.path.exists("dataset/train.csv"):
        train_df = pd.read_csv("dataset/train.csv")
        test_df = pd.read_csv("dataset/test.csv")
    else:
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("!! Error: 데이터 파일을 찾을 수 없습니다.")
    raise

# ============================================================================
# 2. Feature Engineering (Ver 7 Logic - Best Intelligence)
# ============================================================================
print("[Step 2] Feature Engineering...")

def extract_fft_features(df, prefix='x'):
    cols = [c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    if len(cols) < 10: return df
    values = df[cols].values
    fft_vals = np.abs(fft(values, axis=1))
    
    df[f'{prefix}_Harmonic_1'] = fft_vals[:, 1]
    df[f'{prefix}_Harmonic_2'] = fft_vals[:, 2]
    df[f'{prefix}_Harmonic_3'] = fft_vals[:, 3]
    df[f'{prefix}_Harmonic_4'] = fft_vals[:, 4]
    df[f'{prefix}_fft_mean'] = np.nanmean(fft_vals[:, 1:64], axis=1)
    df[f'{prefix}_fft_max'] = np.nanmax(fft_vals[:, 1:64], axis=1)
    return df

def engineer_features(df_raw):
    df = df_raw.copy()
    
    p_cols = [f'p{i}' for i in range(256)]
    x_sim_cols = [f'x{i}' for i in range(256)]
    
    if all(c in df.columns for c in p_cols):
        p_vals = df[p_cols].values
        x_vals = df[x_sim_cols].values
        
        # Physics Features
        df['Contact_Node_Count'] = np.sum(p_vals > 0.01, axis=1)
        edge_idx = np.r_[0:26, 230:256]
        df['Edge_Pressure_Mean'] = np.mean(p_vals[:, edge_idx], axis=1)
        df['Pseudo_Moment_X'] = np.sum(x_vals * p_vals, axis=1)
        
        sum_p = np.sum(p_vals, axis=1) + 1e-6
        cop_x = np.sum(x_vals * p_vals, axis=1) / sum_p 
        width_sim = np.max(x_vals, axis=1) - np.min(x_vals, axis=1)
        geo_center_x = np.mean(x_vals, axis=1)
        df['Pressure_Asymmetry'] = np.abs((cop_x - geo_center_x) / (width_sim + 1e-6))
        df['P_End_Stability'] = np.std(p_vals[:, -10:], axis=1)
        
        x_diff = np.diff(x_vals, axis=1)
        df['X_Sim_Smoothness'] = np.mean(np.abs(x_diff), axis=1)

    # G Efficiency & Design
    g_cols = [f'G{i}' for i in range(1, 5)]
    if all(c in df.columns for c in g_cols) and 'Width' in df.columns:
        vol_approx = df['Width'] * df['Aspect'] * df['Inch']
        df['G_Efficiency'] = df[g_cols].sum(axis=1) / (vol_approx + 1e-6)
        df['G_Min'] = df[g_cols].min(axis=1)

    df['Design_Logical_Error'] = (df['Y1'] >= df['Y3']).astype(int)
    df['Width_Spec_Error_Ratio'] = df['Width'] / ((df['X5'] - df['X1']) + 1e-6)
    df['Design_X_Asymmetry'] = np.abs(np.abs(df['X1']) - np.abs(df['X5']))

    if 'Mass_Pilot' in df.columns:
        df['Mass_Pilot_Int'] = df['Mass_Pilot'].astype(str).map({'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype(int)
        if 'Width' in df.columns: df['Mass_Width'] = df['Mass_Pilot_Int'] * df['Width']
        if all(c in df.columns for c in p_cols):
             df['Mass_P_Std'] = df['Mass_Pilot_Int'] * np.std(p_vals, axis=1)

    df = extract_fft_features(df, 'x')

    drop_cols = ['ID', 'Class', 'label', 'Mass_Pilot'] + \
                [f'x{i}' for i in range(256)] + [f'p{i}' for i in range(256)]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    return df

X = engineer_features(train_df)
X_test = engineer_features(test_df)

if 'Class' in train_df.columns:
    y = pd.to_numeric(train_df['Class'].map({'Good': 0, 'NG': 1}), errors='coerce').fillna(0).astype(int).values
else:
    y = np.zeros(len(train_df))

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=cat_cols)
X_test = pd.get_dummies(X_test, columns=cat_cols)
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Anomaly Score
iso = IsolationForest(n_estimators=100, contamination='auto', random_state=FIXED_SEED, n_jobs=-1)
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
iso.fit(X[num_cols].fillna(0))
X['Anomaly_Score'] = -iso.score_samples(X[num_cols].fillna(0))
X_test['Anomaly_Score'] = -iso.score_samples(X_test[num_cols].fillna(0))

# Scale
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X)), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(imputer.transform(X_test)), columns=X_test.columns)

# ============================================================================
# 3. Training (Fixed Best Parameters)
# ============================================================================
print("\n[Step 3] Training with Fixed Best Parameters...")

# AUC 0.80781 파라미터 고정
cat_params = {
    'iterations': 1728, 'depth': 7, 'learning_rate': 0.07587, 'l2_leaf_reg': 1.7538,
    'auto_class_weights': 'Balanced', 'verbose': 0, 'allow_writing_files': False
}
lgb_params = {
    'n_estimators': 1484, 'learning_rate': 0.04458, 'num_leaves': 38, 'max_depth': 8,
    'reg_alpha': 0.366, 'reg_lambda': 0.360, 'class_weight': 'balanced', 'verbose': -1
}

final_oof = np.zeros(len(X))
final_test = np.zeros(len(X_test))

for seed in SEEDS:
    seed_everything(seed)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    current_cat_params = cat_params.copy()
    current_cat_params['random_seed'] = seed
    current_lgb_params = lgb_params.copy()
    current_lgb_params['random_state'] = seed
    
    seed_oof = np.zeros(len(X))
    seed_test = np.zeros(len(X_test))
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_scaled, y)):
        X_tr, X_va = X_scaled.iloc[tr_idx], X_scaled.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        cat = CatBoostClassifier(**current_cat_params)
        cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
        
        lgbm = lgb.LGBMClassifier(**current_lgb_params)
        lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        seed_oof[va_idx] = 0.5 * cat.predict_proba(X_va)[:, 1] + 0.5 * lgbm.predict_proba(X_va)[:, 1]
        seed_test += (0.5 * cat.predict_proba(X_test_scaled)[:, 1] + 0.5 * lgbm.predict_proba(X_test_scaled)[:, 1]) / N_FOLDS
        
    final_oof += seed_oof / len(SEEDS)
    final_test += seed_test / len(SEEDS)

print(f"\n>>> Final OOF AUC: {roc_auc_score(y, final_oof):.5f}")

# ============================================================================
# 4. Profit Optimization (Strategy Comparison)
# ============================================================================
print("\n[Step 4] Comparing Optimization Strategies...")

def compute_profit(decision, target):
    good_sel = int((decision & (target == 0)).sum())
    ng_sel = int((decision & (target == 1)).sum())
    profit = good_sel * PROFIT_GOOD + ng_sel * LOSS_NG
    return profit, ng_sel, good_sel

# prob_ng (불량 확률) -> prob_good (정상 확률) 변환
prob_good_oof = 1.0 - final_oof
prob_good_test = 1.0 - final_test

# --- Strategy 1: Threshold Search (Conservative) ---
# 불량 선택 시 페널티가 크므로 0.01 ~ 0.2 구간(불량 확률 기준)을 촘촘히 봅니다.
# 즉, 정상 확률(prob_good) 기준으로는 0.8 ~ 0.99 구간
best_thr_profit = -np.inf
best_thr_val = 0.0
thresholds = np.linspace(0.80, 0.999, 500) # 매우 촘촘하게

for thr in thresholds:
    decision = prob_good_oof >= thr
    # Max 200 제한
    if decision.sum() > MAX_SELECTION:
        # 확률 높은 순으로 자름
        top_idx = np.argsort(prob_good_oof)[::-1][:MAX_SELECTION]
        decision = np.zeros_like(decision, dtype=bool)
        decision[top_idx] = True
        
    profit, ng, good = compute_profit(decision, y)
    if profit > best_thr_profit:
        best_thr_profit = profit
        best_thr_val = thr
        best_thr_stats = (ng, good, decision.sum())

print(f"   [Strategy 1] Threshold Optimization")
print(f"     Best Threshold (P_Good): {best_thr_val:.4f}")
print(f"     Max Profit: {best_thr_profit:,} Won")
print(f"     Selected: {best_thr_stats[2]} (Good: {best_thr_stats[1]}, NG: {best_thr_stats[0]})")

# --- Strategy 2: Top-N Search (Rank Based) ---
# 가장 자신 있는 순서대로 줄 세워서 어디서 끊을지 결정
best_topn_profit = -np.inf
best_n_val = 0
order_oof = np.argsort(prob_good_oof)[::-1] # 내림차순 (확률 높은 순)

# 50개부터 200개까지 하나씩 늘려가며 확인
for n in range(50, 201):
    idx = order_oof[:n]
    decision = np.zeros_like(prob_good_oof, dtype=bool)
    decision[idx] = True
    
    profit, ng, good = compute_profit(decision, y)
    if profit > best_topn_profit:
        best_topn_profit = profit
        best_n_val = n
        best_topn_stats = (ng, good, decision.sum())

print(f"\n   [Strategy 2] Top-N Rank Optimization")
print(f"     Best N (Count): {best_n_val}")
print(f"     Max Profit: {best_topn_profit:,} Won")
print(f"     Selected: {best_topn_stats[2]} (Good: {best_topn_stats[1]}, NG: {best_topn_stats[0]})")
# ... (앞부분은 그대로 유지) ...

# ============================================================================
# 4. Profit Optimization (Strategy Comparison)
# ============================================================================
print("\n[Step 4] Comparing Optimization Strategies...")

def compute_profit(decision, target):
    good_sel = int((decision & (target == 0)).sum())
    ng_sel = int((decision & (target == 1)).sum())
    profit = good_sel * PROFIT_GOOD + ng_sel * LOSS_NG
    return profit, ng_sel, good_sel

# prob_ng (불량 확률) -> prob_good (정상 확률) 변환
prob_good_oof = 1.0 - final_oof
prob_good_test = 1.0 - final_test

# --- Strategy 1: Threshold Search ---
best_thr_profit = -np.inf
best_thr_val = 0.0
thresholds = np.linspace(0.80, 0.999, 500)

for thr in thresholds:
    decision = prob_good_oof >= thr
    if decision.sum() > MAX_SELECTION:
        top_idx = np.argsort(prob_good_oof)[::-1][:MAX_SELECTION]
        decision = np.zeros_like(decision, dtype=bool)
        decision[top_idx] = True
        
    profit, ng, good = compute_profit(decision, y)
    if profit > best_thr_profit:
        best_thr_profit = profit
        best_thr_val = thr
        best_thr_stats = (ng, good, decision.sum())

print(f"   [Strategy 1] Threshold Optimization")
print(f"     Best Threshold (P_Good): {best_thr_val:.4f}")
print(f"     Max Profit: {best_thr_profit:,} Won")
print(f"     Selected: {best_thr_stats[2]} (Good: {best_thr_stats[1]}, NG: {best_thr_stats[0]})")

# --- Strategy 2: Top-N Search (Rank Based) ---
best_topn_profit = -np.inf
best_n_val = 0
order_oof = np.argsort(prob_good_oof)[::-1] # 내림차순 (확률 높은 순)

for n in range(50, 201):
    idx = order_oof[:n]
    decision = np.zeros_like(prob_good_oof, dtype=bool)
    decision[idx] = True
    
    profit, ng, good = compute_profit(decision, y)
    if profit > best_topn_profit:
        best_topn_profit = profit
        best_n_val = n
        best_topn_stats = (ng, good, decision.sum())

print(f"\n   [Strategy 2] Top-N Rank Optimization")
print(f"     Best N (Count): {best_n_val}")
print(f"     Max Profit: {best_topn_profit:,} Won")
print(f"     Selected: {best_topn_stats[2]} (Good: {best_topn_stats[1]}, NG: {best_topn_stats[0]})")

# ============================================================================
# 5. Final Decision & Submission
# ============================================================================
print("\n[Step 5] Generating Submission based on Best Strategy...")

# Top-N 방식 사용 (가장 안정적)
optimal_count = best_n_val

# [Safety Margin] 최적점의 98% 수준에서 컷 (FP 방지)
safe_count = int(optimal_count * 0.98)

print(f"   >>> Applying Top-N Strategy with Safety Margin")
print(f"   >>> Optimal Count: {optimal_count} -> Safe Count: {safe_count}")

# Test 데이터 적용
test_result = pd.DataFrame({'ID': test_df['ID'], 'prob_good': prob_good_test})
test_result['original_idx'] = test_result.index

# 정상 확률 높은 순 정렬
test_result = test_result.sort_values('prob_good', ascending=False) 

# 상위 safe_count 개수 선택
test_result['decision'] = 0
test_result.iloc[:safe_count, test_result.columns.get_loc('decision')] = 1

print(f"   Final Test Selection Count: {test_result['decision'].sum()} / 200")

# 저장 (원래 순서 복원)
test_result = test_result.sort_values('original_idx')
decision_str = test_result['decision'].apply(lambda x: 'TRUE' if x==1 else 'FALSE')

# 제출 파일에는 불량 확률(prob_ng)을 넣는 것이 일반적
prob_ng_submit = 1.0 - test_result['prob_good']

sub_l = pd.DataFrame({'ID': [f"{i}_L" for i in test_df['ID']], 'probability': prob_ng_submit, 'decision': decision_str})
sub_p = pd.DataFrame({'ID': [f"{i}_P" for i in test_df['ID']], 'probability': prob_ng_submit, 'decision': decision_str})

filename = f"submission_final_profit_max_{safe_count}.csv"
pd.concat([sub_l, sub_p], axis=0, ignore_index=True).to_csv(filename, index=False)
print(f">>> Saved '{filename}' successfully.")