'''
================================================================================
>>> Final Fusion Solution: Physics Features + Seed Ensemble + Profit Maximization
================================================================================

[Step 1] Loading Data...
[Step 2] Feature Engineering (Merging Best Features)...
Final Features: 315

[Step 3] Training with Seed Ensemble (CatBoost + LightGBM)...
  Processing Seed: 42
  Processing Seed: 2024
  Processing Seed: 91

>>> Final Seed-Ensemble OOF AUC: 0.79384

[Step 4] Optimizing Selection Count (Cumulative Profit)...
   Peak Profit Validation: 13,300 Won
   Optimal Selection Count (Valid): 133
   >>> Simulation [Top 130]: Profit 13,000, Score 0.61806
   (Good Selected: 130, Bad Selected: 0)

[Step 5] Generating Submission...
   Final Test Selection: 130 / 200
>>> Saved 'submission_1123_ver3.csv' successfully.
'''



import pandas as pd
import numpy as np
import warnings
import os
import joblib
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt

from scipy.fft import fft
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# [Configuration]
# ============================================================================
USE_OPTUNA = False      # 시간 여유 있으면 True (이미 튜닝된 값 적용됨)
N_TRIALS = 5            # Optuna 시도 횟수
N_FOLDS = 5             # 교차 검증 폴드 수
SEEDS = [42, 2024, 91]  # 시드 앙상블 (안정성 강화)
AGGRESSIVE_RATIO = 0.98 # 최적 개수 대비 98%만 선택 (안전 마진)

print("="*80)
print(">>> Final Fusion Solution: Physics Features + Seed Ensemble + Profit Maximization")
print("="*80)

# ============================================================================
# 1. Data Loading
# ============================================================================
print("\n[Step 1] Loading Data...")
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
# 2. Advanced Feature Engineering (Fusion)
# ============================================================================
print("[Step 2] Feature Engineering (Merging Best Features)...")

def extract_fft_features(df, prefix='x'):
    cols = [c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    if len(cols) < 10: return df
    values = df[cols].values
    # FFT (주파수 분석)
    fft_vals = np.abs(fft(values, axis=1))[:, 1:64]
    df[f'{prefix}_fft_mean'] = np.nanmean(fft_vals, axis=1)
    df[f'{prefix}_fft_max'] = np.nanmax(fft_vals, axis=1)
    return df

def engineer_features(df_raw, is_train=True):
    df = df_raw.copy()
    
    # ---------------------------------------------------------
    # A. Mass_Pilot Interaction
    # ---------------------------------------------------------
    if 'Mass_Pilot' in df.columns:
        df['Mass_Pilot_Int'] = df['Mass_Pilot'].astype(str).map({'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype(int)
        if 'Width' in df.columns: df['Mass_Width'] = df['Mass_Pilot_Int'] * df['Width']
        if 'Inch' in df.columns: df['Mass_Inch'] = df['Mass_Pilot_Int'] * df['Inch']

    # ---------------------------------------------------------
    # B. Simulation Data Segmentation & Smoothness (Upgraded)
    # ---------------------------------------------------------
    p_cols = [f'p{i}' for i in range(256)]
    x_sim_cols = [f'x{i}' for i in range(256)]
    y_sim_cols = [f'y{i}' for i in range(256)]
    
    if all(c in df.columns for c in p_cols):
        p_vals = df[p_cols].values
        x_vals = df[x_sim_cols].values
        y_vals = df[y_sim_cols].values

        # 1. 구간별 통계
        df['P_Early_Mean'] = np.mean(p_vals[:, :60], axis=1)
        df['P_Mid_Mean'] = np.mean(p_vals[:, 60:170], axis=1)
        df['P_Late_Mean'] = np.mean(p_vals[:, 170:], axis=1)
        df['P_End_Stability'] = np.std(p_vals[:, -10:], axis=1) 

        # 2. 비대칭성 (Asymmetry)
        sum_p = np.sum(p_vals, axis=1)
        sum_p[sum_p == 0] = 1e-6
        cop_x = np.sum(x_vals * p_vals, axis=1) / sum_p 
        geo_center_x = np.mean(x_vals, axis=1)
        width_sim = np.max(x_vals, axis=1) - np.min(x_vals, axis=1)
        df['Pressure_Asymmetry'] = np.abs((cop_x - geo_center_x) / (width_sim + 1e-6))
        
        # 3. [NEW] 매끄러움 (Smoothness) - EDA 2-4 반영
        # 인접한 노드 간의 차이(변화량)가 크면 표면이 거칠다는 뜻 (불량)
        x_diff = np.diff(x_vals, axis=1)
        y_diff = np.diff(y_vals, axis=1)
        df['X_Sim_Smoothness'] = np.mean(np.abs(x_diff), axis=1) # 변화량 평균
        df['Y_Sim_Smoothness'] = np.mean(np.abs(y_diff), axis=1) # 변화량 평균
        
        # Mass_Pilot 교호작용
        if 'Mass_Pilot_Int' in df.columns:
            df['Mass_P_Std'] = df['Mass_Pilot_Int'] * np.std(p_vals, axis=1)

    # ---------------------------------------------------------
    # C. Geometry & Sensor Features
    # ---------------------------------------------------------
    x_cols = [f'X{i}' for i in range(1, 6)]
    y_cols = [f'Y{i}' for i in range(1, 6)]
    if all(c in df.columns for c in x_cols + y_cols):
        df['Geo_X1_X5_Diff'] = np.abs(df['X1'] - df['X5'])
        df['Geo_Convexity'] = df['Y3'] - (df['Y1'] + df['Y5'])/2
    
    if 'Width' in df.columns and 'Inch' in df.columns:
        df['Design_Ratio'] = df['Width'] / (df['Inch'] + 1e-6)

    # [NEW] G Feature Analysis - EDA 3-1 반영
    # G값들이 낮거나 튀는 값이 불량과 연관됨 -> Min, Max 추가
    g_cols = [f'G{i}' for i in range(1, 5)]
    if all(c in df.columns for c in g_cols):
        df['G_Sum'] = df[g_cols].sum(axis=1)
        df['G_Std'] = df[g_cols].std(axis=1)
        df['G_Min'] = df[g_cols].min(axis=1) # 낮은 값 탐지

    # ---------------------------------------------------------
    # D. FFT Signal Processing
    # ---------------------------------------------------------
    df = extract_fft_features(df, 'x')

    # Drop originals
    drop_cols = ['ID', 'Class', 'label', 'Mass_Pilot'] + \
                [f'x{i}' for i in range(256)] + [f'p{i}' for i in range(256)]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    return df

# Feature Engineering 실행
X = engineer_features(train_df, is_train=True)
X_test = engineer_features(test_df, is_train=False)

# Target 설정
if 'Class' in train_df.columns:
    y = pd.to_numeric(train_df['Class'].map({'Good': 0, 'NG': 1}), errors='coerce').fillna(0).astype(int).values
else:
    y = np.zeros(len(train_df))

# 범주형 인코딩 (One-Hot)
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=cat_cols)
X_test = pd.get_dummies(X_test, columns=cat_cols)
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# 스케일링
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X)), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(imputer.transform(X_test)), columns=X_test.columns)

print(f"Final Features: {X.shape[1]}")

# ============================================================================
# 3. Model Training (Seed Ensemble)
# ============================================================================
print("\n[Step 3] Training with Seed Ensemble (CatBoost + LightGBM)...")

# 최적 파라미터 (이전 Optuna 결과 반영)
cat_params = {
    'iterations': 1500, 'learning_rate': 0.03, 'depth': 6, 'l2_leaf_reg': 4,
    'auto_class_weights': 'Balanced', 'verbose': 0, 'allow_writing_files': False
}
lgb_params = {
    'n_estimators': 1200, 'learning_rate': 0.03, 'num_leaves': 40, 'max_depth': 7,
    'reg_alpha': 0.3, 'reg_lambda': 0.3, 'class_weight': 'balanced', 'verbose': -1
}

# 결과 저장소
oof_prob_cat = np.zeros(len(X))
test_prob_cat = np.zeros(len(X_test))
oof_prob_lgb = np.zeros(len(X))
test_prob_lgb = np.zeros(len(X_test))

# --- Seed Loop ---
for seed in SEEDS:
    print(f"  Processing Seed: {seed}")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    # Update Seed
    cat_params['random_seed'] = seed
    lgb_params['random_state'] = seed
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_scaled, y)):
        X_tr, X_va = X_scaled.iloc[tr_idx], X_scaled.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        # 1. CatBoost
        cat = CatBoostClassifier(**cat_params)
        cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
        oof_prob_cat[va_idx] += cat.predict_proba(X_va)[:, 1] / len(SEEDS)
        test_prob_cat += cat.predict_proba(X_test_scaled)[:, 1] / (N_FOLDS * len(SEEDS))
        
        # 2. LightGBM
        lgbm = lgb.LGBMClassifier(**lgb_params)
        lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_prob_lgb[va_idx] += lgbm.predict_proba(X_va)[:, 1] / len(SEEDS)
        test_prob_lgb += lgbm.predict_proba(X_test_scaled)[:, 1] / (N_FOLDS * len(SEEDS))

# 50:50 Weighted Ensemble
final_oof = 0.5 * oof_prob_cat + 0.5 * oof_prob_lgb
final_test_prob = 0.5 * test_prob_cat + 0.5 * test_prob_lgb

print(f"\n>>> Final Seed-Ensemble OOF AUC: {roc_auc_score(y, final_oof):.5f}")

# ============================================================================
# 4. Profit Optimization (Cumulative Sum Method)
# ============================================================================
print("\n[Step 4] Optimizing Selection Count (Cumulative Profit)...")

def calculate_competition_score(auc, tp, fp):
    # 대회 공식 점수 계산 함수 (참고용)
    norm_auc = max(auc - 0.5, 0) / 0.5
    total_net_profit = 100 * tp - 2000 * fp
    norm_profit = max(total_net_profit, 0) / 20000
    if norm_auc > 0 and norm_profit > 0:
        return np.sqrt(norm_auc * norm_profit), total_net_profit
    return 0.0, total_net_profit

# 1. OOF 데이터를 '불량 확률이 낮은 순(정상 확률 높은 순)'으로 정렬
oof_df = pd.DataFrame({'true_ng': y, 'prob_ng': final_oof})
oof_df = oof_df.sort_values('prob_ng', ascending=True) # 오름차순 (0에 가까운게 위로)

# 2. 누적 수익 계산 (정상=+100, 불량=-2000)
# 주의: prob_ng가 낮다는 건 '정상'이라고 예측했다는 뜻.
# 따라서 위에서부터 하나씩 선택(TRUE)한다고 가정하고 누적 수익을 계산.
rewards = np.where(oof_df['true_ng'] == 0, 100, -2000) # 정상이면 +100, 아니면 -2000
cumulative_profit = np.cumsum(rewards)

# 3. 최대 수익 지점 찾기
max_profit_idx = np.argmax(cumulative_profit)
best_count = max_profit_idx + 1
max_profit = cumulative_profit[max_profit_idx]

print(f"   Peak Profit Validation: {max_profit:,} Won")
print(f"   Optimal Selection Count (Valid): {best_count}")

# 4. 안전 마진 적용 (Aggressive Ratio)
target_count = int(best_count * AGGRESSIVE_RATIO)
if target_count > 200: target_count = 200 # Rule

# OOF 시뮬레이션 점수 확인
tp_sim = (oof_df.iloc[:target_count]['true_ng'] == 0).sum()
fp_sim = (oof_df.iloc[:target_count]['true_ng'] == 1).sum()
final_score, net_profit = calculate_competition_score(roc_auc_score(y, final_oof), tp_sim, fp_sim)

print(f"   >>> Simulation [Top {target_count}]: Profit {net_profit:,}, Score {final_score:.5f}")
print(f"   (Good Selected: {tp_sim}, Bad Selected: {fp_sim})")

# ============================================================================
# 5. Submission Generation
# ============================================================================
print("\n[Step 5] Generating Submission...")

# 1. Test 데이터 정렬
test_result = pd.DataFrame({
    'ID_L': [f'{pid}_L' for pid in test_df['ID']],
    'ID_P': [f'{pid}_P' for pid in test_df['ID']],
    'prob_ng': final_test_prob
})
test_result['original_idx'] = test_result.index
test_result = test_result.sort_values('prob_ng', ascending=True)

# 2. 상위 N개 선택
test_result['decision'] = 0
test_result.iloc[:target_count, test_result.columns.get_loc('decision')] = 1

print(f"   Final Test Selection: {test_result['decision'].sum()} / 200")

# 3. 원래 순서대로 정렬 및 저장
test_result = test_result.sort_values('original_idx')
decision_str = test_result['decision'].apply(lambda x: 'TRUE' if x==1 else 'FALSE')

submission_l = pd.DataFrame({'ID': test_result['ID_L'], 'probability': test_result['prob_ng'], 'decision': decision_str})
submission_p = pd.DataFrame({'ID': test_result['ID_P'], 'probability': test_result['prob_ng'], 'decision': decision_str})

submission = pd.concat([submission_l, submission_p], axis=0, ignore_index=True)
submission.to_csv("submission_1123_ver3.csv", index=False)

print(">>> Saved 'submission_1123_ver3.csv' successfully.")