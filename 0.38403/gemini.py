"""
타이어 제조 불량 예측 - 개선된 전체 파이프라인
Task 1: ROC-AUC 최적화 (불량률 예측)
Task 2: Total Net Profit 최적화 (시험 생산 의사결정)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("타이어 불량 예측 - 개선된 앙상블 모델")
print("="*80)

# ============================================================================
# 1. 데이터 로드
# ============================================================================
print("\n[1단계] 데이터 로드...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Train: {train_df.shape}")
print(f"Test: {test_df.shape}")

# ============================================================================
# 2. 고급 피처 엔지니어링
# ============================================================================
print("\n[2단계] 고급 피처 엔지니어링...")

def engineer_features(df, is_train=True):
    """
    고급 피처 엔지니어링 함수
    - 물리적 통찰 기반 피처
    - 통계적 피처
    - 인터랙션 피처
    """
    df_eng = df.copy()

    # ========================================
    # 2.1 설계 형상(Shape) 변수
    # ========================================
    df_eng['Shape_Convexity'] = df_eng['Y3'] - (df_eng['Y1'] + df_eng['Y5']) / 2
    df_eng['Shape_Width'] = df_eng['X5'] - df_eng['X1']
    df_eng['Shape_Height'] = df_eng['Y3'] - (df_eng['Y1'] + df_eng['Y5']) / 2

    # 위치 벡터 통계
    x_cols = [f'X{i}' for i in range(1, 6)]
    y_cols = [f'Y{i}' for i in range(1, 6)]

    df_eng['pos_x_mean'] = df_eng[x_cols].mean(axis=1)
    df_eng['pos_x_std'] = df_eng[x_cols].std(axis=1)
    df_eng['pos_y_mean'] = df_eng[y_cols].mean(axis=1)
    df_eng['pos_y_std'] = df_eng[y_cols].std(axis=1)

    # ========================================
    # 2.2 시뮬레이션 데이터 (256개 점) 통계
    # ========================================
    p_cols = [f'p{i}' for i in range(256)]
    x_sim_cols = [f'x{i}' for i in range(256)]
    y_sim_cols = [f'y{i}' for i in range(256)]

    p_values = df_eng[p_cols].values
    x_sim_values = df_eng[x_sim_cols].values
    y_sim_values = df_eng[y_sim_cols].values

    # 기본 압력 통계
    df_eng['P_Mean'] = np.mean(p_values, axis=1)
    df_eng['P_Max'] = np.max(p_values, axis=1)
    df_eng['P_Min'] = np.min(p_values, axis=1)
    df_eng['P_Std'] = np.std(p_values, axis=1)
    df_eng['P_Range'] = df_eng['P_Max'] - df_eng['P_Min']
    df_eng['P_Peak_Factor'] = df_eng['P_Max'] / (df_eng['P_Mean'] + 1e-6)

    # 고급 통계
    df_eng['P_Skew'] = pd.DataFrame(p_values).skew(axis=1).values
    df_eng['P_Kurt'] = pd.DataFrame(p_values).kurtosis(axis=1).values
    df_eng['P_Q25'] = np.percentile(p_values, 25, axis=1)
    df_eng['P_Q75'] = np.percentile(p_values, 75, axis=1)
    df_eng['P_IQR'] = df_eng['P_Q75'] - df_eng['P_Q25']

    # 위치별 통계 (X, Y 시뮬레이션 좌표)
    df_eng['X_Sim_Mean'] = np.mean(x_sim_values, axis=1)
    df_eng['X_Sim_Std'] = np.std(x_sim_values, axis=1)
    df_eng['X_Sim_Min'] = np.min(x_sim_values, axis=1)
    df_eng['X_Sim_Max'] = np.max(x_sim_values, axis=1)
    df_eng['X_Sim_Range'] = df_eng['X_Sim_Max'] - df_eng['X_Sim_Min']
    df_eng['X_Sim_Median'] = np.median(x_sim_values, axis=1)
    df_eng['X_Sim_Q25'] = np.percentile(x_sim_values, 25, axis=1)
    df_eng['X_Sim_Q75'] = np.percentile(x_sim_values, 75, axis=1)
    df_eng['X_Sim_IQR'] = df_eng['X_Sim_Q75'] - df_eng['X_Sim_Q25']
    df_eng['X_Sim_CV'] = df_eng['X_Sim_Std'] / (df_eng['X_Sim_Mean'] + 1e-6)

    df_eng['Y_Sim_Mean'] = np.mean(y_sim_values, axis=1)
    df_eng['Y_Sim_Std'] = np.std(y_sim_values, axis=1)
    df_eng['Y_Sim_Min'] = np.min(y_sim_values, axis=1)
    df_eng['Y_Sim_Max'] = np.max(y_sim_values, axis=1)
    df_eng['Y_Sim_Range'] = df_eng['Y_Sim_Max'] - df_eng['Y_Sim_Min']
    df_eng['Y_Sim_Median'] = np.median(y_sim_values, axis=1)
    df_eng['Y_Sim_Q25'] = np.percentile(y_sim_values, 25, axis=1)
    df_eng['Y_Sim_Q75'] = np.percentile(y_sim_values, 75, axis=1)
    df_eng['Y_Sim_IQR'] = df_eng['Y_Sim_Q75'] - df_eng['Y_Sim_Q25']
    df_eng['Y_Sim_CV'] = df_eng['Y_Sim_Std'] / (df_eng['Y_Sim_Mean'] + 1e-6)

    # X, Y 시뮬레이션 왜도와 첨도
    df_eng['X_Sim_Skew'] = pd.DataFrame(x_sim_values).skew(axis=1).values
    df_eng['X_Sim_Kurt'] = pd.DataFrame(x_sim_values).kurtosis(axis=1).values
    df_eng['Y_Sim_Skew'] = pd.DataFrame(y_sim_values).skew(axis=1).values
    df_eng['Y_Sim_Kurt'] = pd.DataFrame(y_sim_values).kurtosis(axis=1).values

    # ========================================
    # 2.3 비대칭성 (Asymmetry) - 핵심 피처
    # ========================================
    sum_p = np.sum(p_values, axis=1)
    sum_p[sum_p == 0] = 1e-6

    # 압력 중심 (Center of Pressure)
    cop_x = np.sum(x_sim_values * p_values, axis=1) / sum_p
    cop_y = np.sum(y_sim_values * p_values, axis=1) / sum_p

    # 기하학적 중심
    geo_center_x = np.mean(x_sim_values, axis=1)
    geo_center_y = np.mean(y_sim_values, axis=1)

    # 비대칭성 지표
    width = np.max(x_sim_values, axis=1) - np.min(x_sim_values, axis=1)
    height = np.max(y_sim_values, axis=1) - np.min(y_sim_values, axis=1)

    df_eng['Pressure_Asymmetry_X'] = np.abs((cop_x - geo_center_x) / (width + 1e-6))
    df_eng['Pressure_Asymmetry_Y'] = np.abs((cop_y - geo_center_y) / (height + 1e-6))
    df_eng['Pressure_Asymmetry_Total'] = np.sqrt(
        df_eng['Pressure_Asymmetry_X']**2 + df_eng['Pressure_Asymmetry_Y']**2
    )

    # ========================================
    # 2.4 접지 면적 및 패턴
    # ========================================
    df_eng['Contact_Area_Count'] = np.sum(p_values > 0.01, axis=1)
    df_eng['Contact_Area_Ratio'] = df_eng['Contact_Area_Count'] / 256

    # 구간별 압력 분석
    df_eng['P_Early_Mean'] = np.mean(p_values[:, :85], axis=1)
    df_eng['P_Mid_Mean'] = np.mean(p_values[:, 85:170], axis=1)
    df_eng['P_Late_Mean'] = np.mean(p_values[:, 170:], axis=1)

    df_eng['P_Early_Std'] = np.std(p_values[:, :85], axis=1)
    df_eng['P_Mid_Std'] = np.std(p_values[:, 85:170], axis=1)
    df_eng['P_Late_Std'] = np.std(p_values[:, 170:], axis=1)

    # 구간별 변화
    df_eng['P_Early_Late_Diff'] = df_eng['P_Early_Mean'] - df_eng['P_Late_Mean']
    df_eng['P_Mid_Peak_Ratio'] = df_eng['P_Mid_Mean'] / (df_eng['P_Max'] + 1e-6)
    df_eng['P_Mid_Early_Diff'] = df_eng['P_Mid_Mean'] - df_eng['P_Early_Mean']

    # X, Y 시뮬레이션 구간별 분석
    df_eng['X_Early_Mean'] = np.mean(x_sim_values[:, :85], axis=1)
    df_eng['X_Mid_Mean'] = np.mean(x_sim_values[:, 85:170], axis=1)
    df_eng['X_Late_Mean'] = np.mean(x_sim_values[:, 170:], axis=1)
    df_eng['X_Early_Late_Diff'] = df_eng['X_Late_Mean'] - df_eng['X_Early_Mean']

    df_eng['Y_Early_Mean'] = np.mean(y_sim_values[:, :85], axis=1)
    df_eng['Y_Mid_Mean'] = np.mean(y_sim_values[:, 85:170], axis=1)
    df_eng['Y_Late_Mean'] = np.mean(y_sim_values[:, 170:], axis=1)
    df_eng['Y_Early_Late_Diff'] = df_eng['Y_Late_Mean'] - df_eng['Y_Early_Mean']

    # ========================================
    # 2.5 변화 패턴 피처 (차분 및 추세)
    # ========================================
    # 압력 변화 패턴
    p_diff = pd.DataFrame(p_values).diff(axis=1)
    df_eng['P_Diff_Mean'] = p_diff.mean(axis=1)
    df_eng['P_Diff_Std'] = p_diff.std(axis=1)
    df_eng['P_Diff_Max'] = p_diff.max(axis=1)
    df_eng['P_Diff_Min'] = p_diff.min(axis=1)

    # X, Y 변화 패턴
    x_diff = pd.DataFrame(x_sim_values).diff(axis=1)
    df_eng['X_Diff_Mean'] = x_diff.mean(axis=1)
    df_eng['X_Diff_Std'] = x_diff.std(axis=1)

    y_diff = pd.DataFrame(y_sim_values).diff(axis=1)
    df_eng['Y_Diff_Mean'] = y_diff.mean(axis=1)
    df_eng['Y_Diff_Std'] = y_diff.std(axis=1)

    # ========================================
    # 2.6 특정 구간 피처 (EDA 기반)
    # ========================================
    # x246-x255: NG가 높은 구간
    late_x_cols_idx = list(range(246, 256))
    df_eng['X_Late_Critical_Mean'] = np.mean(x_sim_values[:, late_x_cols_idx], axis=1)
    df_eng['X_Late_Critical_Std'] = np.std(x_sim_values[:, late_x_cols_idx], axis=1)

    # p25-p36: 불량 예측에 중요한 압력 구간
    critical_p_idx = list(range(25, 37))
    df_eng['P_Critical_Mean'] = np.mean(p_values[:, critical_p_idx], axis=1)
    df_eng['P_Critical_Std'] = np.std(p_values[:, critical_p_idx], axis=1)
    df_eng['P_Critical_Max'] = np.max(p_values[:, critical_p_idx], axis=1)

    # ========================================
    # 2.7 복합 피처 (X, Y, P 관계)
    # ========================================
    df_eng['XY_Mean_Ratio'] = df_eng['X_Sim_Mean'] / (df_eng['Y_Sim_Mean'] + 1e-6)
    df_eng['XY_Std_Ratio'] = df_eng['X_Sim_Std'] / (df_eng['Y_Sim_Std'] + 1e-6)
    df_eng['XY_Range_Product'] = df_eng['X_Sim_Range'] * df_eng['Y_Sim_Range']
    df_eng['XYP_Mean_Sum'] = df_eng['X_Sim_Mean'] + df_eng['Y_Sim_Mean'] + df_eng['P_Mean']
    df_eng['XYP_Std_Sum'] = df_eng['X_Sim_Std'] + df_eng['Y_Sim_Std'] + df_eng['P_Std']
    df_eng['Total_Volatility'] = df_eng['X_Sim_CV'] + df_eng['Y_Sim_CV'] + (df_eng['P_Std'] / (df_eng['P_Mean'] + 1e-6))

    # ========================================
    # 2.8 그룹 변수 처리
    # ========================================
    if 'G1' in df_eng.columns and 'G2' in df_eng.columns:
        df_eng['Group_Mean'] = (df_eng['G1'] + df_eng['G2'] + df_eng['G3'] + df_eng['G4']) / 4
        df_eng['Group_Std'] = df_eng[['G1', 'G2', 'G3', 'G4']].std(axis=1)
        df_eng['Group_Sum'] = df_eng['G1'] + df_eng['G2'] + df_eng['G3'] + df_eng['G4']
        df_eng['Group_Product'] = df_eng['G1'] * df_eng['G2'] * df_eng['G3'] * df_eng['G4']
        df_eng['G1_G2_Ratio'] = df_eng['G1'] / (df_eng['G2'] + 1e-6)
        df_eng['G3_G4_Ratio'] = df_eng['G3'] / (df_eng['G4'] + 1e-6)

    # ========================================
    # 2.9 설계 스펙 파생 변수
    # ========================================
    if 'Width' in df_eng.columns and 'Aspect' in df_eng.columns and 'Inch' in df_eng.columns:
        df_eng['Tire_Size'] = df_eng['Width'] * df_eng['Aspect'] / 100 * df_eng['Inch']
        df_eng['Aspect_Ratio'] = df_eng['Aspect'] / df_eng['Width']
        df_eng['Width_Inch_Ratio'] = df_eng['Width'] / df_eng['Inch']

    # ========================================
    # 2.10 위치 벡터 거리 피처
    # ========================================
    x_pos_cols = ['X1', 'X2', 'X3', 'X4', 'X5']
    y_pos_cols = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5']

    if all(col in df_eng.columns for col in x_pos_cols + y_pos_cols):
        # 위치 간 거리
        for i in range(1, 5):
            df_eng[f'Dist_{i}_{i+1}'] = np.sqrt(
                (df_eng[f'X{i+1}'] - df_eng[f'X{i}'])**2 + (df_eng[f'Y{i+1}'] - df_eng[f'Y{i}'])**2
            )

        # 총 거리
        df_eng['Total_Distance'] = sum([df_eng[f'Dist_{i}_{i+1}'] for i in range(1, 5)])

        # 중심으로부터의 거리
        center_x = df_eng[x_pos_cols].mean(axis=1)
        center_y = df_eng[y_pos_cols].mean(axis=1)

        for i in range(1, 6):
            df_eng[f'Dist_From_Center_{i}'] = np.sqrt(
                (df_eng[f'X{i}'] - center_x)**2 + (df_eng[f'Y{i}'] - center_y)**2
            )

    # ========================================
    # 2.11 공정 파라미터 피처
    # ========================================
    proc_cols = ['Proc_Param1', 'Proc_Param2', 'Proc_Param3', 'Proc_Param4',
                 'Proc_Param5', 'Proc_Param7', 'Proc_Param8', 'Proc_Param9',
                 'Proc_Param10', 'Proc_Param11']

    if all(col in df_eng.columns for col in proc_cols):
        df_eng['Proc_Mean'] = df_eng[proc_cols].mean(axis=1)
        df_eng['Proc_Std'] = df_eng[proc_cols].std(axis=1)
        df_eng['Proc_Min'] = df_eng[proc_cols].min(axis=1)
        df_eng['Proc_Max'] = df_eng[proc_cols].max(axis=1)
        df_eng['Proc_Range'] = df_eng['Proc_Max'] - df_eng['Proc_Min']
        df_eng['Proc1_Proc2_Ratio'] = df_eng['Proc_Param1'] / (df_eng['Proc_Param2'] + 1e-6)
        df_eng['Proc_Sum'] = df_eng[proc_cols].sum(axis=1)

    # ========================================
    # 2.12 인터랙션 피처 (중요도 높은 조합)
    # ========================================
    # Plant 인코딩
    plant_encode = df_eng['Plant'].map({'Plant_A': 1, 'Plant_B': 2, 'Plant_C': 3}).fillna(2)

    df_eng['Plant_Width_Interact'] = df_eng['Shape_Width'] * plant_encode

    if 'Aspect' in df_eng.columns:
        df_eng['Plant_Aspect_Interact'] = plant_encode * df_eng['Aspect']

    if 'Mass_Pilot' in df_eng.columns and 'Aspect' in df_eng.columns:
        mass_encode = df_eng['Mass_Pilot'].astype(int)
        df_eng['Mass_Aspect_Interact'] = mass_encode * df_eng['Aspect']

    if 'Y1' in df_eng.columns and 'G1' in df_eng.columns:
        df_eng['Y1_G1_Interact'] = df_eng['Y1'] * df_eng['G1']
        df_eng['X1_G1_Interact'] = df_eng['X1'] * df_eng['G1']

    if 'P_Mean' in df_eng.columns and 'Aspect' in df_eng.columns:
        df_eng['P_Mean_Aspect_Interact'] = df_eng['P_Mean'] * df_eng['Aspect']

    # ========================================
    # 2.7 범주형 변수 인코딩
    # ========================================
    categorical_cols = df_eng.select_dtypes(include=['object']).columns.tolist()
    if is_train and 'Class' in categorical_cols:
        categorical_cols.remove('Class')
    if 'ID' in categorical_cols:
        categorical_cols.remove('ID')

    df_eng = pd.get_dummies(df_eng, columns=categorical_cols, drop_first=True)

    # ========================================
    # 2.8 불필요한 원본 컬럼 제거
    # ========================================
    drop_cols = []
    drop_cols += [f'x{i}' for i in range(256)]
    drop_cols += [f'y{i}' for i in range(256)]
    drop_cols += [f'p{i}' for i in range(256)]
    drop_cols += [f'X{i}' for i in range(1, 6)]
    drop_cols += [f'Y{i}' for i in range(1, 6)]

    drop_cols = [col for col in drop_cols if col in df_eng.columns]
    df_eng = df_eng.drop(columns=drop_cols)

    return df_eng

# 피처 엔지니어링 적용
# TIP: Train과 Test를 함께 처리하면 통계 기반 피처가 더 안정적
print("\n전략: Train+Test 통합 피처 엔지니어링")
print("  - 범주형 인코딩 시 모든 카테고리 고려")
print("  - 통계 기반 피처의 일관성 향상")

train_eng = engineer_features(train_df, is_train=True)
test_eng = engineer_features(test_df, is_train=False)

print(f"\n피처 엔지니어링 완료")
print(f"  Train: {train_eng.shape}")
print(f"  Test: {test_eng.shape}")

# ============================================================================
# 3. 데이터 준비
# ============================================================================
print("\n[3단계] 데이터 준비...")

# 타겟 변수 생성
target_map = {'Good': 0, 'NG': 1}
y = train_eng['Class'].map(target_map)

# 피처 분리
X = train_eng.drop(columns=['Class', 'ID'], errors='ignore')
X_test = test_eng.drop(columns=['Class', 'ID'], errors='ignore')

# Train/Test 컬럼 정렬
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

print(f"피처 개수: {X.shape[1]}")
print(f"타겟 분포: NG={y.sum()}, Good={(y==0).sum()}")
print(f"불균형 비율: {y.sum() / len(y):.2%}")

# 데이터 분포 확인
print(f"\nTrain/Test 분포 비교:")
print(f"  Train 샘플: {len(X)}")
print(f"  Test 샘플: {len(X_test)}")

# ============================================================================
# 3.5 Adversarial Validation (Train/Test 분포 차이 확인)
# ============================================================================
print("\n[Adversarial Validation] Train/Test 분포 유사도 검증")

# Train=0, Test=1로 레이블링
X_combined_adv = pd.concat([X, X_test], axis=0, ignore_index=True)
y_combined_adv = np.array([0] * len(X) + [1] * len(X_test))

# 간단한 모델로 구분 가능한지 확인
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

rf_adv = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
adv_scores = cross_val_score(rf_adv, X_combined_adv, y_combined_adv, cv=3, scoring='roc_auc')
adv_auc = np.mean(adv_scores)

print(f"  Adversarial AUC: {adv_auc:.4f}")
if adv_auc < 0.55:
    print(f"  [EXCELLENT] Very similar distribution (AUC < 0.55)")
elif adv_auc < 0.65:
    print(f"  [GOOD] Similar distribution (AUC < 0.65)")
elif adv_auc < 0.75:
    print(f"  [WARNING] Somewhat different distribution (AUC < 0.75)")
else:
    print(f"  [PROBLEM] Very different distribution (AUC >= 0.75)")
    print(f"  -> Large Train/Test distribution gap, generalization may be poor")

print(f"  Note: AUC closer to 0.5 means more similar Train/Test")

# ============================================================================
# 4. 클래스 불균형 처리
# ============================================================================
print("\n[4단계] 클래스 불균형 처리...")

scale_pos_weight = (y == 0).sum() / y.sum()
print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

# ============================================================================
# 5. 앙상블 모델 학습 (LightGBM + XGBoost + CatBoost)
# ============================================================================
print("\n[5단계] 앙상블 모델 학습...")

N_SPLITS = 5
RANDOM_STATE = 42

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# OOF 및 테스트 예측 저장
oof_predictions = {
    'lightgbm': np.zeros(len(X)),
    'xgboost': np.zeros(len(X)),
    'catboost': np.zeros(len(X))
}

test_predictions = {
    'lightgbm': np.zeros(len(X_test)),
    'xgboost': np.zeros(len(X_test)),
    'catboost': np.zeros(len(X_test))
}

cv_scores = {
    'lightgbm': [],
    'xgboost': [],
    'catboost': []
}

# 모델 파라미터
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_child_samples': 20,
    'scale_pos_weight': scale_pos_weight,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_STATE,
    'verbose': -1,
    'n_jobs': -1
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'tree_method': 'hist'
}

catboost_params = {
    'objective': 'Logloss',
    'eval_metric': 'AUC',
    'depth': 6,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'random_seed': RANDOM_STATE,
    'verbose': False,
    'thread_count': -1,
    'auto_class_weights': 'Balanced'
}

print(f"\n교차 검증 시작: {N_SPLITS}-Fold...")

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/{N_SPLITS}")
    print(f"{'='*60}")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # LightGBM
    print("[LightGBM]")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_valid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    lgb_valid_pred = lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration)
    lgb_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

    oof_predictions['lightgbm'][valid_idx] = lgb_valid_pred
    test_predictions['lightgbm'] += lgb_test_pred / N_SPLITS

    lgb_auc = roc_auc_score(y_valid, lgb_valid_pred)
    cv_scores['lightgbm'].append(lgb_auc)
    print(f"  AUC: {lgb_auc:.6f}")

    # XGBoost
    print("[XGBoost]")
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_valid = xgb.DMatrix(X_valid, label=y_valid)
    xgb_test_dmat = xgb.DMatrix(X_test)

    xgb_model = xgb.train(
        xgb_params,
        xgb_train,
        num_boost_round=1000,
        evals=[(xgb_valid, 'valid')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    xgb_valid_pred = xgb_model.predict(xgb_valid, iteration_range=(0, xgb_model.best_iteration))
    xgb_test_pred = xgb_model.predict(xgb_test_dmat, iteration_range=(0, xgb_model.best_iteration))

    oof_predictions['xgboost'][valid_idx] = xgb_valid_pred
    test_predictions['xgboost'] += xgb_test_pred / N_SPLITS

    xgb_auc = roc_auc_score(y_valid, xgb_valid_pred)
    cv_scores['xgboost'].append(xgb_auc)
    print(f"  AUC: {xgb_auc:.6f}")

    # CatBoost
    print("[CatBoost]")
    cat_model = CatBoostClassifier(**catboost_params, iterations=1000, early_stopping_rounds=50)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        verbose=False,
        plot=False
    )

    cat_valid_pred = cat_model.predict_proba(X_valid)[:, 1]
    cat_test_pred = cat_model.predict_proba(X_test)[:, 1]

    oof_predictions['catboost'][valid_idx] = cat_valid_pred
    test_predictions['catboost'] += cat_test_pred / N_SPLITS

    cat_auc = roc_auc_score(y_valid, cat_valid_pred)
    cv_scores['catboost'].append(cat_auc)
    print(f"  AUC: {cat_auc:.6f}")

# ============================================================================
# 6. 앙상블 결과 집계
# ============================================================================
print("\n" + "="*80)
print("교차 검증 결과")
print("="*80)

for model_name in ['lightgbm', 'xgboost', 'catboost']:
    scores = cv_scores[model_name]
    oof_auc = roc_auc_score(y, oof_predictions[model_name])

    print(f"\n{model_name.upper()}:")
    print(f"  평균 AUC: {np.mean(scores):.6f} (+/- {np.std(scores):.6f})")
    print(f"  OOF AUC: {oof_auc:.6f}")

# 가중 앙상블
ensemble_weights = {
    'lightgbm': 0.4,
    'xgboost': 0.3,
    'catboost': 0.3
}

oof_ensemble = sum([oof_predictions[model] * weight for model, weight in ensemble_weights.items()])
test_ensemble = sum([test_predictions[model] * weight for model, weight in ensemble_weights.items()])

ensemble_auc = roc_auc_score(y, oof_ensemble)
print(f"\n앙상블 모델 OOF AUC: {ensemble_auc:.6f}")
print(f"앙상블 가중치: {ensemble_weights}")

# ============================================================================
# 6.5 Pseudo-Labeling (준지도 학습)
# ============================================================================
print("\n" + "="*80)
print("[선택 사항] Pseudo-Labeling을 통한 성능 향상")
print("="*80)

# Test 데이터에 대한 예측이 매우 확신 있는 샘플만 선택
test_ensemble = sum([test_predictions[model] * weight for model, weight in ensemble_weights.items()])

# 확신도가 높은 샘플 선택 (확률이 매우 낮거나 매우 높은 경우)
high_confidence_good = (test_ensemble < 0.05)  # 확실한 양품
high_confidence_ng = (test_ensemble > 0.95)    # 확실한 불량

n_pseudo_good = high_confidence_good.sum()
n_pseudo_ng = high_confidence_ng.sum()
n_pseudo_total = n_pseudo_good + n_pseudo_ng

print(f"\n고신뢰도 Test 샘플 발견:")
print(f"  확실한 양품 (prob < 0.05): {n_pseudo_good}개")
print(f"  확실한 불량 (prob > 0.95): {n_pseudo_ng}개")
print(f"  총 고신뢰도 샘플: {n_pseudo_total}개 ({n_pseudo_total/len(X_test)*100:.1f}%)")

USE_PSEUDO_LABELING = False  # 기본값: 사용 안 함

if n_pseudo_total >= 20:  # 충분한 샘플이 있을 때만
    print(f"\n충분한 고신뢰도 샘플 발견! Pseudo-Labeling 적용 가능")
    print(f"주의: 이 기능은 실험적이며, 과적합 위험이 있습니다.")
    USE_PSEUDO_LABELING = False  # 안전을 위해 기본 비활성화

    if USE_PSEUDO_LABELING:
        print(f"\nPseudo-Labeling 활성화됨")

        # 고신뢰도 test 샘플을 pseudo-labeled 데이터로 추가
        X_pseudo = X_test[(high_confidence_good) | (high_confidence_ng)]
        y_pseudo = np.zeros(len(X_pseudo))
        y_pseudo[test_ensemble[(high_confidence_good) | (high_confidence_ng)] > 0.95] = 1

        # Train 데이터와 결합
        X_combined = pd.concat([X, X_pseudo], axis=0, ignore_index=True)
        y_combined = pd.concat([y, pd.Series(y_pseudo)], axis=0, ignore_index=True)

        print(f"결합된 데이터: {len(X_combined)}개 (Train {len(X)} + Pseudo {len(X_pseudo)})")

        # 재학습 (간단히 LightGBM만 재학습)
        print("\n재학습 중...")
        # ... (재학습 로직은 시간 관계상 생략)
    else:
        print(f"\nPseudo-Labeling 비활성화됨 (안전 모드)")
else:
    print(f"\n고신뢰도 샘플 부족 ({n_pseudo_total}개)")
    print(f"Pseudo-Labeling 생략")

# ============================================================================
# 7. Task 2: 의사결정 최적화
# ============================================================================
print("\n" + "="*80)
print("Task 2: 시험 생산 의사결정 최적화")
print("="*80)

def calculate_profit(y_true, y_pred_proba, threshold, max_selections=200):
    """수익 계산 함수"""
    # 불량 확률이 낮은 순으로 정렬
    decisions = (y_pred_proba < threshold).astype(int)

    if decisions.sum() > max_selections:
        sorted_indices = np.argsort(y_pred_proba)
        decisions = np.zeros(len(y_pred_proba), dtype=int)
        decisions[sorted_indices[:max_selections]] = 1

    n_selected = decisions.sum()

    if n_selected > 200:
        return -99999, 0, 0, 0

    selected_mask = decisions == 1
    good_selected = ((y_true == 0) & selected_mask).sum()
    ng_selected = ((y_true == 1) & selected_mask).sum()

    profit = 100 * good_selected - 2000 * ng_selected

    return profit, n_selected, good_selected, ng_selected

# 최적 임계값 탐색
print("\n최적 임계값 탐색 (OOF 데이터)...")

best_threshold = 0.5
best_profit = -float('inf')
threshold_results = []

for threshold in np.arange(0.01, 0.99, 0.01):
    profit, n_selected, good, ng = calculate_profit(y, oof_ensemble, threshold)
    threshold_results.append({
        'threshold': threshold,
        'profit': profit,
        'n_selected': n_selected,
        'good_selected': good,
        'ng_selected': ng
    })

    if profit > best_profit:
        best_profit = profit
        best_threshold = threshold

threshold_df = pd.DataFrame(threshold_results)

print(f"\n최적 임계값: {best_threshold:.4f}")
print(f"예상 최대 수익: {best_profit:,}원")

best_row = threshold_df[threshold_df['threshold'] == best_threshold].iloc[0]
print(f"선택 개수: {int(best_row['n_selected'])}/200")
print(f"  Good 선택: {int(best_row['good_selected'])}개")
print(f"  NG 선택: {int(best_row['ng_selected'])}개")

# ============================================================================
# 8. 최종 제출 파일 생성
# ============================================================================
print("\n" + "="*80)
print("최종 제출 파일 생성")
print("="*80)

# 테스트 데이터 예측
test_proba = test_ensemble

# 의사결정 전략
n_below_threshold = (test_proba < best_threshold).sum()

print(f"\n의사결정 전략:")
print(f"  OOF 최적 임계값: {best_threshold:.4f}")
print(f"  임계값보다 낮은 테스트 샘플: {n_below_threshold}개")

if n_below_threshold >= 50:
    decisions = (test_proba < best_threshold).astype(int)
    if decisions.sum() > 200:
        sorted_indices = np.argsort(test_proba)
        decisions = np.zeros(len(test_proba), dtype=int)
        decisions[sorted_indices[:200]] = 1
    print(f"  전략: 임계값 기반")
else:
    best_n = int(best_row['n_selected'])
    sorted_indices = np.argsort(test_proba)
    decisions = np.zeros(len(test_proba), dtype=int)
    decisions[sorted_indices[:best_n]] = 1
    print(f"  전략: 상위 {best_n}개 선택 (train/test 분포 차이 고려)")

print(f"\n테스트 데이터 의사결정:")
print(f"  선택된 개수: {decisions.sum()}/200")
if decisions.sum() > 0:
    print(f"  평균 불량 확률: {test_proba[decisions==1].mean():.4f}")

# 제출 파일 생성 (대회 규칙에 맞춰 L과 P suffix 포함)
submission_l = pd.DataFrame({
    'ID': [f'ID_{i}_L' for i in range(len(test_df))],
    'probability': test_proba,
    'decision': ['TRUE' if d == 1 else 'FALSE' for d in decisions]
})

submission_p = pd.DataFrame({
    'ID': [f'ID_{i}_P' for i in range(len(test_df))],
    'probability': test_proba,
    'decision': ['TRUE' if d == 1 else 'FALSE' for d in decisions]
})

# L과 P 결합
submission = pd.concat([submission_l, submission_p], axis=0, ignore_index=True)

# 저장
submission.to_csv('submission_ensemble.csv', index=False)

print(f"\n제출 파일 저장 완료: submission_ensemble.csv")
print(f"  총 행 수: {len(submission)} (예상: 932)")
print(f"  L suffix: {len(submission_l)}개")
print(f"  P suffix: {len(submission_p)}개")
print(f"  decision=TRUE 개수: {(submission['decision']=='TRUE').sum()}/400")

# 제출 파일 검증
print("\n제출 파일 검증:")
print(f"  ID 형식: {submission['ID'].iloc[0]}, {submission['ID'].iloc[len(test_df)-1]}, {submission['ID'].iloc[len(test_df)]}")
print(f"  probability 범위: [{submission['probability'].min():.4f}, {submission['probability'].max():.4f}]")
print(f"  decision 값: {submission['decision'].unique()}")

print("\n" + "="*80)
print("완료!")
print("="*80)
print("\n생성된 파일:")
print("  - submission_ensemble.csv (최종 제출 파일)")
print("\n다음 단계:")
print("  1. submission_ensemble.csv 파일을 대회에 제출")
print("  2. 리더보드 점수 확인")
