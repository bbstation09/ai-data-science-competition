"""
타이어 불량 예측 모델
- Task 1: ROC-AUC 최대화 (probability)  
- Task 2: 극보수적 decision (False Positive 비용 20배!)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 머신러닝
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# XGBoost와 LightGBM
import xgboost as xgb
import lightgbm as lgb

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

class TireDefectPredictor:
    """타이어 불량 예측 클래스"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def load_and_preprocess_data(self, train_path='datasets/train.csv', test_path='datasets/test.csv'):
        """데이터 로드 및 전처리"""
        print("📊 데이터 로드 중...")
        
        try:
            self.train = pd.read_csv(train_path)
            self.test = pd.read_csv(test_path)
        except FileNotFoundError:
            print("❌ 데이터 파일을 찾을 수 없습니다.")
            print("경로 확인: train.csv, test.csv")
            return False
            
        print(f"✅ 훈련 데이터: {self.train.shape}")
        print(f"✅ 테스트 데이터: {self.test.shape}")
        
        # 기본 전처리 (EDA에서 확인된 내용)
        self._basic_preprocessing()
        
        # 피처 엔지니어링
        self._feature_engineering()
        
        return True
    
    def _basic_preprocessing(self):
        """기본 전처리"""
        print("🔧 기본 전처리 수행...")
        
        # 카테고리형 변수 처리 (EDA에서 확인)
        for df in [self.train, self.test]:
            # Proc_Param6: P6_0 -> 0
            if 'Proc_Param6' in df.columns and df['Proc_Param6'].dtype == 'object':
                df['Proc_Param6'] = df['Proc_Param6'].str.split('_').str[-1].astype(int)
            
            # Plant: Plant_1 -> 1  
            if 'Plant' in df.columns and df['Plant'].dtype == 'object':
                df['Plant'] = df['Plant'].str.split('_').str[-1].astype(int)
        
        # 타겟 변수 이진 인코딩
        self.train['target'] = (self.train['Class'] == 'NG').astype(int)
        
        print(f"   • 클래스 분포: NG={self.train['target'].sum()}, Good={len(self.train) - self.train['target'].sum()}")
        print(f"   • 불량률: {self.train['target'].mean()*100:.2f}%")
        
    def _feature_engineering(self):
        """피처 엔지니어링 - EDA 결과 기반"""
        print("⚙️  피처 엔지니어링 수행...")
        
        # 시뮬레이션 데이터 통계 피처 생성
        self._create_simulation_features()
        
        # 공장별 품질 피처
        self._create_plant_features()
        
        # 위치 기반 피처  
        self._create_location_features()
        
        # 공정 파라미터 조합 피처
        self._create_process_features()
        
    def _create_simulation_features(self):
        """시뮬레이션 데이터 통계 피처"""
        print("   📈 시뮬레이션 통계 피처 생성...")
        
        for df in [self.train, self.test]:
            # X 시뮬레이션 통계 (x0~x255)
            x_cols = [f'x{i}' for i in range(256) if f'x{i}' in df.columns]
            if x_cols:
                df['x_mean'] = df[x_cols].mean(axis=1)
                df['x_std'] = df[x_cols].std(axis=1)
                df['x_max'] = df[x_cols].max(axis=1)
                df['x_min'] = df[x_cols].min(axis=1)
                df['x_range'] = df['x_max'] - df['x_min']
                
                # 특정 구간 통계 (EDA에서 중요하다고 나온 x246-x255)
                x_end_cols = [f'x{i}' for i in range(246, 256) if f'x{i}' in df.columns]
                if x_end_cols:
                    df['x_end_mean'] = df[x_end_cols].mean(axis=1)
                    df['x_end_std'] = df[x_end_cols].std(axis=1)
            
            # Y 시뮬레이션 통계 (y0~y255)
            y_cols = [f'y{i}' for i in range(256) if f'y{i}' in df.columns]
            if y_cols:
                df['y_mean'] = df[y_cols].mean(axis=1)
                df['y_std'] = df[y_cols].std(axis=1)
                df['y_max'] = df[y_cols].max(axis=1)
                df['y_min'] = df[y_cols].min(axis=1)
                
            # P 시뮬레이션 통계 (p0~p255)  
            p_cols = [f'p{i}' for i in range(256) if f'p{i}' in df.columns]
            if p_cols:
                df['p_mean'] = df[p_cols].mean(axis=1)
                df['p_std'] = df[p_cols].std(axis=1)
                df['p_max'] = df[p_cols].max(axis=1)
                df['p_min'] = df[p_cols].min(axis=1)
                
                # 특정 구간 통계 (EDA에서 중요하다고 나온 p25-p36)
                p_mid_cols = [f'p{i}' for i in range(25, 37) if f'p{i}' in df.columns]
                if p_mid_cols:
                    df['p_mid_mean'] = df[p_mid_cols].mean(axis=1)
                    df['p_mid_std'] = df[p_mid_cols].std(axis=1)
    
    def _create_plant_features(self):
        """공장별 품질 피처"""
        print("   🏭 공장별 품질 피처 생성...")
        
        # 공장별 불량률 계산 (훈련 데이터 기준)
        plant_ng_rate = self.train.groupby('Plant')['target'].mean().to_dict()
        
        # 테스트 데이터에 적용
        for df in [self.train, self.test]:
            df['plant_ng_rate'] = df['Plant'].map(plant_ng_rate).fillna(self.train['target'].mean())
            
            # 공장 크기 (생산량 대리 지표)
            plant_size = self.train['Plant'].value_counts().to_dict()
            df['plant_size'] = df['Plant'].map(plant_size).fillna(1)
    
    def _create_location_features(self):
        """위치 기반 피처"""  
        print("   📍 위치 기반 피처 생성...")
        
        for df in [self.train, self.test]:
            # 위치 벡터 통계
            x_pos_cols = [f'X{i}' for i in range(1, 6) if f'X{i}' in df.columns]
            y_pos_cols = [f'Y{i}' for i in range(1, 6) if f'Y{i}' in df.columns]
            
            if x_pos_cols:
                df['x_pos_mean'] = df[x_pos_cols].mean(axis=1)
                df['x_pos_std'] = df[x_pos_cols].std(axis=1)
                
            if y_pos_cols:
                df['y_pos_mean'] = df[y_pos_cols].mean(axis=1)  
                df['y_pos_std'] = df[y_pos_cols].std(axis=1)
                
            # 중심점으로부터의 거리
            if x_pos_cols and y_pos_cols:
                df['center_distance'] = np.sqrt(df['x_pos_mean']**2 + df['y_pos_mean']**2)
    
    def _create_process_features(self):
        """공정 파라미터 조합 피처"""
        print("   ⚙️  공정 파라미터 피처 생성...")
        
        for df in [self.train, self.test]:
            # 공정 파라미터 통계
            proc_cols = [f'Proc_Param{i}' for i in range(1, 12) if f'Proc_Param{i}' in df.columns]
            if proc_cols:
                df['proc_mean'] = df[proc_cols].mean(axis=1)
                df['proc_std'] = df[proc_cols].std(axis=1)
                df['proc_max'] = df[proc_cols].max(axis=1)
                df['proc_min'] = df[proc_cols].min(axis=1)
                
            # 타이어 사양 조합
            if all(col in df.columns for col in ['Width', 'Aspect', 'Inch']):
                df['tire_size'] = df['Width'] * df['Aspect'] * df['Inch']
                df['aspect_ratio'] = df['Aspect'] / df['Width']
    
    def prepare_features(self):
        """모델링용 피처 준비"""
        print("🎯 모델링용 피처 준비...")
        
        # 피처 선택 (숫자형만)
        feature_cols = []
        for col in self.train.columns:
            if col not in ['ID', 'Class', 'target'] and self.train[col].dtype in ['int64', 'float64']:
                feature_cols.append(col)
        
        self.feature_cols = feature_cols
        print(f"   • 총 피처 수: {len(feature_cols)}")
        
        # 훈련 데이터 분리
        self.X = self.train[feature_cols].copy()
        self.y = self.train['target'].copy()
        
        # 테스트 데이터
        self.X_test = self.test[feature_cols].copy()
        
        # 결측값 처리
        self.X = self.X.fillna(self.X.median())
        self.X_test = self.X_test.fillna(self.X.median())
        
        print(f"   • X shape: {self.X.shape}")
        print(f"   • 불량 샘플: {self.y.sum()}/{len(self.y)} ({self.y.mean()*100:.2f}%)")
        
    def train_models(self):
        """모델 학습"""
        print("🤖 모델 학습 시작...")
        
        # 1. XGBoost (주력 모델)
        self._train_xgboost()
        
        # 2. LightGBM (보완 모델)
        self._train_lightgbm()
        
        # 3. Random Forest (안정성)
        self._train_random_forest()
        
        # 4. 앙상블 가중치 설정
        self._set_ensemble_weights()
        
    def _train_xgboost(self):
        """XGBoost 모델 학습"""
        print("   🚀 XGBoost 학습...")
        
        # 클래스 가중치 계산
        scale_pos_weight = (self.y == 0).sum() / (self.y == 1).sum()
        
        # 하이퍼파라미터 설정 (XGBoost 3.x 호환)
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': scale_pos_weight,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 500,  # 고정된 부스팅 라운드
            'random_state': 42
        }
        
        # 교차 검증 
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(self.X))
        test_preds = np.zeros(len(self.X_test))
        
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # SMOTE 적용 (훈련 세트에만)
            smote = SMOTE(random_state=42)
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            
            # 모델 학습 (XGBoost 3.x 호환)
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train_sm, y_train_sm, verbose=False)
            
            # 예측
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(self.X_test)[:, 1] / 5
            
            # 점수 계산
            val_auc = roc_auc_score(y_val, oof_preds[val_idx])
            cv_scores.append(val_auc)
            print(f"      Fold {fold+1}: AUC = {val_auc:.4f}")
        
        # 최종 모델 (전체 데이터로 학습)
        smote = SMOTE(random_state=42)
        X_full_sm, y_full_sm = smote.fit_resample(self.X, self.y)
        
        self.models['xgb'] = xgb.XGBClassifier(**xgb_params)
        self.models['xgb'].fit(X_full_sm, y_full_sm, verbose=False)
        
        # 피처 중요도 저장
        self.feature_importance['xgb'] = pd.Series(
            self.models['xgb'].feature_importances_,
            index=self.feature_cols
        ).sort_values(ascending=False)
        
        print(f"   ✅ XGBoost CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        self.xgb_oof = oof_preds
        self.xgb_test = test_preds
        
    def _train_lightgbm(self):
        """LightGBM 모델 학습 - sklearn wrapper 사용"""
        print("   💡 LightGBM 학습...")
        
        # sklearn wrapper 사용으로 안정성 확보
        from lightgbm import LGBMClassifier
        
        # 하이퍼파라미터 설정 (sklearn wrapper 방식)
        lgb_model = LGBMClassifier(
            objective='binary',
            metric='auc',
            class_weight='balanced',
            num_leaves=31,
            learning_rate=0.1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            n_estimators=500,
            random_state=42,
            verbose=-1
        )
        
        # 교차 검증
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(self.X))
        test_preds = np.zeros(len(self.X_test))
        
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # SMOTE 적용
            smote = SMOTE(random_state=42)
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            
            # 모델 학습
            model = LGBMClassifier(
                objective='binary',
                metric='auc',
                class_weight='balanced',
                num_leaves=31,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                n_estimators=500,
                random_state=42,
                verbose=-1
            )
            model.fit(X_train_sm, y_train_sm)
            
            # 예측
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(self.X_test)[:, 1] / 5
            
            # 점수 계산
            val_auc = roc_auc_score(y_val, oof_preds[val_idx])
            cv_scores.append(val_auc)
            print(f"      Fold {fold+1}: AUC = {val_auc:.4f}")
        
        # 최종 모델
        smote = SMOTE(random_state=42)
        X_full_sm, y_full_sm = smote.fit_resample(self.X, self.y)
        
        self.models['lgb'] = lgb_model
        self.models['lgb'].fit(X_full_sm, y_full_sm)
        
        # 피처 중요도 저장  
        self.feature_importance['lgb'] = pd.Series(
            self.models['lgb'].feature_importances_, 
            index=self.feature_cols
        ).sort_values(ascending=False)
        
        print(f"   ✅ LightGBM CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        self.lgb_oof = oof_preds
        self.lgb_test = test_preds
        
    def _train_random_forest(self):
        """Random Forest 모델 학습"""
        print("   🌲 Random Forest 학습...")
        
        # 교차 검증
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf, self.X, self.y, cv=skf, scoring='roc_auc')
        
        # 최종 모델 학습
        self.models['rf'] = rf.fit(self.X, self.y)
        
        # 피처 중요도 저장
        self.feature_importance['rf'] = pd.Series(
            self.models['rf'].feature_importances_,
            index=self.feature_cols
        ).sort_values(ascending=False)
        
        # 예측
        self.rf_test = self.models['rf'].predict_proba(self.X_test)[:, 1]
        
        print(f"   ✅ Random Forest CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
    def _set_ensemble_weights(self):
        """앙상블 가중치 설정"""
        print("   🎯 앙상블 가중치 설정...")
        
        # 단순 가중 평균 (XGBoost 중심)
        self.ensemble_weights = {
            'xgb': 0.5,
            'lgb': 0.3, 
            'rf': 0.2
        }
        
        print(f"      XGBoost: {self.ensemble_weights['xgb']}")
        print(f"      LightGBM: {self.ensemble_weights['lgb']}")
        print(f"      Random Forest: {self.ensemble_weights['rf']}")
        
    def evaluate_train_performance(self):
        """훈련 데이터에서의 성능 평가"""
        print("\n📊 훈련 데이터 성능 평가...")
        
        # 앙상블 예측 (훈련 데이터용)
        train_proba = (
            self.xgb_oof * self.ensemble_weights['xgb'] +
            (self.models['lgb'].predict_proba(self.X)[:, 1] * self.ensemble_weights['lgb']) +
            (self.models['rf'].predict_proba(self.X)[:, 1] * self.ensemble_weights['rf'])
        )
        
        # Task 1: ROC-AUC
        train_auc = roc_auc_score(self.y, train_proba)
        print(f"   🎯 Task 1 (ROC-AUC): {train_auc:.4f}")
        
        # Task 2: 다양한 임계값에 대한 Net Profit 계산
        print(f"\n   💰 Task 2 (Net Profit) - 다양한 임계값:")
        print(f"   {'임계값':<8} {'FALSE수':<8} {'TP':<6} {'FP':<6} {'Net Profit':<12} {'종합점수':<10}")
        print(f"   {'-'*60}")
        
        best_threshold = 0.5
        best_total_score = -float('inf')
        
        # 임계값별 성능 평가 (FALSE 예측 관점)
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            # FALSE = 불량 판정 (threshold 이상이면 불량으로 판정)
            decisions_false = train_proba >= threshold
            n_false = decisions_false.sum()
            
            if n_false > 200:
                continue  # 200개 초과하면 스킵
                
            # Task 2 점수 계산
            # TRUE decisions (양품 판정)
            decisions_true = ~decisions_false
            tp = ((decisions_true) & (self.y == 0)).sum()  # 양품을 양품으로 
            fp = ((decisions_true) & (self.y == 1)).sum()  # 불량을 양품으로 (패널티)
            
            net_profit = 100 * tp - 2000 * fp
            
            # Task 1 점수 (ROC-AUC 0.5 기준)
            auc_score = max(train_auc - 0.5, 0) / 0.5
            
            # 종합 점수
            total_score = np.sqrt(auc_score * max(net_profit, 0) / 20000)
            
            print(f"   {threshold:<8.2f} {n_false:<8} {tp:<6} {fp:<6} {net_profit:<12,} {total_score:<10.4f}")
            
            if total_score > best_total_score:
                best_total_score = total_score
                best_threshold = threshold
        
        print(f"\n   🏆 최적 임계값: {best_threshold:.2f} (종합점수: {best_total_score:.4f})")
        return best_threshold, best_total_score
        
    def predict(self):
        """최종 예측"""
        print("🎯 최종 예측 수행...")
        
        # 앙상블 예측 (확률)
        self.final_proba = (
            self.xgb_test * self.ensemble_weights['xgb'] +
            self.lgb_test * self.ensemble_weights['lgb'] + 
            self.rf_test * self.ensemble_weights['rf']
        )
        
        print(f"   • 예측 확률 범위: {self.final_proba.min():.4f} ~ {self.final_proba.max():.4f}")
        print(f"   • 평균 불량 확률: {self.final_proba.mean():.4f}")
        
        # 훈련 데이터 성능 평가 및 최적 임계값 탐색
        optimal_threshold, best_score = self.evaluate_train_performance()
        
        # 최적 임계값 기반 decision 생성
        self._make_optimal_decisions(optimal_threshold)
        
    def _make_optimal_decisions(self, optimal_threshold):
        """최적 임계값 기반 decision 생성"""
        print("🎯 최적화된 decision 생성...")
        
        # 올바른 Task 2 해석:
        # TRUE = 양품 판정 (+100점)
        # FALSE = 불량 판정 (틀리면 -2000점)
        # 200개 제한은 FALSE의 제한
        
        # FALSE = 불량 판정 (optimal_threshold 이상)
        self.final_decisions_false = self.final_proba >= optimal_threshold
        false_count = self.final_decisions_false.sum()
        
        print(f"   • 최적 임계값: {optimal_threshold:.3f}")
        print(f"   • FALSE (불량) 판정: {false_count}개")
        print(f"   • TRUE (양품) 판정: {len(self.final_proba) - false_count}개")
        
        # 200개 제한 확인
        if false_count > 200:
            print(f"   ⚠️  WARNING: FALSE 판정이 200개 초과! ({false_count}개)")
            # 상위 200개만 불량으로 판정
            top_200_idx = np.argsort(self.final_proba)[-200:]
            self.final_decisions_false = np.zeros(len(self.final_proba), dtype=bool)
            self.final_decisions_false[top_200_idx] = True
            false_count = 200
            print(f"   🔧 조정: 상위 200개만 FALSE로 설정")
        
        print(f"\n   ✅ 최종 결정:")
        print(f"      FALSE (불량): {false_count}/200개 사용")
        print(f"      TRUE (양품): {len(self.final_proba) - false_count}개")
        
        # submission 형식에 맞게 변환 (TRUE/FALSE 문자열)
        self.final_decisions = ['FALSE' if is_defect else 'TRUE' 
                              for is_defect in self.final_decisions_false]
    
    def create_submission(self, output_path='/mnt/c/Users/Admin/PycharmProjects/Tire-Content/outputs/submission.csv'):
        """submission 파일 생성"""
        print("📄 Submission 파일 생성...")
        
        # ID 생성 (테스트 데이터에 ID가 없는 경우)
        if 'ID' in self.test.columns:
            test_ids = self.test['ID']
        else:
            test_ids = [f'ID_{i}_L' for i in range(len(self.test))]
        
        # submission 데이터프레임 생성 (이미 문자열로 변환됨)
        submission = pd.DataFrame({
            'ID': test_ids,
            'probability': self.final_proba,
            'decision': self.final_decisions
        })
        
        # 저장
        submission.to_csv(output_path, index=False)
        
        # 요약 정보 출력
        print(f"   ✅ 파일 저장: {output_path}")
        print(f"   📊 요약:")
        print(f"      • 총 샘플: {len(submission)}개")
        print(f"      • TRUE decisions (양품): {(submission['decision'] == 'TRUE').sum()}개")
        print(f"      • FALSE decisions (불량): {(submission['decision'] == 'FALSE').sum()}개")
        print(f"      • 평균 확률: {submission['probability'].mean():.4f}")
        print(f"      • 최대 확률: {submission['probability'].max():.4f}")
        
        return submission
    
    def analyze_results(self):
        """결과 분석 및 시각화"""
        print("📈 결과 분석...")
        
        # 1. 피처 중요도 분석
        self._plot_feature_importance()
        
        # 2. 예측 분포 분석  
        self._plot_prediction_distribution()
        
        # 3. Decision 분석
        self._analyze_decisions()
        
    def _plot_feature_importance(self):
        """피처 중요도 시각화"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            
            for i, (model_name, importance) in enumerate(self.feature_importance.items()):
                top_features = importance.head(15)
                
                axes[i].barh(range(len(top_features)), top_features.values)
                axes[i].set_yticks(range(len(top_features)))
                axes[i].set_yticklabels(top_features.index, fontsize=8)
                axes[i].set_title(f'{model_name.upper()} - Top 15 Features')
                axes[i].set_xlabel('Importance')

            plt.tight_layout()
            plt.savefig('/mnt/c/Users/Admin/PycharmProjects/Tire-Content/outputs/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("   ✅ 피처 중요도 그래프 저장 완료")
        except Exception as e:
            print(f"   ⚠️  피처 중요도 그래프 생성 오류: {e}")
            print("   → 그래프 없이 계속 진행...")
        
    def _plot_prediction_distribution(self):
        """예측 확률 분포 시각화"""
        try:
            plt.figure(figsize=(12, 5))
            
            # 전체 분포
            plt.subplot(1, 2, 1)
            plt.hist(self.final_proba, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            # FALSE 임계값 표시 (불량 판정)
            false_threshold = self.final_proba[self.final_decisions_false].min() if hasattr(self, 'final_decisions_false') and self.final_decisions_false.sum() > 0 else 0.95
            plt.axvline(false_threshold, color='red', linestyle='--', label=f'FALSE Threshold ({false_threshold:.3f})')
            plt.title('예측 확률 분포 (전체)')
            plt.xlabel('불량 확률')
            plt.ylabel('빈도')
            plt.legend()
            
            # 고확률 구간 확대
            plt.subplot(1, 2, 2)
            high_prob = self.final_proba[self.final_proba > 0.8]
            if len(high_prob) > 0:
                plt.hist(high_prob, bins=20, alpha=0.7, color='orange', edgecolor='black')
                plt.axvline(false_threshold, color='red', linestyle='--', label=f'FALSE Threshold ({false_threshold:.3f})')
                plt.title('예측 확률 분포 (고확률 구간)')
                plt.xlabel('불량 확률')
                plt.ylabel('빈도')
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'No high probability samples', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=plt.gca().transAxes)
                plt.title('예측 확률 분포 (고확률 구간)')
            
            plt.tight_layout()
            plt.savefig('/mnt/c/Users/Admin/PycharmProjects/Tire-Content/outputs/prediction_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("   ✅ 예측 분포 그래프 저장 완료")
        except Exception as e:
            print(f"   ⚠️  예측 분포 그래프 생성 오류: {e}")
            print("   → 그래프 없이 계속 진행...")
        
    def _analyze_decisions(self):
        """Decision 분석"""
        false_decisions = sum(1 for d in self.final_decisions if d == 'FALSE')
        true_decisions = len(self.final_decisions) - false_decisions
        
        print(f"\n📊 Decision 분석:")
        print(f"   • FALSE (불량): {false_decisions}개 ({false_decisions/len(self.final_decisions)*100:.1f}%)")
        print(f"   • TRUE (양품): {true_decisions}개 ({true_decisions/len(self.final_decisions)*100:.1f}%)")
        
        if false_decisions > 0:
            false_probas = self.final_proba[[i for i, d in enumerate(self.final_decisions) if d == 'FALSE']]
            print(f"   • FALSE 판정의 평균 확률: {false_probas.mean():.4f}")
            print(f"   • FALSE 판정의 최소 확률: {false_probas.min():.4f}")
        
        # 시뮬레이션된 점수 계산 (실제 정답 모름)
        print(f"\n🎯 예상 시나리오 분석:")
        print("   (FALSE 판정 정확률에 따른 예상 점수)")
        
        # 실제 불량률 14.9% 고려
        expected_defects_in_false = int(false_decisions * 0.149)  # 기대값
        
        for accuracy in [0.2, 0.3, 0.4, 0.5, 0.6]:
            # FALSE 판정 중 실제 불량 개수
            actual_defects = int(false_decisions * accuracy) if false_decisions > 0 else 0
            actual_goods = false_decisions - actual_defects
            
            # TRUE 판정 중 실제 양품/불량 (전체 비율 가정)
            total_goods = int(len(self.final_decisions) * 0.851)  # 전체 양품 수
            total_defects = len(self.final_decisions) - total_goods  # 전체 불량 수
            
            true_goods = total_goods - actual_goods  # TRUE로 판정된 실제 양품
            true_defects = total_defects - actual_defects  # TRUE로 판정된 실제 불량 (FALSE Positive)
            
            # Task 2 점수
            tp = true_goods  # 양품을 양품(TRUE)으로
            fp = true_defects  # 불량을 양품(TRUE)으로
            net_profit = 100 * tp - 2000 * fp
            
            print(f"   • FALSE 정확률 {accuracy*100:2.0f}%: TP={tp}, FP={fp}, Net Profit={net_profit:,}")
            
        print(f"\n💡 최적 전략 달성:")
        print(f"   • {false_decisions}/200 FALSE 판정 사용 (여유: {200-false_decisions}개)")
        print(f"   • 대부분을 양품(TRUE)으로 예측하여 +100점 최대화")
        print(f"   • 확실한 불량만 FALSE로 판정하여 -2000점 최소화")

def main():
    """메인 실행 함수"""
    print("🚀 타이어 불량 예측 모델 시작")
    print("="*50)
    
    try:
        # 모델 객체 생성
        predictor = TireDefectPredictor()
        
        # 데이터 로드 및 전처리
        if not predictor.load_and_preprocess_data():
            print("❌ 데이터 로드 실패")
            return None, None
        
        # 피처 준비
        predictor.prepare_features()
        
        # 모델 학습
        print("\n🤖 모델 학습 단계...")
        predictor.train_models()
        
        # 예측 수행
        print("\n🎯 예측 단계...")
        predictor.predict()
        
        # submission 파일 생성
        print("\n📄 결과 파일 생성...")
        submission = predictor.create_submission()
        
        # 결과 분석
        print("\n📈 결과 분석...")
        predictor.analyze_results()
        
        print("\n🎉 모든 과정 완료!")
        print("="*50)
        
        return predictor, submission
        
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")
        print("스택 트레이스:")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    result = main()
    if result[0] is not None:
        predictor, submission = result
        print("✅ 프로그램이 성공적으로 완료되었습니다!")
    else:
        print("❌ 프로그램 실행에 실패했습니다.")
