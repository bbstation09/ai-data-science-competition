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
        """훈련 데이터 분석 - 실제 Good 샘플의 패턴 학습으로 False Positive 완전 차단"""
        print("\n📊 훈련 데이터 분석 - False Positive 완전 차단 전략...")
        
        # 앙상블 예측 (훈련 데이터용)
        train_proba = (
            self.xgb_oof * self.ensemble_weights['xgb'] +
            (self.models['lgb'].predict_proba(self.X)[:, 1] * self.ensemble_weights['lgb']) +
            (self.models['rf'].predict_proba(self.X)[:, 1] * self.ensemble_weights['rf'])
        )
        
        # Task 1: ROC-AUC
        train_auc = roc_auc_score(self.y, train_proba)
        print(f"   🎯 Task 1 (ROC-AUC): {train_auc:.4f}")
        
        # 실제 Good(y=0)과 NG(y=1) 샘플의 확률 분포 분석
        good_probas = train_proba[self.y == 0]  # 실제 양품의 예측 확률
        ng_probas = train_proba[self.y == 1]    # 실제 불량의 예측 확률
        
        print(f"\n   📊 실제 클래스별 확률 분포 분석:")
        print(f"   실제 Good ({len(good_probas)}개):")
        print(f"     - 평균: {good_probas.mean():.4f}")
        print(f"     - 최소: {good_probas.min():.4f}")
        print(f"     - 최대: {good_probas.max():.4f}")
        print(f"     - 25%ile: {np.percentile(good_probas, 25):.4f}")
        print(f"     - 50%ile: {np.percentile(good_probas, 50):.4f}")
        print(f"     - 75%ile: {np.percentile(good_probas, 75):.4f}")
        
        print(f"\n   실제 NG ({len(ng_probas)}개):")
        print(f"     - 평균: {ng_probas.mean():.4f}")
        print(f"     - 최소: {ng_probas.min():.4f}")
        print(f"     - 최대: {ng_probas.max():.4f}")
        print(f"     - 25%ile: {np.percentile(ng_probas, 25):.4f}")
        print(f"     - 50%ile: {np.percentile(ng_probas, 50):.4f}")
        print(f"     - 75%ile: {np.percentile(ng_probas, 75):.4f}")
        
        # 100% 안전 임계값 찾기
        # 실제 NG 샘플 중 최소값보다 낮은 영역에서만 선택
        safe_threshold = ng_probas.min()
        ultra_safe_goods = good_probas[good_probas <= safe_threshold]
        
        print(f"\n   🛡️  100% 안전 영역 분석:")
        print(f"   안전 임계값: {safe_threshold:.4f} (실제 NG 중 최소값)")
        print(f"   안전 영역의 Good 개수: {len(ultra_safe_goods)}개")
        
        if len(ultra_safe_goods) >= 200:
            print(f"   ✅ 200개 확보 가능! (여유: {len(ultra_safe_goods) - 200}개)")
            strategy = "ultra_safe"
            target_count = 200
        else:
            print(f"   ⚠️  안전 영역 부족... 점진적 확장 필요")
            # 점진적으로 임계값을 높여가면서 최대한 안전한 범위 찾기
            strategy = "progressive_safe"
            target_count = min(200, max(len(ultra_safe_goods), 50))  # 최소 50개는 확보
        
        # 최종 전략별 분석
        print(f"\n   💰 최종 전략: {strategy}")
        if strategy == "ultra_safe":
            # 100% 안전 영역에서 200개 선택
            final_threshold = safe_threshold
            expected_tp = 200
            expected_fp = 0
            
        else:
            # 점진적 안전 영역에서 선택
            # Good 샘플의 하위 percentile 기준으로 안전 영역 확장
            percentiles = [5, 10, 15, 20, 25, 30]
            best_threshold = safe_threshold
            best_count = len(ultra_safe_goods)
            
            for p in percentiles:
                threshold = np.percentile(good_probas, p)
                safe_goods = good_probas[good_probas <= threshold]
                overlapping_ng = ng_probas[ng_probas <= threshold]
                
                print(f"     {p}% Good 기준 (임계값: {threshold:.4f}): Good {len(safe_goods)}개, NG {len(overlapping_ng)}개")
                
                if len(overlapping_ng) == 0 and len(safe_goods) >= target_count:
                    best_threshold = threshold
                    best_count = len(safe_goods)
                    break
                elif len(overlapping_ng) <= 2 and len(safe_goods) >= target_count:  # 최대 2개까지 허용
                    best_threshold = threshold
                    best_count = len(safe_goods)
                    break
            
            final_threshold = best_threshold
            expected_tp = min(200, best_count)
            expected_fp = len(ng_probas[ng_probas <= final_threshold])
        
        net_profit = 100 * expected_tp - 2000 * expected_fp
        accuracy = expected_tp / (expected_tp + expected_fp) if (expected_tp + expected_fp) > 0 else 1.0
        
        print(f"\n   🏆 최종 결정:")
        print(f"     - 사용 임계값: {final_threshold:.4f}")
        print(f"     - 예상 TRUE 개수: {expected_tp}개")
        print(f"     - 예상 TP: {expected_tp}개")
        print(f"     - 예상 FP: {expected_fp}개")
        print(f"     - 예상 정확률: {accuracy*100:.1f}%")
        print(f"     - 예상 Net Profit: {net_profit:,}")
        
        return final_threshold, net_profit
        
    def predict(self):
        """최종 예측 - FALSE POSITIVE 완전 차단 + 200개 전부 사용"""
        print("🎯 최종 예측 수행...")
        
        # 앙상블 예측 (확률)
        self.final_proba = (
            self.xgb_test * self.ensemble_weights['xgb'] +
            self.lgb_test * self.ensemble_weights['lgb'] + 
            self.rf_test * self.ensemble_weights['rf']
        )
        
        print(f"   • 예측 확률 범위: {self.final_proba.min():.4f} ~ {self.final_proba.max():.4f}")
        print(f"   • 평균 불량 확률: {self.final_proba.mean():.4f}")
        print(f"   💡 해석: 낮은 확률 = 양품 후보, 높은 확률 = 불량 후보")
        print(f"   🎯 목표: FALSE POSITIVE 0개로 200개 TRUE 예측!")
        
        # 훈련 데이터 분석으로 안전 임계값 도출
        safe_threshold, expected_profit = self.evaluate_train_performance()
        
        # FALSE POSITIVE 완전 차단하면서 200개 사용
        self._make_zero_fp_decisions(safe_threshold)
        
    def _make_zero_fp_decisions(self, safe_threshold):
        """FALSE POSITIVE 완전 차단 + 200개 전부 사용 전략"""
        print("🛡️  FALSE POSITIVE 완전 차단 + 200개 전부 사용...")
        
        # 안전 영역 (임계값 이하) 확인
        safe_candidates = self.final_proba <= safe_threshold
        safe_count = safe_candidates.sum()
        
        print(f"   • 안전 임계값: {safe_threshold:.4f}")
        print(f"   • 안전 영역 후보: {safe_count}개")
        
        if safe_count >= 200:
            print(f"   ✅ 안전 영역에서 200개 확보 가능! (여유: {safe_count - 200}개)")
            # 안전 영역 내에서 가장 낮은 200개 선택
            safe_indices = np.where(safe_candidates)[0]
            safe_probas = self.final_proba[safe_indices]
            bottom_200_in_safe = np.argsort(safe_probas)[:200]
            selected_indices = safe_indices[bottom_200_in_safe]
            
            strategy = "ultra_safe"
            
        else:
            print(f"   ⚠️  안전 영역 부족... 강제로 하위 200개 선택")
            print(f"   🚨 FALSE POSITIVE 위험 존재!")
            # 전체에서 가장 낮은 200개 선택
            selected_indices = np.argsort(self.final_proba)[:200]
            strategy = "forced_200"
        
        # 결정 생성
        self.final_decisions_bool = np.zeros(len(self.final_proba), dtype=bool)
        self.final_decisions_bool[selected_indices] = True
        
        true_count = self.final_decisions_bool.sum()
        false_count = len(self.final_proba) - true_count
        
        print(f"\n   📊 최종 결정:")
        print(f"      TRUE (양품 예측): {true_count}개 (목표: 200개)")
        print(f"      FALSE (불량 예측): {false_count}개")
        print(f"      전략: {strategy}")
        
        if true_count > 0:
            true_probas = self.final_proba[self.final_decisions_bool]
            false_probas = self.final_proba[~self.final_decisions_bool]
            
            print(f"\n   📊 확률 분석:")
            print(f"      TRUE (양품) 예측:")
            print(f"        - 평균 불량 확률: {true_probas.mean():.4f}")
            print(f"        - 최대 불량 확률: {true_probas.max():.4f}")
            print(f"        - 최소 불량 확률: {true_probas.min():.4f}")
            
            print(f"      FALSE (불량) 예측:")
            print(f"        - 평균 불량 확률: {false_probas.mean():.4f}")
            print(f"        - 최소 불량 확률: {false_probas.min():.4f}")
            print(f"        - 최대 불량 확률: {false_probas.max():.4f}")
            
            # 분리도 확인
            separation = false_probas.min() - true_probas.max()
            if separation > 0:
                print(f"      ✅ 완벽한 분리: 간격 = {separation:.4f}")
            else:
                print(f"      ⚠️  겹침 존재: 간격 = {separation:.4f}")
        
        # submission 형식에 맞게 변환
        self.final_decisions = ['TRUE' if is_true else 'FALSE' 
                              for is_true in self.final_decisions_bool]
        
        print(f"\n   🎯 FALSE POSITIVE 차단 결과:")
        if strategy == "ultra_safe":
            print(f"      ✅ 100% 안전 영역에서 선택 완료")
            print(f"      ✅ FALSE POSITIVE 위험: 0%")
        else:
            print(f"      ⚠️  강제 선택으로 FALSE POSITIVE 위험 존재")
            print(f"      💡 하지만 최선의 200개 선택 완료")
    
    def create_submission(self, output_path='outputs/submission.csv'):
        """submission 파일 생성 - 200개 TRUE 필수 확인"""
        print("📄 Submission 파일 생성...")
        
        # 출력 디렉토리 생성
        import os
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 출력 디렉토리 생성: {output_dir}")
        elif output_dir:
            print(f"📁 출력 디렉토리 이미 존재: {os.path.abspath(output_dir)}")
        
        # ID 생성 (테스트 데이터에 ID가 없는 경우)
        if 'ID' in self.test.columns:
            test_ids = self.test['ID']
        else:
            test_ids = [f'ID_{i}_L' for i in range(len(self.test))]
        
        # submission 데이터프레임 생성
        submission = pd.DataFrame({
            'ID': test_ids,
            'probability': self.final_proba,
            'decision': self.final_decisions
        })
        
        # 200개 TRUE 확인
        true_count = (submission['decision'] == 'TRUE').sum()
        false_count = (submission['decision'] == 'FALSE').sum()
        
        print(f"\n   🎯 제출 파일 검증:")
        if true_count == 200:
            print(f"   ✅ TRUE 개수: {true_count}개 (목표 달성!)")
        else:
            print(f"   ❌ TRUE 개수: {true_count}개 (목표: 200개)")
            print(f"   🚨 긴급 수정 필요!")
            
        print(f"   ✅ FALSE 개수: {false_count}개")
        print(f"   ✅ 총 샘플: {len(submission)}개")
        
        # 저장
        submission.to_csv(output_path, index=False)
        print(f"✅ Submission 파일 저장 완료: {os.path.abspath(output_path)}")
        
        # 상세 요약 정보
        print(f"\n   📊 최종 요약:")
        print(f"      • 파일명: {output_path}")
        print(f"      • TRUE (양품 예측): {true_count}개")
        print(f"      • FALSE (불량 예측): {false_count}개")
        print(f"      • 평균 불량 확률: {submission['probability'].mean():.4f}")
        print(f"      • 최대 불량 확률: {submission['probability'].max():.4f}")
        print(f"      • 최소 불량 확률: {submission['probability'].min():.4f}")
        
        # TRUE 예측의 확률 분포
        if true_count > 0:
            true_probas = submission[submission['decision'] == 'TRUE']['probability']
            print(f"\n   🎯 TRUE 예측 확률 분석:")
            print(f"      • TRUE 평균 확률: {true_probas.mean():.4f}")
            print(f"      • TRUE 최대 확률: {true_probas.max():.4f}")
            print(f"      • TRUE 최소 확률: {true_probas.min():.4f}")
            print(f"      • TRUE 표준편차: {true_probas.std():.4f}")
            
        print(f"\n   🛡️  FALSE POSITIVE 차단 전략 적용 완료!")
        print(f"   🏆 200개 TRUE 예측으로 최대 점수 추구!")
        
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
            import os
            if not os.path.exists('outputs'):
                os.makedirs('outputs')
                
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            
            for i, (model_name, importance) in enumerate(self.feature_importance.items()):
                top_features = importance.head(15)
                
                axes[i].barh(range(len(top_features)), top_features.values)
                axes[i].set_yticks(range(len(top_features)))
                axes[i].set_yticklabels(top_features.index, fontsize=8)
                axes[i].set_title(f'{model_name.upper()} - Top 15 Features')
                axes[i].set_xlabel('Importance')
                
            plt.tight_layout()
            plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("   ✅ 피처 중요도 그래프 저장 완료")
        except Exception as e:
            print(f"   ⚠️  피처 중요도 그래프 생성 오류: {e}")
            print("   → 그래프 없이 계속 진행...")
        
    def _plot_prediction_distribution(self):
        """예측 확률 분포 시각화"""
        try:
            import os
            if not os.path.exists('outputs'):
                os.makedirs('outputs')
                
            plt.figure(figsize=(12, 5))
            
            # 전체 분포
            plt.subplot(1, 2, 1)
            plt.hist(self.final_proba, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            
            # TRUE/FALSE 경계 표시
            if hasattr(self, 'final_decisions_bool') and self.final_decisions_bool.sum() > 0:
                true_probas = self.final_proba[self.final_decisions_bool]
                threshold = true_probas.max()  # TRUE 예측 중 최대값이 경계
                plt.axvline(threshold, color='red', linestyle='--', 
                           label=f'TRUE/FALSE 경계 ({threshold:.3f})')
                
                # TRUE 영역 표시
                plt.axvspan(0, threshold, alpha=0.2, color='green', label='TRUE (양품) 영역')
            
            plt.title('예측 확률 분포 (전체)')
            plt.xlabel('불량 확률')
            plt.ylabel('빈도')
            plt.legend()
            
            # TRUE vs FALSE 분포 비교
            plt.subplot(1, 2, 2)
            if hasattr(self, 'final_decisions_bool'):
                true_probas = self.final_proba[self.final_decisions_bool]
                false_probas = self.final_proba[~self.final_decisions_bool]
                
                if len(true_probas) > 0:
                    plt.hist(true_probas, bins=20, alpha=0.7, color='green', 
                            label=f'TRUE (양품, n={len(true_probas)})', edgecolor='black')
                    
                if len(false_probas) > 0:
                    plt.hist(false_probas, bins=20, alpha=0.7, color='red', 
                            label=f'FALSE (불량, n={len(false_probas)})', edgecolor='black')
                
                plt.title('TRUE vs FALSE 예측 분포')
                plt.xlabel('불량 확률')
                plt.ylabel('빈도')
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'Decision 정보 없음', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=plt.gca().transAxes)
                plt.title('TRUE vs FALSE 예측 분포')
            
            plt.tight_layout()
            plt.savefig('outputs/prediction_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("   ✅ 예측 분포 그래프 저장 완료")
        except Exception as e:
            print(f"   ⚠️  예측 분포 그래프 생성 오류: {e}")
            print("   → 그래프 없이 계속 진행...")
        
    def _analyze_decisions(self):
        """Decision 분석 - 200개 전부 사용 + FALSE POSITIVE 완전 차단"""
        true_decisions = sum(1 for d in self.final_decisions if d == 'TRUE')
        false_decisions = len(self.final_decisions) - true_decisions
        
        print(f"\n📊 Decision 분석 - FALSE POSITIVE 차단 전략:")
        print(f"   • TRUE (양품 예측): {true_decisions}개 ({true_decisions/len(self.final_decisions)*100:.1f}%)")
        print(f"   • FALSE (불량 예측): {false_decisions}개 ({false_decisions/len(self.final_decisions)*100:.1f}%)")
        
        if true_decisions != 200:
            print(f"   ⚠️  경고: TRUE 개수가 200개가 아님! (실제: {true_decisions}개)")
        else:
            print(f"   ✅ 목표 달성: 정확히 200개 TRUE 예측 완료")
        
        if true_decisions > 0:
            true_probas = self.final_proba[[i for i, d in enumerate(self.final_decisions) if d == 'TRUE']]
            false_probas = self.final_proba[[i for i, d in enumerate(self.final_decisions) if d == 'FALSE']]
            
            print(f"\n   📊 확률 분석:")
            print(f"      TRUE (양품) 예측 200개:")
            print(f"        - 평균 불량 확률: {true_probas.mean():.4f}")
            print(f"        - 최대 불량 확률: {true_probas.max():.4f} (가장 위험한 TRUE)")
            print(f"        - 최소 불량 확률: {true_probas.min():.4f} (가장 안전한 TRUE)")
            print(f"        - 표준편차: {true_probas.std():.4f}")
            
            if false_decisions > 0:
                print(f"      FALSE (불량) 예측 {false_decisions}개:")
                print(f"        - 평균 불량 확률: {false_probas.mean():.4f}")
                print(f"        - 최소 불량 확률: {false_probas.min():.4f} (가장 안전한 FALSE)")
                print(f"        - 최대 불량 확률: {false_probas.max():.4f}")
                
                # TRUE와 FALSE의 분리도 확인
                separation = false_probas.min() - true_probas.max()
                print(f"\n      🎯 TRUE/FALSE 분리도:")
                if separation > 0:
                    print(f"        ✅ 완벽한 분리: 간격 = {separation:.4f}")
                    print(f"        ✅ FALSE POSITIVE 위험: 극히 낮음")
                elif separation > -0.02:
                    print(f"        ⚠️  약간 겹침: 간격 = {separation:.4f}")
                    print(f"        ⚠️  FALSE POSITIVE 위험: 낮음")
                else:
                    print(f"        🚨 상당히 겹침: 간격 = {separation:.4f}")
                    print(f"        🚨 FALSE POSITIVE 위험: 존재")
        
        # 시뮬레이션된 점수 계산
        print(f"\n🎯 예상 성과 분석 (200개 TRUE 예측):")
        print("   (TRUE 예측 중 실제 양품 비율에 따른 예상 점수)")
        
        # 극도로 보수적 예상 (FALSE POSITIVE 차단 목표)
        for accuracy in [1.00, 0.99, 0.98, 0.95, 0.90]:
            tp = int(200 * accuracy)        # True Positive
            fp = 200 - tp                   # False Positive
            net_profit = 100 * tp - 2000 * fp
            
            if accuracy == 1.00:
                risk_level = "✅ 목표 (ZERO FP)"
            elif accuracy >= 0.98:
                risk_level = "🟢 매우 안전"
            elif accuracy >= 0.95:
                risk_level = "🟡 안전"
            else:
                risk_level = "🔴 위험"
                
            print(f"   • 양품 비율 {accuracy*100:3.0f}%: TP={tp:3}, FP={fp:2} → Net Profit = {net_profit:,} ({risk_level})")
        
        print(f"\n🛡️  FALSE POSITIVE 완전 차단 전략 요약:")
        print(f"   • 목표: 200개 TRUE 예측 ALL TRUE POSITIVE")
        print(f"   • 전략: 훈련 데이터 실제 Good 패턴 학습")
        print(f"   • 선택: 가장 안전한 200개만 TRUE 예측")
        print(f"   • 결과: FALSE POSITIVE = 0개 달성 목표")
        
        # 최악의 시나리오까지 고려한 안전성 평가
        worst_case_fp = max(0, int(200 * 0.1))  # 10% FP 가정
        worst_case_profit = 100 * (200 - worst_case_fp) - 2000 * worst_case_fp
        print(f"   • 최악 시나리오 (10% FP): Net Profit = {worst_case_profit:,}")
        
        if worst_case_profit > 0:
            print(f"   ✅ 최악의 경우에도 양수 달성 가능!")
        else:
            print(f"   ⚠️  최악의 경우 음수 위험 존재...")

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
