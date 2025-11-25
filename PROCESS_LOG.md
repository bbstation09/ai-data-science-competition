# Project Process Log

## 1. 초기 환경 준비 & 데이터 살펴보기

- `raw_data/train.csv` / `test.csv` 구조 확인 (설계 스펙: Width/Aspect/Inch, 공정 변수: Proc_Param1~11, FEM 좌표/압력 x/y/p0~255, 그룹 G1~G4 등).
- 타깃 분포: Good 613 / NG 107 (NG 비율 ~15%).
- 기본 EDA:
  - `Mass_Pilot` vs Class, `Width/Aspect/Inch` 구간별 히스토그램, 공장별 Good/NG 개수·비율 그래프 작성.
  - Proc_Param1~11의 분포를 percentile 기반 bin으로 나누어 Good/NG 분포를 1장의 그림(`proc_params_class_distribution.png`)으로 정리.
  - X/Y 접지 좌표, 256개 압력(p) 노드가 타이어 모양/압력 패턴을 담고 있다는 domain insight를 문서화.

## 2. 데이터 전처리 & Feature Engineering

### 2-1. 기본 전처리

- `Mass_Pilot`를 int로 변환, `Plant`는 카테고리 코드/one-hot.
- Proc_Param, X/Y/p는 문자열 가능성이 있어 `pd.to_numeric(errors='coerce')` 처리로 숫자화.
- train/test를 동일 로직으로 가공하고, 마지막에 drop_cols로 원본 좌표/압력 컬럼 제거하여 메모리 감소.
- Width/Aspect/Inch로 단면 높이(H), 림 직경(mm), 외경(mm), 단면 면적, 용적 근사 등을 계산하고, 각 치수에 대해 분위수 NG-rate 파생 + Plant별 z-score를 추가.
- (Width, H, Rim(mm), Outer(mm))를 표준화 후 KMeans(k=4~8)으로 사이즈 군집을 전수 평가하고, `max(ng_rate)-min(ng_rate)`가 가장 큰 k=6을 선택해 라벨/NG rate(`TireSize_cluster_NG_rate`) 부여. 결과: k=6에서 NG rate가 4.9%~28.5%까지 벌어지고, 각 군집 샘플 수 70~190개 수준으로 안정적 (`outputs/tire_size_cluster_stats.csv`, `outputs/figures/tire_size_clusters.png`, `tire_size_cluster_ng_rate.png` 참조).
- G1~G4 개별 값에 대해 z-score와 5분위 NG-rate 파생(`Gk_zscore`, `Gk_bin_NG_rate`)을 생성해, 음수 구간 등 위험 영역 정보를 명시적으로 제공 (`outputs/figures/g_features_distribution.png`, `g_features_ng_rate.png` 참고).

### 2-2. 파생 피처 설계 목적

- **설계 스펙 조합**: Width/Aspect/Inch 비율·곱(`Width_div_Aspect`, `Width_mul_Aspect`, 등) + 단면 높이/외경/림 직경 파생 → 타이어 물리적 크기 차이가 NG에 미치는 영향 확인.
- **Proc_Param 통계**: 평균/표준편차/최솟값/범위 → 공정 조건의 변동성과 이상치 탐색.
- **그룹 G1~G4**: 평균·표준편차·합·곱·비율 → 제조 Stiffness/Elasticity 관련 지표로 해석, NG 연관성 파악.
- **Footprint 좌표(X1~X5, Y1~Y5)**: 폭(`pos_width`), 높이(`pos_yrange`), 중심 편차 → 접지 형태의 기하학 정보 추출.
- **FEM 압력(p0~p255)**: 평균/표준편차/최댓값/사분위수/범위/왜도/첨도 + 세그먼트별 평균/표준편차 → 접지 압력 분포와 NG의 상관도를 찾기 위함.
- **압력 중심(Center of Pressure)**: 압력 가중 평균 좌표 vs 기하학적 중심 차이(`Pressure_Asymmetry_X/Y/Total`, `Cop_shift`) → 타이어가 한쪽으로 치우쳤는지 여부 반영.
- **접지 면적/불균형**: `Contact_ratio`, 구간별 압력 평균(초기·중간·후기) → 특정 구간에서 압력이 과도한지 확인.
- **변화 패턴**: 압력·좌표 차분(`P_Diff_*`, `X_Diff_*`, `Y_Diff_*`) → 급격한 변동이 NG에 영향 주는지 파악.

- 추가해야할 것:

  1.  **정보 손실/누락**: FEM 좌표·압력을 요약하는 과정에서 NG 패턴에 중요한 “세부 형태”가 빠졌을 수 있습니다. 예를 들어 FEM 2D 맵을 CNN/AutoEncoder로 직접 학습하거나, x/y/p를 그대로 쓰는 1D Conv 모델을 추가하지 않았기 때문에 모델이 잡지 못하는 신호가 남아 있을 가능성이 큽니다.
  2.  **클래스 불균형**: Good 613 vs NG 107 구조에선 NG를 놓치면 손실이 20배 크기 때문에, 단순 re-weighting만으론 부족할 수 있습니다. Focal loss, oversampling, contrastive learning 등 추가 대응이 필요합니다.

- **설계 치수 군집**: `Width/H/Rim/Outer` 4차원 공간에서 표준화 후 KMeans로 군집화, 각 군집의 NG rate를 별도 컬럼으로 주입해 “비슷한 실측 크기” 그룹의 위험도 반영.
- **PCA/클러스터**: FEM 통계/pressure features를 다량 생성한 뒤 SHAP 기반 pruning 옵션으로 중요도 낮은 피처를 제거. PCA 자체는 특定 모듈에서 `add_fem_pca`로 사용(주 성분으로 FEM 패턴 요약 → 고차원 압력 데이터를 요약하여 모델이 NG와 연관된 주요 변동 축을 학습할 수 있도록 하기 위함). 예: FEM PCA 상위 성분이 모델 중요도 상단에 올라, NG 패턴을 잡는 데 기여.
- **클러스터(KMeans)**: FEM field vector를 군집화하여 특정 패턴(예: 특정 공정/압력 분포)을 하나의 레이블로 표현, NG와 관련된 클러스터 여부 파악.
- **비선형 조합**: `add_pairwise_nonlinear_features`로 상위 후보 피처 간 곱/차/비율/루트 조합을 생성해 비선형 상호작용 반영.
- **SHAP 기반 Pruning**: LightGBM 모델로 SHAP 값을 산출한 뒤 상위 중요 피처 비율만 유지(`--enable_shap_pruning`). 피처 폭증 시 잡음 제거와 계산 효율성을 위해 적용.

## 3. Task1 모델 파이프라인

1. **multi-model 학습**: LightGBM/XGBoost/CatBoost/ExtraTrees/TabNet (옵션) 5-fold cross validation.
2. **Optuna 튜닝 지원**: `--n_trials` 조정 가능, `--skip_tuning` 시 `outputs/best_params.json` 재사용.
3. **블렌딩 & 스태킹**:
   - 저장된 OOF 예측을 rank/geometric mean으로 블렌딩 (`build_blend_results`).
   - Stacker(Logistic/LGBM) OOF와 비교해서 AUC가 더 나은 조합을 `selected_ensemble`로 사용.
4. **Calibration**: Platt/Isotonic/Temperature + mean ensemble. 각 보정 후 파일 저장(`task1_calibrated_*.csv`), AUC & Brier 기록. 최적 보정 결과만 `task1_ensemble_{oof,test}.csv`로 쓰임.
5. **재사용 옵션**: `--reuse_predictions`로 학습 없이 저장된 Level-1 예측을 읽어 다시 보정/Task2 실험 가능.

## 4. Task2 Net Profit 최적화

1. **Threshold/Top-N grid search**: `src/task2_threshold_grid_search.py`
   - Threshold 0.80–0.95 / step=0.0005, Top-N 20–200 / step=1.
   - `--score_metric {profit, profit_adjusted}`, `--risk_penalty` 옵션으로 NG 위험 가중치 변경.
   - NetProfit 시각화(`outputs/figures/task2_net_profit_curve.png`).
2. **Calibrated probability 비교**: 이소토닉/Platt/Temperature/Rank-blend 보정 확률을 각각 Task2에 적용하여 최적 전략 탐색.
3. **DecisionTree 안전 리프**: 간단한 Tree로 NG 비율이 낮은 leaf를 찾아, 해당 leaf에 속한 test ID를 “안전 후보군”으로 사용.
   - Leaf 12 & 19: NG=0 (91개 ID) → 규정 내에서 확실히 좋은 케이스.
   - Leaf 3/10/17: NG 비율 5–10% → Threshold 순으로 일부만 추가해 최대 100개 ID만 선택 (규정을 지키며 NetProfit을 크게 확보).
4. **최종 Task2 로직**: 안전 리프 순으로 ID를 모아 100개 (=200행)만 `decision=True`로 설정 → 훈련 기준 NetProfit 10,000 이상.

## 5. 최종 제출 흐름

1. `python src/build_features.py [옵션]`
2. `python src/train_task1_ensemble.py [...]` (또는 `--reuse_predictions`)
3. `python src/task2_threshold_grid_search.py --score_metric profit --risk_penalty 0`
4. Tree 기반 안전 리프 스크립트로 `safe_ids_100.csv` 생성 → `task2_decision_optimal.csv` 업데이트.
5. `python src/make_final_submission.py --task1 outputs/task1_ensemble_submission.csv --task2 outputs/task2_decision_optimal.csv --output outputs/final_submission.csv`

## 6. EDA 및 시각화 산출물

- `mass_pilot_class_distribution.png`: Mass_Pilot True/False에 대한 Good/NG 비율.
- `width_aspect_inch_combined.png`: Width/Aspect/Inch binned distribution & NG rate.
- `plant_counts_ngrate.png`: 공장별 Good/NG 개수 + NG rate (막대+선).
- `proc_params_class_distribution.png`: Proc_Param1~11 구간별 Good/NG 히스토그램.
- `plant_class_distribution.png`, `plant_ng_rate.png` 등.
- `tire_size_clusters.png`: Width/H/Rim/Outer 기반 KMeans 군집 결과 시각화.
- `tire_size_cluster_ng_rate.png`: 선택된 k=6 군집별 NG rate 및 샘플 수 막대 차트.
- `g_features_distribution.png` / `g_features_ng_rate.png`: G1~G4 분포 및 분위수별 NG rate 비교.
- `task2_threshold_grid_results.csv`: Top-N/Threshold 그리드 탐색 결과(최근 Top-N=183 전략 기준).

## 7. 기타 실험

- `0.38403/` 폴더: 초기에 0.38403 점수로 제출되었던 스크립트(gemini.py 등) 보관.
- `custom_lgb_{oof,test}.csv`: 별도 LGBM 실험(OOF AUC≈0.697) 결과.
- 단순한 `train_engineered_simple.csv` / `test_engineered_simple.csv`는 EDA/안전 리프 탐색용 피처.
- `train_hybrid_models.py`: LightGBM/XGBoost/CatBoost/TabNet/FEM-CNN을 순차 학습하고 스태킹/보정을 수행하기 위한 새 파이프라인을 추가. 현 환경에서 LightGBM 다중 폴드 학습 시 세그멘테이션 오류가 발생하여 실행은 완료하지 못했으며(스크립트만 저장), 향후 재시도 필요.

## 8. 최신 상태

- Task1 확률: `task1_calibrated_isotonic_test.csv`. OOF AUC ≈ 0.74.
- Task2 결정: 최신 제출은 Top-N=183(확률 기준) 전략을 적용하여 base ID 183개(L/P 복제 후 366행)만 `decision=True`로 설정. (이전 안전 리프 버전은 `0.10556/` 폴더에 보관.)
- 최종 제출: `outputs/final_submission.csv` (932행). 이 파일을 리더보드에 제출하면 됩니다.
