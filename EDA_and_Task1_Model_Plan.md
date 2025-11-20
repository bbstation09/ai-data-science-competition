
# 한국타이어 FEM 대회 EDA & Task1 모델링 전략

본 문서는 `train.csv`, `test.csv`를 기반으로 **EDA(탐색적 데이터 분석) 플랜**과  
**Task1(불량률 예측)용 최고 성능 모델 설계 전략**을 정리한 것이다.

---

## 1. 데이터 구조 이해

### 1.1 기본 정보

- Train: 720 × 799 (feature 798 + Class)
- Test: 466 × 799 (feature 799, Class 없음)
- 타깃: `Class` (Good / NG, 불균형 데이터)
  - Good: 613
  - NG: 107 (약 1:5.7 비율)

### 1.2 컬럼 그룹 구조

1. **설계 스펙 / 공정 파라미터**
   - `Mass_Pilot` (bool)
   - `Width`, `Aspect`, `Inch`
   - `Plant`
   - `Proc_Param1` ~ `Proc_Param11`

2. **위치 벡터(저차원)**
   - `X1` ~ `X5`, `Y1` ~ `Y5`

3. **FEM 시뮬레이션 고차원 데이터**
   - `x0` ~ `x255` : 256개 지점의 x 좌표
   - `y0` ~ `y255` : 256개 지점의 y 좌표
   - `p0` ~ `p255` : 각 지점의 물리량(압력/응력 등)

4. **FEM 변화량 통계**
   - `G1` ~ `G4`

---

## 2. EDA(탐색적 데이터 분석) 플랜

EDA의 목표는
- **(1) 불량 발생 패턴이 어디에서 나오느냐?**
- **(2) 어떤 피처 그룹이 중요한 신호를 갖고 있느냐?**
- **(3) 전처리/모델 설계를 어떻게 해야 하느냐?**  
를 파악하는 것이다.

### 2.1 기본 분포 확인

1. `Class` 비율 확인  
   - `value_counts(normalize=True)` → 불균형 정도 확인
2. 기초 통계
   - `train.describe()`로 수치형 피처 범위/평균/표준편차 확인
3. 결측치/이상치
   - `train.isna().mean().sort_values(ascending=False)`  
   - 필요 시 단순 대체(평균/중앙값) or 모델 내장 처리 사용

### 2.2 설계 스펙 & 공정 파라미터 EDA

#### 2.2.1 범주형

- `Mass_Pilot`, `Plant`
  - `pd.crosstab(train['Plant'], train['Class'])`  
  - 공장별 NG 비율 시각화 (bar plot)
  - Mass_Pilot별 NG 비율 비교

#### 2.2.2 수치형 설계 스펙

- `Width`, `Aspect`, `Inch`
  - NG vs Good에 대해 boxplot / kdeplot
  - 예: 폭이 넓을수록 NG 높은지 확인

#### 2.2.3 공정 파라미터

- `Proc_Param1` ~ `Proc_Param11`
  - 각 파라미터별로 NG/Good 분포 비교
    - 히스토그램 / KDE
  - 상관계수
    - `train[proc_cols].corr()`  
    - 다중공선성 높은지 확인

### 2.3 FEM 시뮬레이션 좌표(x, y) & 값(p) EDA

256 지점을 그대로 모두 보기에는 너무 많으므로,
**요약 통계 + 간단한 시각화** 위주로 진행한다.

#### 2.3.1 전체 통계

- `p0~p255`에 대해:
  - 전체 평균/표준편차/최소/최대
  - NG와 Good을 나눠 비교
    - `train[train['Class']=='NG'][p_cols].mean(axis=0)` vs Good
- `x0~x255`, `y0~y255`는 좌표이므로
  - 범위 및 grid 형태(예: x, y가 규칙적인지)만 확인

#### 2.3.2 위치별 특징

- NG에서 특히 값이 높은/낮은 지점 찾기
  - `mean_NG - mean_Good` 를 위치별로 계산
  - 차이가 큰 인덱스를 상위 10~20개 뽑아서 중요 포인트로 사용

#### 2.3.3 요약 피처 생성 아이디어(EDA 결과를 바탕으로)

- `p` 전체의
  - 평균, 표준편차, 최대, 최소
  - 상위 K개 포인트 평균 (e.g. 상위 5개, 10개)
  - 하위 K개 포인트 평균
- 특정 영역(예: bead, shoulder)에 해당하는 인덱스 그룹을 만들 수 있다면
  - 영역별 평균/표준편차
  - → domain knowledge가 있다면 추가

### 2.4 상관 관계 & 중요 피처 대략 보기

- 라벨 인코딩 후, LightGBM을 대충 학습시켜 **feature importance**를 먼저 본다.
  - `feature_importances_`를 이용해
    - 어떤 Proc_Param / G1~G4 / p 인덱스들이 중요한지 체크
- 이후 EDA에서 중요한 구간만 추가 분석

### 2.5 Train/Test 분포 차이(데이터 드리프트)

- 주요 피처(Width, Aspect, Inch, Plant, G1~G4, p 요약통계 등)에 대해  
  train vs test 분포 비교
  - 분포가 많이 다르면 domain shift 주의  
  - 필요시 reweighting 또는 robust 모델 사용

---

## 3. Task1 최고 성능 모델 설계 전략

### 3.1 문제 특성 정리

- 입력 차원: 798개의 피처 (고차원)
- 데이터 수: 720개 (적음 → 과적합 위험)
- 레이블: 심하게 불균형 (NG: 107, Good: 613)
- 피처 종류:
  - 표준 테이블 데이터(설계 스펙/공정)
  - grid-like FEM 데이터(256 포인트)

→ **트리 기반 앙상블 + 고차원 정규화 + 불균형 대응** 이 적합.

---

### 3.2 베이스라인 모델: Gradient Boosting (LightGBM / CatBoost)

#### 3.2.1 피처 전처리

- 범주형
  - `Mass_Pilot`, `Plant` → LightGBM/CatBoost에서 카테고리로 처리
- 수치형
  - 특별한 스케일링 없이 사용 가능
- FEM 데이터
  - `x0~x255`, `y0~y255`, `p0~p255` 그대로 사용 + 요약 통계 피처 추가

#### 3.2.2 불균형 처리

- LightGBM:
  - `is_unbalance = true` 또는 `scale_pos_weight = (num_negative / num_positive)`  
- CatBoost:
  - `scale_pos_weight` 사용
- 또는
  - Focal loss를 직접 구현 (advanced)

#### 3.2.3 교차 검증

- 데이터 수가 적으므로 **Stratified K-Fold (k=5 or 10)** 필수
  - `stratify=Class`
- 각 fold의 ROC-AUC를 평균 내어 모델 튜닝

#### 3.2.4 하이퍼파라미터 초기값 예시 (LightGBM)

- `num_leaves`: 31~64
- `max_depth`: 5~8
- `learning_rate`: 0.03~0.1
- `n_estimators`: 1000 이상 + early stopping
- `feature_fraction`: 0.7~0.9
- `bagging_fraction`: 0.7~0.9, `bagging_freq`: 1
- `min_data_in_leaf`: 20~50
- `lambda_l1`, `lambda_l2`: 0~10 범위 탐색

튜닝 방법:
- Optuna / Bayesian Optimization 또는  
- Grid / Random Search

---

### 3.3 고급 모델링: Tabular + FEM 특화 피처

#### 3.3.1 FEM 요약 피처 추가

FEM 256 포인트에 대해 다음과 같은 요약 피처를 추가:

- 전체 평균, 표준편차, 최대, 최소
- 상위 5/10/20 포인트 평균
- 하위 5/10/20 포인트 평균
- (가능하면) 영역별 평균 (예: shoulder, center 등)

이 요약 피처들을 별도 컬럼으로 추가하면
- 모델이 고차원 노이즈에 덜 민감해지고
- 중요한 구조적 정보를 더 잘 활용할 수 있다.

#### 3.3.2 차원 축소 + 트리

- `p0~p255`에 PCA를 적용 (e.g. 10~30개의 주성분)
- 원래 256차원 + PCA 피처를 함께 넣고 importance를 평가
- 의미 없는 원본 p 인덱스는 제거하기도 함

---

### 3.4 다른 접근: 1D CNN 융합 모델 (고급)

데이터 양이 적어서 overfitting 위험이 크지만,
FEM 데이터를 **1D 시퀀스**로 보고 CNN을 적용하는 방법도 있다.

- 입력1: `p0~p255` (1D 시퀀스)
- 입력2: 나머지 tabular 피처
- 구조:
  - 1D Conv → GlobalMaxPooling → dense
  - tabular는 별도 dense → 둘을 concat → 최종 출력(sigmoid)
- 손실: Binary Cross Entropy or Focal Loss
- 정규화: Dropout, L2, early stopping 강하게

이 방법은 구현 비용이 크고 튜닝이 필요하지만,
트리 기반 모델과 **블렌딩/앙상블**하면 성능 향상 여지가 있다.

---

### 3.5 앙상블 전략

최종 리더보드 성능을 위해:

1. LightGBM (원본 피처 + 요약 피처)
2. CatBoost (카테고리 강점 활용)
3. (선택) 1D CNN + Tabular NN

위 모델들의 예측 확률을
- 단순 평균 또는
- 가중 평균(검증 AUC 기반)으로 앙상블

→ 일반적으로 Kaggle에서 단일 모델보다 0.01~0.02 AUC 개선되는 경우가 많다.

---

### 3.6 실전 워크플로우 요약

1. **기본 EDA**
   - 분포, 결측, 불균형, 간단한 피처 중요도 확인
2. **기본 LightGBM 베이스라인**
   - 간단 튜닝 + 5-Fold CV
3. **FEM 요약 피처 생성**
   - 평균/표준편차/최대/상위 K 등 추가 후 다시 학습
4. **하이퍼파라미터 튜닝**
   - Optuna로 AUC 최대화
5. **CatBoost / 다른 모델 학습**
   - 같은 피처셋으로 멀티 모델 학습
6. **앙상블**
   - Out-of-fold / test 예측을 평균하여 최종 제출 확률 생성

이 과정을 따르면
- 도메인을 잘 이해한 상태에서
- 안정적인 높은 AUC를 얻고
- Task2의 profit 최적화(의사결정)는 그 다음 단계에서 설계할 수 있다.
