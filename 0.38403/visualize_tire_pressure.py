"""
타이어 접지 압력 시각화
x, y 좌표에 점을 찍고 p 값으로 색상 표현
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
print("데이터 로드 중...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def visualize_tire_sample(df, index, save_path=None):
    """
    특정 샘플의 타이어 접지 압력 시각화

    Parameters:
    -----------
    df : DataFrame
        train 또는 test 데이터
    index : int
        시각화할 샘플 인덱스
    save_path : str, optional
        저장 경로
    """
    # 해당 행 데이터 추출
    sample = df.iloc[index]

    # x, y, p 값 추출
    x_cols = [f'x{i}' for i in range(256)]
    y_cols = [f'y{i}' for i in range(256)]
    p_cols = [f'p{i}' for i in range(256)]

    x_values = sample[x_cols].values
    y_values = sample[y_cols].values
    p_values = sample[p_cols].values

    # 메타 정보
    sample_id = sample['ID'] if 'ID' in sample.index else f'Index_{index}'
    sample_class = sample.get('Class', 'Unknown')

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Scatter plot with color by pressure
    scatter = axes[0].scatter(x_values, y_values, c=p_values,
                             cmap='YlOrRd', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

    axes[0].set_xlabel('X 좌표 (mm)', fontsize=12)
    axes[0].set_ylabel('Y 좌표 (mm)', fontsize=12)
    axes[0].set_title(f'타이어 접지 압력 분포\nID: {sample_id}, Class: {sample_class}',
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', adjustable='box')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('압력 (p)', fontsize=11)

    # 2. 3D-like visualization (contour plot)
    # 그리드 생성
    xi = np.linspace(x_values.min(), x_values.max(), 100)
    yi = np.linspace(y_values.min(), y_values.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # 보간 (Interpolation)
    from scipy.interpolate import griddata
    zi = griddata((x_values, y_values), p_values, (xi, yi), method='cubic')

    # Contour plot
    contour = axes[1].contourf(xi, yi, zi, levels=20, cmap='YlOrRd', alpha=0.8)
    axes[1].scatter(x_values, y_values, c='black', s=10, alpha=0.5, label='실제 측정점')

    axes[1].set_xlabel('X 좌표 (mm)', fontsize=12)
    axes[1].set_ylabel('Y 좌표 (mm)', fontsize=12)
    axes[1].set_title(f'압력 분포 (보간)\nID: {sample_id}',
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].legend()

    # Colorbar
    cbar2 = plt.colorbar(contour, ax=axes[1])
    cbar2.set_label('압력 (p)', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"저장 완료: {save_path}")

    plt.show()

    # 통계 정보 출력
    print(f"\n{'='*60}")
    print(f"샘플 정보: {sample_id}")
    print(f"{'='*60}")
    print(f"Class: {sample_class}")
    print(f"\n[압력 통계]")
    print(f"  평균: {p_values.mean():.4f}")
    print(f"  표준편차: {p_values.std():.4f}")
    print(f"  최소: {p_values.min():.4f}")
    print(f"  최대: {p_values.max():.4f}")
    print(f"\n[좌표 범위]")
    print(f"  X: [{x_values.min():.2f}, {x_values.max():.2f}]")
    print(f"  Y: [{y_values.min():.2f}, {y_values.max():.2f}]")
    print(f"{'='*60}\n")

def visualize_multiple_samples(df, indices, save_dir='output'):
    """
    여러 샘플을 비교 시각화

    Parameters:
    -----------
    df : DataFrame
        데이터프레임
    indices : list
        시각화할 샘플 인덱스 리스트
    save_dir : str
        저장 디렉토리
    """
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_samples = len(indices)
    fig, axes = plt.subplots(2, n_samples, figsize=(6*n_samples, 10))

    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, idx in enumerate(indices):
        sample = df.iloc[idx]

        # x, y, p 값 추출
        x_cols = [f'x{i}' for i in range(256)]
        y_cols = [f'y{i}' for i in range(256)]
        p_cols = [f'p{i}' for i in range(256)]

        x_values = sample[x_cols].values
        y_values = sample[y_cols].values
        p_values = sample[p_cols].values

        sample_id = sample['ID'] if 'ID' in sample.index else f'Index_{idx}'
        sample_class = sample.get('Class', 'Unknown')

        # Scatter plot
        scatter = axes[0, col_idx].scatter(x_values, y_values, c=p_values,
                                          cmap='YlOrRd', s=30, alpha=0.7,
                                          edgecolors='black', linewidth=0.3)
        axes[0, col_idx].set_xlabel('X')
        axes[0, col_idx].set_ylabel('Y')
        axes[0, col_idx].set_title(f'{sample_id}\nClass: {sample_class}', fontsize=11)
        axes[0, col_idx].set_aspect('equal', adjustable='box')
        axes[0, col_idx].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, col_idx], label='Pressure')

        # Contour plot
        xi = np.linspace(x_values.min(), x_values.max(), 80)
        yi = np.linspace(y_values.min(), y_values.max(), 80)
        xi, yi = np.meshgrid(xi, yi)

        from scipy.interpolate import griddata
        zi = griddata((x_values, y_values), p_values, (xi, yi), method='cubic')

        contour = axes[1, col_idx].contourf(xi, yi, zi, levels=15, cmap='YlOrRd')
        axes[1, col_idx].set_xlabel('X')
        axes[1, col_idx].set_ylabel('Y')
        axes[1, col_idx].set_title(f'Contour: {sample_id}', fontsize=11)
        axes[1, col_idx].set_aspect('equal', adjustable='box')
        plt.colorbar(contour, ax=axes[1, col_idx], label='Pressure')

    plt.suptitle('타이어 접지 압력 비교', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = f'{save_dir}/comparison_samples.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"비교 그래프 저장: {save_path}")
    plt.show()

# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("타이어 접지 압력 시각화")
    print("="*80)

    # 예시 1: Train 데이터의 첫 번째 샘플
    print("\n[예시 1] Train 데이터 첫 번째 샘플 (Good)")
    visualize_tire_sample(train, 0, save_path='output/tire_sample_0_good.png')

    # 예시 2: NG 샘플 찾아서 시각화
    ng_samples = train[train['Class'] == 'NG']
    if len(ng_samples) > 0:
        print("\n[예시 2] NG 샘플")
        ng_index = ng_samples.index[0]
        visualize_tire_sample(train, ng_index, save_path='output/tire_sample_ng.png')

    # 예시 3: 여러 샘플 비교 (Good 2개, NG 2개)
    print("\n[예시 3] 여러 샘플 비교")
    good_indices = train[train['Class'] == 'Good'].index[:2].tolist()
    ng_indices = train[train['Class'] == 'NG'].index[:2].tolist()
    compare_indices = good_indices + ng_indices

    visualize_multiple_samples(train, compare_indices)

    # 예시 4: Test 데이터 샘플
    print("\n[예시 4] Test 데이터 샘플")
    visualize_tire_sample(test, 0, save_path='output/tire_sample_test.png')

    print("\n" + "="*80)
    print("시각화 완료!")
    print("="*80)
    print("\n생성된 파일:")
    print("  - output/tire_sample_0_good.png")
    print("  - output/tire_sample_ng.png")
    print("  - output/comparison_samples.png")
    print("  - output/tire_sample_test.png")
