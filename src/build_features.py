#!/usr/bin/env python
"""
Advanced feature engineering pipeline for the tire FEM competition.
The script now includes SHAP 기반 feature pruning, FEM cluster labels,
pairwise 비선형 조합, 그리고 확장된 FEM 통계까지 모두 자동화한다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    import shap
except Exception:  # pragma: no cover - optional deps handled at runtime
    lgb = None
    shap = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "5-ai-and-datascience-competition"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SEED = 42
SIM_COLUMN_PATTERN = re.compile(r"^[xypXY][0-9]+$")


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def basic_preprocessing(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Unify raw categorical/numeric columns before feature engineering."""

    def _normalize_mass_pilot(df: pd.DataFrame) -> None:
        if "Mass_Pilot" not in df.columns:
            return
        numeric = pd.to_numeric(df["Mass_Pilot"], errors="coerce")
        df["Mass_Pilot"] = numeric.fillna(-1).astype(int)

    def _encode_plant(df: pd.DataFrame, categories: pd.Index) -> None:
        if "Plant" not in df.columns:
            return
        df["Plant"] = pd.Categorical(df["Plant"], categories=categories)
        dummies = pd.get_dummies(df["Plant"], prefix="Plant", dtype=int)
        df[dummies.columns] = dummies

    def _coerce_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> None:
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                # Guard against stray string tokens (e.g., spaces, '?') in numeric-looking fields.
                df[col] = pd.to_numeric(df[col], errors="coerce")

    for df in [train, test]:
        _normalize_mass_pilot(df)

    plant_sources = [df["Plant"] for df in [train, test] if "Plant" in df.columns]
    if plant_sources:
        combined = pd.concat(plant_sources, axis=0)
        categories = pd.Categorical(combined.dropna()).categories
        for df in [train, test]:
            _encode_plant(df, categories)

    numeric_candidates: set[str] = set()
    for df in [train, test]:
        for col in df.columns:
            if col.startswith("Proc_Param") or SIM_COLUMN_PATTERN.match(col):
                numeric_candidates.add(col)
    numeric_cols = sorted(numeric_candidates)
    for df in [train, test]:
        _coerce_numeric(df, numeric_cols)


def drop_simulation_source_columns(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Drop high-dimensional raw FEM coordinate/pressure columns after aggregation."""
    drop_cols = (
        [f"x{i}" for i in range(256)]
        + [f"y{i}" for i in range(256)]
        + [f"p{i}" for i in range(256)]
        + [f"X{i}" for i in range(1, 6)]
        + [f"Y{i}" for i in range(1, 6)]
    )
    for df in [train, test]:
        existing = [col for col in drop_cols if col in df.columns]
        if existing:
            df.drop(columns=existing, inplace=True)


def add_design_features(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series) -> None:
    eps = 1e-6
    for df in [train, test]:
        df["Width_div_Aspect"] = df["Width"] / (df["Aspect"] + eps)
        df["Width_div_Inch"] = df["Width"] / (df["Inch"] + eps)
        df["Aspect_div_Inch"] = df["Aspect"] / (df["Inch"] + eps)
        df["Width_mul_Aspect"] = df["Width"] * df["Aspect"]
        df["Width_mul_Inch"] = df["Width"] * df["Inch"]
        df["Aspect_mul_Inch"] = df["Aspect"] * df["Inch"]
        df["Width_sqrt"] = np.sqrt(np.clip(df["Width"], a_min=0, a_max=None))
        df["Aspect_sqrt"] = np.sqrt(np.clip(df["Aspect"], a_min=0, a_max=None))
        df["Inch_sqrt"] = np.sqrt(np.clip(df["Inch"], a_min=0, a_max=None))
        df["Width_log1p"] = np.log1p(np.clip(df["Width"], a_min=0, a_max=None))
        df["Aspect_log1p"] = np.log1p(np.clip(df["Aspect"], a_min=0, a_max=None))
        df["Inch_log1p"] = np.log1p(np.clip(df["Inch"], a_min=0, a_max=None))

    plant_stats = (
        train.groupby("Plant")[["Width", "Aspect", "Inch"]]
        .agg(["mean", "std"])
        .swaplevel(axis=1)
    )
    global_stats = {
        ("mean", "Width"): train["Width"].mean(),
        ("std", "Width"): train["Width"].std(),
        ("mean", "Aspect"): train["Aspect"].mean(),
        ("std", "Aspect"): train["Aspect"].std(),
        ("mean", "Inch"): train["Inch"].mean(),
        ("std", "Inch"): train["Inch"].std(),
    }

    def _apply_z(df: pd.DataFrame) -> None:
        aligned = plant_stats.reindex(df["Plant"]).copy()
        for key, value in global_stats.items():
            aligned[key].fillna(value, inplace=True)
        for col in ["Width", "Aspect", "Inch"]:
            df[f"{col}_zscore"] = (df[col] - aligned[("mean", col)].values) / (
                aligned[("std", col)].values + 1e-6
            )

    _apply_z(train)
    _apply_z(test)

    for col in ["Width", "Aspect", "Inch"]:
        add_bin_ng_rate_feature(train, test, column=col, series=train[col], target=target, q=6)


def add_tire_geometry_features(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series) -> None:
    """타이어 규격(폭/편평비/림)을 이용해 물리적 치수 피처 생성."""
    required_cols = {"Width", "Aspect", "Inch"}
    if not required_cols.issubset(train.columns):
        return
    for df in [train, test]:
        df["Section_Height"] = df["Width"] * (df["Aspect"] / 100.0)
        df["Rim_Diameter_mm"] = df["Inch"] * 25.4
        df["Outer_Diameter_mm"] = 2.0 * df["Section_Height"] + df["Rim_Diameter_mm"]
        df["Section_Area"] = np.pi * (df["Width"] / 2.0) * df["Section_Height"]
        df["Volume_Proxy"] = df["Section_Area"] * (np.pi * (df["Rim_Diameter_mm"] / 2.0))
        df["Height_to_Width"] = df["Section_Height"] / (df["Width"] + 1e-6)
        df["Height_to_Rim"] = df["Section_Height"] / (df["Rim_Diameter_mm"] + 1e-6)
        df["Outer_to_Rim"] = df["Outer_Diameter_mm"] / (df["Rim_Diameter_mm"] + 1e-6)
    for col in ["Section_Height", "Rim_Diameter_mm", "Outer_Diameter_mm"]:
        add_bin_ng_rate_feature(train, test, column=col, series=train[col], target=target, q=5)


def add_plant_zscores(train: pd.DataFrame, test: pd.DataFrame, columns: List[str]) -> None:
    """공장별 평균 대비 얼마나 벗어났는지 z-score로 표현."""
    cols = [col for col in columns if col in train.columns]
    if not cols or "Plant" not in train.columns:
        return
    plant_stats = train.groupby("Plant")[cols].agg(["mean", "std"]).swaplevel(axis=1)
    global_stats = {}
    for col in cols:
        global_stats[("mean", col)] = train[col].mean()
        global_stats[("std", col)] = train[col].std()

    def _apply(df: pd.DataFrame) -> None:
        aligned = plant_stats.reindex(df["Plant"]).copy()
        for key, val in global_stats.items():
            aligned[key].fillna(val, inplace=True)
        for col in cols:
            df[f"{col}_plant_z"] = (df[col] - aligned[("mean", col)].values) / (
                aligned[("std", col)].values + 1e-6
            )

    _apply(train)
    _apply(test)


def add_g_feature_stats(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series) -> None:
    """G1~G4 변수에 z-score 및 분위수 NG rate 피처 추가."""
    g_cols = [col for col in ["G1", "G2", "G3", "G4"] if col in train.columns]
    if not g_cols:
        return
    means = train[g_cols].mean()
    stds = train[g_cols].std().replace(0, 1)
    for df in [train, test]:
        for col in g_cols:
            df[f"{col}_zscore"] = (df[col] - means[col]) / stds[col]
    for col in g_cols:
        add_bin_ng_rate_feature(train, test, column=col, series=train[col], target=target, q=5)


def add_bin_ng_rate_feature(
    train: pd.DataFrame,
    test: pd.DataFrame,
    column: str,
    series: pd.Series,
    target: pd.Series,
    q: int,
) -> None:
    bins = pd.qcut(series, q=q, duplicates="drop")
    intervals = bins.cat.categories
    edges = [intervals[0].left]
    edges.extend(interval.right for interval in intervals)
    edges[0] = min(edges[0], series.min())
    edges[-1] = max(edges[-1], series.max())
    train_bins = pd.cut(series, bins=edges, include_lowest=True)
    test_bins = pd.cut(test[column], bins=edges, include_lowest=True)
    rates = (
        train.assign(target=target, _bin=train_bins)
        .groupby("_bin", observed=True)["target"]
        .mean()
    )
    new_col = f"{column}_bin_NG_rate"
    train_values = pd.to_numeric(train_bins.map(rates), errors="coerce")
    test_values = pd.to_numeric(test_bins.map(rates), errors="coerce")
    train[new_col] = train_values.fillna(rates.mean())
    test[new_col] = test_values.fillna(rates.mean())


def add_fem_statistics(train: pd.DataFrame, test: pd.DataFrame, p_cols: List[str]) -> None:
    for df in [train, test]:
        values = df[p_cols].to_numpy(dtype=float)
        df["p_min"] = values.min(axis=1)
        df["p_max"] = values.max(axis=1)
        df["p_mean"] = values.mean(axis=1)
        df["p_std"] = values.std(axis=1)
        df["p_skew"] = skew(values, axis=1, bias=False, nan_policy="omit")
        df["bottom5_mean"] = np.partition(values, kth=4, axis=1)[:, :5].mean(axis=1)
        df["top5_mean"] = np.partition(values, kth=values.shape[1] - 5, axis=1)[:, -5:].mean(axis=1)


def add_fem_quantiles(train: pd.DataFrame, test: pd.DataFrame, p_cols: List[str]) -> None:
    """FEM 필드에서 추가적인 분위수 통계를 생성"""
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    for df in [train, test]:
        values = df[p_cols].to_numpy(dtype=float)
        for q in quantiles:
            df[f"p_quantile_{int(q*100):02d}"] = np.quantile(values, q, axis=1)


def add_fem_cluster_features(train: pd.DataFrame, test: pd.DataFrame, p_cols: List[str], n_clusters: List[int]) -> None:
    """FEM field vector를 KMeans 클러스터로 요약하고 label/one-hot 부여"""
    if not p_cols:
        return
    scaler = StandardScaler()
    combined = pd.concat([train[p_cols], test[p_cols]], ignore_index=True)
    scaled = scaler.fit_transform(combined)
    start = len(train)
    for k in n_clusters:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(scaled)
        train_labels = labels[:start]
        test_labels = labels[start:]
        train[f"FEM_cluster_{k}"] = train_labels
        test[f"FEM_cluster_{k}"] = test_labels
        # one-hot encoding for stability
        for cluster_id in range(k):
            train[f"FEM_cluster_{k}_{cluster_id}"] = (train_labels == cluster_id).astype(int)
            test[f"FEM_cluster_{k}_{cluster_id}"] = (test_labels == cluster_id).astype(int)


def add_tire_size_clusters(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: List[str],
    target: pd.Series,
    n_clusters_list: List[int],
    min_cluster_size: int = 40,
) -> pd.DataFrame | None:
    """폭/단면높이/림 직경 기반 KMeans 군집을 여러 k로 시도하고 최적 k를 선택."""
    cols = [col for col in columns if col in train.columns]
    if len(cols) < 3 or not n_clusters_list:
        return None
    combined = pd.concat([train[cols], test[cols]], ignore_index=True)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined.fillna(combined.mean()))
    cluster_records = []
    best_choice = None
    best_metric = -np.inf

    for k in n_clusters_list:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=30)
        labels = km.fit_predict(scaled)
        train_labels = labels[: len(train)]
        test_labels = labels[len(train) :]
        stats = (
            pd.DataFrame({"cluster": train_labels, "target": target})
            .groupby("cluster", observed=True)["target"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "ng_rate", "count": "samples"})
        )
        stats["k"] = k
        stats["cluster"] = stats.index
        cluster_records.append(stats.reset_index(drop=True))
        min_samples = stats["samples"].min()
        spread = stats["ng_rate"].max() - stats["ng_rate"].min()
        metric = spread if min_samples >= min_cluster_size else spread - 0.05
        if metric > best_metric:
            best_metric = metric
            best_choice = {"k": k, "train_labels": train_labels, "test_labels": test_labels, "stats": stats}

    if best_choice is None:
        return None

    train["TireSize_cluster"] = best_choice["train_labels"]
    test["TireSize_cluster"] = best_choice["test_labels"]
    ng_map = best_choice["stats"]["ng_rate"].to_dict()
    train["TireSize_cluster_NG_rate"] = train["TireSize_cluster"].map(ng_map)
    test["TireSize_cluster_NG_rate"] = test["TireSize_cluster"].map(ng_map)

    stats_df = pd.concat(cluster_records, ignore_index=True)
    stats_df["selected"] = stats_df["k"] == best_choice["k"]
    stats_df.to_csv(OUTPUT_DIR / "tire_size_cluster_stats.csv", index=False)
    print(f"[build_features] Tire size cluster candidate stats saved (selected k={best_choice['k']}).")
    return stats_df


def add_fem_pca(train: pd.DataFrame, test: pd.DataFrame, p_cols: List[str]) -> List[str]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[p_cols])
    test_scaled = scaler.transform(test[p_cols])
    pca_full = PCA(random_state=SEED).fit(train_scaled)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cum, 0.95) + 1)
    pca = PCA(n_components=n_components, random_state=SEED)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    feature_names = []
    for idx in range(n_components):
        name = f"FEM_PCA_{idx+1}"
        train[name] = train_pca[:, idx]
        test[name] = test_pca[:, idx]
        feature_names.append(name)
    return feature_names


def add_fem_risk_score(train: pd.DataFrame, test: pd.DataFrame, p_cols: List[str], target: pd.Series) -> None:
    ng_mean = train.loc[target == 1, p_cols].mean()
    good_mean = train.loc[target == 0, p_cols].mean()
    delta = ng_mean - good_mean
    train["FEM_risk_score"] = np.dot(train[p_cols].to_numpy(dtype=float), delta.to_numpy())
    test["FEM_risk_score"] = np.dot(test[p_cols].to_numpy(dtype=float), delta.to_numpy())


def add_proc_features(train: pd.DataFrame, test: pd.DataFrame, proc_cols: List[str], target: pd.Series) -> None:
    if not proc_cols:
        return
    plant_means = train.groupby("Plant")[proc_cols].mean()
    global_means = train[proc_cols].mean()

    def _apply_interactions(df: pd.DataFrame) -> None:
        aligned = plant_means.reindex(df["Plant"]).fillna(global_means)
        for col in proc_cols:
            denom = aligned[col].values
            df[f"{col}_plant_ratio"] = df[col] / (denom + 1e-6)
            df[f"{col}_plant_diff"] = df[col] - denom

    _apply_interactions(train)
    _apply_interactions(test)
    add_proc_bin_features(train, test, proc_cols, target, q=6)


def add_proc_bin_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    proc_cols: List[str],
    target: pd.Series,
    q: int,
) -> None:
    if not proc_cols:
        return
    for col in proc_cols:
        series = train[col]
        try:
            bins = pd.qcut(series, q=q, duplicates="drop")
        except ValueError:
            continue
        intervals = bins.cat.categories
        if len(intervals) == 0:
            continue
        edges = [intervals[0].left]
        edges.extend(interval.right for interval in intervals)
        edges[0] = min(edges[0], series.min())
        edges[-1] = max(edges[-1], series.max())
        train_bins = pd.cut(series, bins=edges, include_lowest=True)
        test_bins = pd.cut(test[col], bins=edges, include_lowest=True)
        rates = (
            train.assign(target=target, _bin=train_bins)
            .groupby("_bin", observed=True)["target"]
            .mean()
        )
        new_col = f"{col}_bin_NG_rate"
        train_values = pd.to_numeric(train_bins.map(rates), errors="coerce")
        test_values = pd.to_numeric(test_bins.map(rates), errors="coerce")
        train[new_col] = train_values.fillna(rates.mean())
        test[new_col] = test_values.fillna(rates.mean())


def add_plant_features(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series, p_cols: List[str]) -> None:
    plant_ng_rate = train.assign(target=target).groupby("Plant")["target"].mean()
    global_rate = target.mean()
    train["Plant_NG_rate"] = train["Plant"].map(plant_ng_rate).fillna(global_rate)
    test["Plant_NG_rate"] = test["Plant"].map(plant_ng_rate).fillna(global_rate)

    plant_p_mean = train.groupby("Plant")[p_cols].mean()
    global_p_mean = train[p_cols].mean()

    def _apply(df: pd.DataFrame) -> None:
        aligned = plant_p_mean.reindex(df["Plant"]).copy()
        aligned = aligned.fillna(global_p_mean)
        plant_vectors = aligned.to_numpy(dtype=float)
        samples = df[p_cols].to_numpy(dtype=float)
        numerator = np.sum(samples * plant_vectors, axis=1)
        denom = (
            np.linalg.norm(samples, axis=1) * np.linalg.norm(plant_vectors, axis=1) + 1e-8
        )
        df["Plant_FEM_cosine"] = numerator / denom
        df["Plant_FEM_l2"] = np.linalg.norm(samples - plant_vectors, axis=1)

    _apply(train)
    _apply(test)


def add_pairwise_nonlinear_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_cols: List[str],
    max_pairs: int = 50,
) -> None:
    """상위 주요 feature 조합에서 비선형 변환 (곱/차/비/루트) 생성"""
    pairs = list(itertools.combinations(base_cols, 2))[:max_pairs]
    for left, right in pairs:
        if left not in train.columns or right not in train.columns:
            continue
        for df in [train, test]:
            prod_name = f"{left}__mul__{right}"
            diff_name = f"{left}__diff__{right}"
            ratio_name = f"{left}__ratio__{right}"
            sqrt_name = f"{left}__sqrtmul__{right}"
            df[prod_name] = df[left] * df[right]
            df[diff_name] = df[left] - df[right]
            df[ratio_name] = df[left] / (df[right].replace(0, np.nan) + 1e-6)
            df[sqrt_name] = np.sqrt(np.abs(df[left] * df[right]))


def add_pca_alignment_features(train: pd.DataFrame, test: pd.DataFrame) -> None:
    cols = ["Width", "Aspect", "Inch", "G1", "G2", "G3", "G4", "p_mean", "p_std", "p_max", "p_min"]
    combined = pd.concat([train[cols], test[cols]], ignore_index=True)
    scaler = StandardScaler()
    aligned = scaler.fit_transform(combined.fillna(combined.mean()))
    pca = PCA(n_components=3, random_state=SEED)
    pcs = pca.fit_transform(aligned)
    train_pcs = pcs[: len(train)]
    test_pcs = pcs[len(train) :]
    global_center = pcs.mean(axis=0)
    train_center = train_pcs.mean(axis=0)
    test_center = test_pcs.mean(axis=0)

    def _dist(arr: np.ndarray, center: np.ndarray) -> np.ndarray:
        return np.linalg.norm(arr - center, axis=1)

    train["dist_to_global_pca_center"] = _dist(train_pcs, global_center)
    train["dist_to_train_center"] = _dist(train_pcs, train_center)
    train["dist_to_test_center"] = _dist(train_pcs, test_center)
    test["dist_to_global_pca_center"] = _dist(test_pcs, global_center)
    test["dist_to_train_center"] = _dist(test_pcs, train_center)
    test["dist_to_test_center"] = _dist(test_pcs, test_center)


def drop_correlated_proc(train: pd.DataFrame, test: pd.DataFrame, proc_cols: List[str], threshold: float = 0.95) -> None:
    if not proc_cols:
        return
    numeric_cols = train[proc_cols].select_dtypes(include=np.number).columns
    corr = train[numeric_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    if drop_cols:
        train.drop(columns=drop_cols, inplace=True)
        test.drop(columns=drop_cols, inplace=True)


def scale_features(train: pd.DataFrame, test: pd.DataFrame, columns: List[str]) -> None:
    cols = [col for col in columns if col in train.columns]
    if not cols:
        return
    scaler = StandardScaler()
    scaler.fit(train[cols])
    train[cols] = scaler.transform(train[cols])
    test[cols] = scaler.transform(test[cols])


def shap_prune_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: pd.Series,
    keep_ratio: float,
    min_keep: int = 120,
) -> List[str]:
    """LightGBM + SHAP 으로 중요도가 낮은 피처를 제거"""
    if lgb is None or shap is None:
        print("[build_features] LightGBM/SHAP 미설치 → pruning 생략")
        return [c for c in train.columns if c != "Class"]
    feature_cols = [c for c in train.columns if c != "Class"]
    train_local = train[feature_cols].copy()
    for col in train_local.columns:
        if train_local[col].dtype == "object":
            train_local[col] = train_local[col].astype("category")
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=96,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(train_local, target)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_local)
    shap_arr = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs = np.abs(shap_arr).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    keep_n = max(min_keep, int(len(feature_cols) * keep_ratio))
    keep_features = shap_df.head(keep_n)["feature"].tolist()
    drop_cols = [col for col in feature_cols if col not in keep_features]
    if drop_cols:
        train.drop(columns=drop_cols, inplace=True)
        test.drop(columns=drop_cols, inplace=True)
    shap_df.to_csv(OUTPUT_DIR / "shap_feature_importance.csv", index=False)
    return keep_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature engineering builder with SHAP pruning")
    parser.add_argument("--enable_shap_pruning", action="store_true", help="SHAP 기반 feature pruning 실행")
    parser.add_argument("--shap_keep_ratio", type=float, default=0.65, help="유지할 feature 비율")
    parser.add_argument(
        "--interaction_cols",
        type=str,
        default="Width,Aspect,Inch,p_mean,p_std,Plant_FEM_cosine",
        help="비선형 조합을 생성할 기준 컬럼 comma list",
    )
    parser.add_argument("--interaction_pairs", type=int, default=40, help="생성할 pair 최대 갯수")
    parser.add_argument(
        "--cluster_sizes",
        type=str,
        default="4,6",
        help="FEM KMeans cluster 개수 리스트",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    train, test = load_data()
    target = (train["Class"] == "NG").astype(int)

    train_feats = train.copy()
    test_feats = test.copy()
    basic_preprocessing(train_feats, test_feats)

    p_cols = [c for c in train_feats.columns if c.startswith("p")]
    proc_cols = [
        c
        for c in train_feats.columns
        if c.startswith("Proc_Param") and pd.api.types.is_numeric_dtype(train_feats[c])
    ]

    add_design_features(train_feats, test_feats, target)
    add_tire_geometry_features(train_feats, test_feats, target)
    add_fem_statistics(train_feats, test_feats, p_cols)
    add_fem_quantiles(train_feats, test_feats, p_cols)
    add_fem_cluster_features(
        train_feats,
        test_feats,
        p_cols,
        n_clusters=[int(x) for x in args.cluster_sizes.split(",") if x.strip()],
    )
    pca_cols = add_fem_pca(train_feats, test_feats, p_cols)
    add_fem_risk_score(train_feats, test_feats, p_cols, target)
    add_proc_features(train_feats, test_feats, proc_cols, target)
    add_plant_features(train_feats, test_feats, target, p_cols)
    add_g_feature_stats(train_feats, test_feats, target)
    add_plant_zscores(
        train_feats,
        test_feats,
        columns=[
            "Section_Height",
            "Rim_Diameter_mm",
            "Outer_Diameter_mm",
        ],
    )
    add_pca_alignment_features(train_feats, test_feats)
    drop_correlated_proc(train_feats, test_feats, proc_cols)

    interaction_cols = [col for col in args.interaction_cols.split(",") if col.strip()]
    add_pairwise_nonlinear_features(
        train_feats,
        test_feats,
        base_cols=interaction_cols,
        max_pairs=args.interaction_pairs,
    )
    size_cols = ["Width", "Section_Height", "Rim_Diameter_mm", "Outer_Diameter_mm"]
    cluster_stats = add_tire_size_clusters(
        train_feats,
        test_feats,
        size_cols,
        target,
        n_clusters_list=[4, 5, 6, 7, 8],
    )
    if cluster_stats is not None:
        print("[build_features] Tire size cluster NG rate comparison:")
        print(cluster_stats.groupby("k")[["ng_rate"]].agg(["min", "max"]))
    drop_simulation_source_columns(train_feats, test_feats)

    for df in [train_feats, test_feats]:
        id_cols = [c for c in df.columns if c.lower().startswith("id")]
        if id_cols:
            df.drop(columns=id_cols, inplace=True)

    scale_cols = [
        "Width_div_Aspect",
        "Width_div_Inch",
        "Aspect_div_Inch",
        "Width_mul_Aspect",
        "Width_mul_Inch",
        "Aspect_mul_Inch",
        "Width_sqrt",
        "Aspect_sqrt",
        "Inch_sqrt",
        "Width_log1p",
        "Aspect_log1p",
        "Inch_log1p",
        "Section_Height",
        "Rim_Diameter_mm",
        "Outer_Diameter_mm",
        "Section_Area",
        "Volume_Proxy",
        "p_min",
        "p_max",
        "p_mean",
        "p_std",
        "p_skew",
        "bottom5_mean",
        "top5_mean",
        "FEM_risk_score",
        "dist_to_global_pca_center",
        "dist_to_train_center",
        "dist_to_test_center",
    ] + pca_cols
    scale_features(train_feats, test_feats, scale_cols)

    feature_cols = [c for c in train_feats.columns if c not in ["Class"]]
    if args.enable_shap_pruning:
        keep_features = shap_prune_features(
            train_feats,
            test_feats,
            target,
            keep_ratio=args.shap_keep_ratio,
            min_keep=max(150, int(len(feature_cols) * 0.4)),
        )
        feature_cols = keep_features
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_output = train_feats[feature_cols + ["Class"]]
    test_output = test_feats[feature_cols]
    train_output.to_csv(OUTPUT_DIR / "train_features.csv", index=False)
    test_output.to_csv(OUTPUT_DIR / "test_features.csv", index=False)
    with (OUTPUT_DIR / "feature_list.json").open("w", encoding="utf-8") as fp:
        json.dump(feature_cols, fp, indent=2)
    print(f"Generated {len(feature_cols)} features.")


if __name__ == "__main__":
    main()
