#!/usr/bin/env python
"""
Threshold and Top-N grid search for Task2 profit maximization.
Outputs:
- CSV of threshold/top-N simulations
- Optimal decision file for test set
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "5-ai-and-datascience-competition"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
# 대회 규칙: Task2에서 선택할 수 있는 샘플은 최대 200개
MAX_SELECTION = 200
# GOOD/NG 선택에 따른 수익/손실 (문제 정의 그대로 사용)
PROFIT_GOOD = 100
LOSS_NG = -2000


def load_predictions(oof_path: Path, test_path: Path, column: str) -> tuple[np.ndarray, np.ndarray]:
    """저장된 확률 CSV에서 특정 column을 읽어온다."""
    oof_df = pd.read_csv(oof_path)
    test_df = pd.read_csv(test_path)
    if column not in oof_df.columns:
        raise ValueError(f"Column {column} not found in {oof_path}")
    if column not in test_df.columns:
        raise ValueError(f"Column {column} not found in {test_path}")
    return oof_df[column].to_numpy(), test_df[column].to_numpy()


def evaluate_thresholds(prob_good: np.ndarray, target: np.ndarray, thresholds: np.ndarray, risk_penalty: float) -> pd.DataFrame:
    """여러 p_good 임계치를 적용했을 때 선택 수/순이익을 기록"""
    records = []
    for thr in thresholds:
        decision = prob_good >= thr  # p_good이 threshold 이상인 샘플만 선택
        if decision.sum() > MAX_SELECTION:
            # 선택 개수가 200개를 넘으면 상위 확률 순으로 200개만 유지
            top_idx = np.argsort(prob_good)[::-1][:MAX_SELECTION]
            decision = np.zeros_like(decision, dtype=bool)
            decision[top_idx] = True
        profit, ng_sel = compute_profit(decision, target)
        records.append(
            {
                "mode": "threshold",
                "value": thr,
                "selected": int(decision.sum()),
                "profit": profit,
                "ng_selected": int(ng_sel),
                "profit_adjusted": profit - risk_penalty * ng_sel,
            }
        )
    return pd.DataFrame(records)


def evaluate_topn(prob_good: np.ndarray, target: np.ndarray, n_values: range, risk_penalty: float) -> pd.DataFrame:
    """상위 N개만 선택하는 전략도 비교"""
    records = []
    order = np.argsort(prob_good)[::-1]
    for n in n_values:
        idx = order[: min(n, MAX_SELECTION)]
        decision = np.zeros_like(prob_good, dtype=bool)
        decision[idx] = True
        profit, ng_sel = compute_profit(decision, target)
        records.append(
            {
                "mode": "topn",
                "value": n,
                "selected": min(n, MAX_SELECTION),
                "profit": profit,
                "ng_selected": int(ng_sel),
                "profit_adjusted": profit - risk_penalty * ng_sel,
            }
        )
    return pd.DataFrame(records)


def compute_profit(decision: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    """선택된 샘플에서 GOOD/NG 수를 세어 순이익 계산"""
    good_sel = int((decision & (target == 0)).sum())
    ng_sel = int((decision & (target == 1)).sum())
    profit = good_sel * PROFIT_GOOD + ng_sel * LOSS_NG
    return profit, ng_sel


def evaluate_expected_profit(prob_good: np.ndarray, target: np.ndarray, margins: np.ndarray, risk_penalty: float) -> pd.DataFrame:
    """기대수익 기반 마진 컷 전략"""
    prob_ng = 1.0 - prob_good
    expected = PROFIT_GOOD * prob_good + LOSS_NG * prob_ng
    records = []
    for margin in margins:
        decision = expected >= margin
        if decision.sum() > MAX_SELECTION:
            top_idx = np.argsort(expected)[::-1][:MAX_SELECTION]
            mask = np.zeros_like(decision, dtype=bool)
            mask[top_idx] = True
            decision = mask
        profit, ng_sel = compute_profit(decision, target)
        records.append(
            {
                "mode": "expected",
                "value": margin,
                "selected": int(decision.sum()),
                "profit": profit,
                "ng_selected": int(ng_sel),
                "profit_adjusted": profit - risk_penalty * ng_sel,
            }
        )
    return pd.DataFrame(records)


def apply_best_strategy(prob_good: np.ndarray, strategy: dict, ids: pd.Series) -> pd.DataFrame:
    """최적 전략(Threshold/Top-N)에 따라 최종 의사결정 테이블 생성"""
    decision = np.zeros(len(prob_good), dtype=bool)
    if strategy["mode"] == "threshold":
        decision = prob_good >= strategy["value"]
        if decision.sum() > MAX_SELECTION:
            top_idx = np.argsort(prob_good)[::-1][:MAX_SELECTION]
            decision = np.zeros_like(decision, dtype=bool)
            decision[top_idx] = True
    elif strategy["mode"] == "topn":
        order = np.argsort(prob_good)[::-1]
        idx = order[: min(int(strategy["value"]), MAX_SELECTION)]
        decision[idx] = True
    else:
        prob_ng = 1.0 - prob_good
        expected = PROFIT_GOOD * prob_good + LOSS_NG * prob_ng
        decision = expected >= strategy["value"]
        if decision.sum() > MAX_SELECTION:
            top_idx = np.argsort(expected)[::-1][:MAX_SELECTION]
            mask = np.zeros_like(decision, dtype=bool)
            mask[top_idx] = True
            decision = mask
    return pd.DataFrame(
        {
            "ID": ids,
            "prob_ng": 1.0 - prob_good,
            "prob_good": prob_good,
            "decision": decision,
        }
    )


def build_submission(decisions: pd.DataFrame, pred_path: Path, output_path: Path) -> None:
    """sample_submission 포맷을 복사해 확률과 결정 값을 덮어쓴다"""
    submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    base_prob = pd.read_csv(pred_path)["probability"].to_numpy()
    submission["probability"] = np.repeat(base_prob, 2)[: len(submission)]
    base_id = submission["ID"].str.replace("_L", "", regex=False).str.replace("_P", "", regex=False)
    decision_map = decisions.set_index("ID")["decision"]
    submission["decision"] = base_id.map(decision_map).fillna(False).astype(bool)
    submission.to_csv(output_path, index=False)


def build_threshold_candidates(min_thr: float, max_thr: float, step: float, extra: list[float]) -> np.ndarray:
    """조밀한 기본 grid + 추가 후보(threshold)들을 합쳐 최종 candidate 생성"""
    grid = np.arange(min_thr, max_thr + 1e-9, step)
    extra_arr = np.array(extra, dtype=float) if extra else np.array([], dtype=float)
    candidates = np.unique(np.concatenate([grid, extra_arr]))
    candidates = candidates[(candidates >= 0.0) & (candidates <= 1.0)]
    return np.sort(candidates)


def plot_profit_curve(threshold_df: pd.DataFrame, output_path: Path) -> None:
    """threshold 별 순이익 곡선을 저장해 과도한/부족한 선택 영역을 시각화"""
    if threshold_df.empty:
        return
    threshold_df = threshold_df.sort_values("value")
    plt.figure(figsize=(8, 4))
    plt.plot(threshold_df["value"], threshold_df["profit"], marker="o", label="Profit")
    if "profit_adjusted" in threshold_df:
        plt.plot(threshold_df["value"], threshold_df["profit_adjusted"], marker="x", label="Adj Profit")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Threshold (p_good)")
    plt.ylabel("Net Profit")
    plt.title("Task2 Net Profit vs Threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=str, default=str(OUTPUT_DIR / "task1_ensemble_oof.csv"))
    parser.add_argument("--test", type=str, default=str(OUTPUT_DIR / "task1_ensemble_test_pred.csv"))
    parser.add_argument("--pred", type=str, default=str(OUTPUT_DIR / "task1_ensemble_test_pred.csv"))
    parser.add_argument("--prob_column", type=str, default="probability", help="확률 column 이름")
    parser.add_argument(
        "--thr_range",
        type=str,
        default="0.80,0.95,0.0005",
        help="min,max,step 형식으로 threshold(p_good) 탐색 범위를 지정",
    )
    parser.add_argument(
        "--extra_thr",
        type=str,
        default="0.90,0.92,0.95",
        help="콤마로 구분된 추가 threshold 후보 (예: 기대값 기반 등)",
    )
    parser.add_argument(
        "--expected_margins",
        type=str,
        default="0,200,500,800",
        help="기대수익 마진 컷 후보 리스트",
    )
    parser.add_argument(
        "--risk_penalty",
        type=float,
        default=200.0,
        help="선택된 NG 1건당 패널티",
    )
    parser.add_argument(
        "--score_metric",
        type=str,
        default="profit",
        choices=["profit", "profit_adjusted"],
        help="최적 전략 선택 시 사용할 지표",
    )
    return parser.parse_args()

def run(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()

    oof_prob_ng, test_prob_ng = load_predictions(Path(args.oof), Path(args.test), args.prob_column)
    prob_good_oof = 1.0 - oof_prob_ng
    prob_good_test = 1.0 - test_prob_ng
    target = (pd.read_csv(OUTPUT_DIR / "train_features.csv")["Class"] == "NG").astype(int).to_numpy()

    thr_min, thr_max, thr_step = (float(x) for x in args.thr_range.split(","))
    extra = [float(x) for x in args.extra_thr.split(",") if x.strip()]
    # 기대 순이익 > 0 조건: 100 - 2100 * p_ng > 0 → p_good > 0.95238
    expect_thr = 1.0 - (PROFIT_GOOD / (PROFIT_GOOD - LOSS_NG))
    thresholds = build_threshold_candidates(thr_min, thr_max, thr_step, extra + [expect_thr])
    threshold_df = evaluate_thresholds(prob_good_oof, target, thresholds, args.risk_penalty)
    topn_df = evaluate_topn(prob_good_oof, target, range(20, MAX_SELECTION + 1), args.risk_penalty)
    expected_margins = [float(x) for x in args.expected_margins.split(",") if x.strip()]
    expected_df = evaluate_expected_profit(prob_good_oof, target, np.array(expected_margins), args.risk_penalty)
    results = pd.concat([threshold_df, topn_df, expected_df], ignore_index=True)
    results.to_csv(OUTPUT_DIR / "task2_threshold_grid_results.csv", index=False)
    plot_profit_curve(threshold_df, OUTPUT_DIR / "figures/task2_net_profit_curve.png")

    score_col = args.score_metric
    best_idx = results[score_col].idxmax()
    best_row = results.loc[best_idx].to_dict()
    decisions = apply_best_strategy(prob_good_test, best_row, pd.read_csv(DATA_DIR / "test.csv")["ID"])
    decisions.to_csv(OUTPUT_DIR / "task2_decision_optimal.csv", index=False)
    build_submission(decisions, Path(args.pred), OUTPUT_DIR / "task2_submission_optimal.csv")
    print(f"Best strategy: {best_row}")


if __name__ == "__main__":
    run()
