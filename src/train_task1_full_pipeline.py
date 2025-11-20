#!/usr/bin/env python
"""전체 Task1 + Task2 + Submission 자동화 파이프라인"""

from __future__ import annotations

import argparse
from pathlib import Path

from build_features import main as build_features_main
from make_final_submission import main as final_submission_main
from task2_threshold_grid_search import run as task2_run
from train_task1_ensemble import run as train_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full pipeline automation")
    parser.add_argument("--enable_shap_pruning", action="store_true")
    parser.add_argument("--shap_keep_ratio", type=float, default=0.65)
    parser.add_argument("--interaction_cols", type=str, default="Width,Aspect,Inch,p_mean,p_std,Plant_FEM_cosine")
    parser.add_argument("--interaction_pairs", type=int, default=40)
    parser.add_argument("--cluster_sizes", type=str, default="4,6")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="111,222,333,444,555")
    parser.add_argument("--n_trials", type=int, default=400)
    parser.add_argument("--skip_tuning", action="store_true")
    parser.add_argument("--stacker", type=str, default="logistic", choices=["logistic", "lightgbm"])
    parser.add_argument("--enable_tabnet", action="store_true")
    parser.add_argument("--calibration_metric", type=str, default="auc", choices=["auc", "brier"])
    parser.add_argument("--task2_risk_penalty", type=float, default=500.0)
    parser.add_argument("--task2_expected_margins", type=str, default="0,500,1000")
    parser.add_argument("--task2_score_metric", type=str, default="profit_adjusted", choices=["profit", "profit_adjusted"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Feature engineering
    build_args = argparse.Namespace(
        enable_shap_pruning=args.enable_shap_pruning,
        shap_keep_ratio=args.shap_keep_ratio,
        interaction_cols=args.interaction_cols,
        interaction_pairs=args.interaction_pairs,
        cluster_sizes=args.cluster_sizes,
    )
    print("[Pipeline] Step1: build_features")
    build_features_main(build_args)

    # 2) Train models + stacking + calibration
    train_args = argparse.Namespace(
        n_splits=args.n_splits,
        seeds=args.seeds,
        n_trials=args.n_trials,
        skip_tuning=args.skip_tuning,
        stacker=args.stacker,
        enable_tabnet=args.enable_tabnet,
        calibration_metric=args.calibration_metric,
    )
    print("[Pipeline] Step2: train_task1_ensemble")
    train_run(train_args)

    # 3) Task2 profit optimizer
    task2_args = argparse.Namespace(
        oof=str(OUTPUT_DIR / "task1_ensemble_oof.csv"),
        test=str(OUTPUT_DIR / "task1_ensemble_test_pred.csv"),
        pred=str(OUTPUT_DIR / "task1_ensemble_test_pred.csv"),
        prob_column="probability",
        thr_range="0.80,0.99,0.01",
        extra_thr="0.90,0.92,0.95",
        expected_margins=args.task2_expected_margins,
        risk_penalty=args.task2_risk_penalty,
        score_metric=args.task2_score_metric,
    )
    print("[Pipeline] Step3: task2_threshold_grid_search")
    task2_run(task2_args)

    # 4) Final submission merge
    print("[Pipeline] Step4: make_final_submission")
    final_args = argparse.Namespace(
        task1=str(OUTPUT_DIR / "task1_ensemble_submission.csv"),
        task2=str(OUTPUT_DIR / "task2_submission_optimal.csv"),
        output=str(OUTPUT_DIR / "final_submission.csv"),
    )
    final_submission_main(final_args)
    print("[Pipeline] All steps finished → outputs/final_submission.csv")


if __name__ == "__main__":
    main()
