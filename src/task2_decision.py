#!/usr/bin/env python
"""
Task2 decision logic based on ensemble predictions.
Steps:
- Load test IDs and NG probabilities from outputs/task1_ensemble_test_pred.csv
- Compute Good probability and expected profit threshold (p_good > 0.95238)
- Cap selections at 200 samples (highest p_good)
- Produce decision table and updated submission file
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "5-ai-and-datascience-competition"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
THRESHOLD_GOOD = 0.95238  # p(Good) > 2000/(2000+100)
MAX_SELECTION = 200


def load_predictions() -> pd.DataFrame:
    pred_path = OUTPUT_DIR / "task1_ensemble_test_pred.csv"
    if not pred_path.exists():
        pred_path = OUTPUT_DIR / "task1_test_pred.csv"
    preds = pd.read_csv(pred_path)
    if "probability" not in preds.columns and preds.shape[1] == 1:
        preds.columns = ["probability"]
    if "probability" not in preds.columns and "prob_ng" in preds.columns:
        preds = preds.rename(columns={"prob_ng": "probability"})
    if "probability" not in preds.columns:
        raise ValueError("prediction file must contain 'probability' column")
    preds = preds[["probability"]].rename(columns={"probability": "prob_ng"})
    return preds


def select_samples(preds: pd.DataFrame, test_ids: pd.Series) -> pd.DataFrame:
    df = preds.copy()
    df["ID"] = test_ids.values
    df["prob_good"] = 1.0 - df["prob_ng"]
    df["decision"] = df["prob_good"] > THRESHOLD_GOOD
    if df["decision"].sum() > MAX_SELECTION:
        top_idx = df.sort_values("prob_good", ascending=False).head(MAX_SELECTION).index
        df["decision"] = df.index.isin(top_idx)
    return df


def build_submission(decisions: pd.DataFrame) -> pd.DataFrame:
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    base_prob = np.repeat(decisions["prob_ng"].values, 2)
    sample["probability"] = base_prob

    base_id = sample["ID"].str.replace("_L", "", regex=False).str.replace("_P", "", regex=False)
    decision_map = decisions.set_index("ID")["decision"]
    sample["decision"] = base_id.map(decision_map).fillna(False)
    sample["decision"] = sample["decision"].astype(bool)
    return sample


def main() -> None:
    preds = load_predictions()
    test_ids = pd.read_csv(DATA_DIR / "test.csv")["ID"]
    decisions = select_samples(preds, test_ids)
    decisions.to_csv(OUTPUT_DIR / "task2_decision_table.csv", index=False)
    submission = build_submission(decisions)
    submission.to_csv(OUTPUT_DIR / "task2_submission.csv", index=False)
    print(f"Selected {decisions['decision'].sum()} samples out of {len(decisions)}")


if __name__ == "__main__":
    main()
