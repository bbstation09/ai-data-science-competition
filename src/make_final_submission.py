#!/usr/bin/env python
"""
Combine Task1 probability/decision outputs with Task2 decision results
to produce a 932-row final submission (L/P) in final_submission.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "5-ai-and-datascience-competition"


def load_source_files(task1_path: Path, task2_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    task1 = pd.read_csv(task1_path)
    task2 = pd.read_csv(task2_path)
    required_cols = {"ID", "probability", "decision"}
    if not required_cols.issubset(task1.columns):
        raise ValueError(f"Task1 file missing columns: {required_cols - set(task1.columns)}")
    if not required_cols.issubset(task2.columns):
        raise ValueError(f"Task2 file missing columns: {required_cols - set(task2.columns)}")
    test_ids = pd.read_csv(DATA_DIR / "test.csv")["ID"].astype(str)
    task1_ids = task1["ID"].astype(str)
    if not task1_ids.str.startswith("ID_").all():
        task1_ids = "ID_" + task1_ids
    prob_map = pd.Series(task1["probability"].values, index=task1_ids).reindex(test_ids)
    dec_map = (
        task2.assign(base_id=task2["ID"].astype(str).str.replace("_L", "", regex=False).str.replace("_P", "", regex=False))
        .drop_duplicates(subset="base_id")
        .set_index("base_id")["decision"]
        .reindex(test_ids)
    )
    if prob_map.isna().any():
        missing = prob_map.index[prob_map.isna()]
        raise ValueError(f"Task1 probabilities missing for IDs: {missing.tolist()[:5]}...")
    if dec_map.isna().any():
        missing = dec_map.index[dec_map.isna()]
        raise ValueError(f"Task2 decisions missing for IDs: {missing.tolist()[:5]}...")
    merged = pd.DataFrame({"ID": test_ids, "probability": prob_map.values, "decision": dec_map.values})
    return merged


def build_submission(base: pd.DataFrame) -> pd.DataFrame:
    merged = base.copy()
    upper = merged.copy()
    upper["ID"] = upper["ID"] + "_L"
    lower = merged.copy()
    lower["ID"] = lower["ID"] + "_P"
    submission = pd.concat([upper, lower], ignore_index=True)
    return submission[["ID", "probability", "decision"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task1",
        type=str,
        default=str(OUTPUT_DIR / "task1_ensemble_submission.csv"),
        help="Path to Task1 probability/decision CSV",
    )
    parser.add_argument(
        "--task2",
        type=str,
        default=str(OUTPUT_DIR / "task2_submission_optimal.csv"),
        help="Path to Task2 decision CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR / "final_submission.csv"),
        help="Path for final combined submission",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()

    merged = load_source_files(Path(args.task1), Path(args.task2))
    combined = build_submission(merged)
    combined.to_csv(args.output, index=False)
    print(f"Final submission saved to {args.output}, rows: {len(combined)}")


if __name__ == "__main__":
    main()
