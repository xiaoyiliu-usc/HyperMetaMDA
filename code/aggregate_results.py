"""
Aggregate `runs/results.csv` into summary tables (mean±std) for paper-ready reporting.

Example:
  python code/aggregate_results.py --in_csv runs/results.csv --out_csv runs/summary_by_dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, default=str(Path(__file__).resolve().parents[1] / "runs" / "results.csv"))
    parser.add_argument("--out_csv", type=str, default=str(Path(__file__).resolve().parents[1] / "runs" / "summary.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    # numeric columns (robust parsing)
    for col in ["acc", "auroc", "aupr", "f1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # group by exp/dataset/cv, aggregate over seeds (and repeated runs)
    gcols = [c for c in ["exp_name", "dataset", "cv"] if c in df.columns]
    agg = df.groupby(gcols).agg(
        n=("auroc", "count"),
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        aupr_mean=("aupr", "mean"),
        aupr_std=("aupr", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
    ).reset_index()

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(outp, index=False)
    print(f"[Saved] {outp}")


if __name__ == "__main__":
    main()

