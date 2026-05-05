from __future__ import annotations

"""
Plot classifier ablation bars in paper Figure 3 style:
- (A) MDAD and (B) aBiofilm
- 5 columns: AUROC / AUPR / ACC / F1-score / Average
- each subplot shows 8 classifiers.
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSIFIER_ORDER = ["DNN", "DT", "LR", "SVM", "AdaBoost", "GBDT", "RF", "XGBoost"]
METRICS = ["auroc", "aupr", "acc", "f1", "average"]
METRIC_LABELS = {
    "auroc": "AUROC",
    "aupr": "AUPR",
    "acc": "ACC",
    "f1": "F1-score",
    "average": "Average",
}
BAR_COLORS = [
    "#7f8c8d",
    "#95a5a6",
    "#f39c12",
    "#3498db",
    "#9b59b6",
    "#2ecc71",
    "#1abc9c",
    "#e74c3c",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prep_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _row_by_dataset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    sub = df[df["dataset"] == dataset].copy()
    if len(sub) == 0:
        raise ValueError(f"CSV 中未找到 dataset={dataset}")
    sub = sub.set_index("classifier").reindex(CLASSIFIER_ORDER).reset_index()
    if sub[METRICS].isna().any().any():
        missing = sub[sub[METRICS].isna().any(axis=1)]["classifier"].tolist()
        raise ValueError(f"dataset={dataset} 缺少分类器结果: {missing}")
    return sub


def plot_classifier_ablation(df: pd.DataFrame, datasets: List[str], out_path: Path) -> None:
    n_rows = len(datasets)
    fig, axes = plt.subplots(n_rows, 5, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    x = np.arange(len(CLASSIFIER_ORDER))
    width = 0.65
    for r, dataset in enumerate(datasets):
        sub = _row_by_dataset(df, dataset)
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            values = sub[metric].astype(float).values
            bars = ax.bar(x, values, width=width, color=BAR_COLORS, edgecolor="black", linewidth=0.5)
            # 采用分面自适应纵轴并做标签防重叠/防越界：
            # 1) 相邻柱标签上下交错
            # 2) 超出上界则放入柱内
            # 3) 标签加白底提升可读性
            v_min, v_max = float(values.min()), float(values.max())
            if v_max > v_min:
                margin = max(0.01, (v_max - v_min) * 0.22)
            else:
                margin = 0.02
            y_lo = max(0.0, v_min - margin)
            y_hi = min(1.0, v_max + margin)
            # 避免区间过窄导致视觉拥挤
            if (y_hi - y_lo) < 0.08:
                pad = (0.08 - (y_hi - y_lo)) / 2
                y_lo = max(0.0, y_lo - pad)
                y_hi = min(1.0, y_hi + pad)
            ax.set_ylim(y_lo, y_hi)
            for i, (bar, val) in enumerate(zip(bars, values)):
                y_range = y_hi - y_lo
                y_offset = 0.012 * y_range + (i % 2) * 0.014 * y_range
                y_text = float(bar.get_height()) + y_offset
                va = "bottom"
                if y_text > y_hi - 0.01 * y_range:
                    y_text = float(bar.get_height()) - 0.015 * y_range
                    va = "top"
                y_text = min(max(y_text, y_lo + 0.01 * y_range), y_hi - 0.01 * y_range)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_text,
                    f"{val:.4f}",
                    ha="center",
                    va=va,
                    fontsize=6.5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.1),
                )
            ax.set_xticks(x)
            ax.set_xticklabels(CLASSIFIER_ORDER, rotation=25, ha="right", fontsize=8)
            ax.set_title(METRIC_LABELS[metric], fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            ax.set_ylabel(f"{dataset}\nScore" if c == 0 else "Score", fontsize=9)

    plt.suptitle("Classifier Ablation (10-fold CV)", fontsize=12, y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="runs/classifier_ablation_cv10_seed42.csv")
    parser.add_argument("--datasets", nargs="+", default=["MDAD", "aBiofilm"])
    parser.add_argument("--out", type=str, default="runs/classifier_ablation_figure3_style.png")
    args = parser.parse_args()

    csv_path = Path(args.csv) if Path(args.csv).is_absolute() else _repo_root() / args.csv
    out_path = Path(args.out) if Path(args.out).is_absolute() else _repo_root() / args.out

    df = _prep_df(csv_path)
    plot_classifier_ablation(df, args.datasets, out_path)


if __name__ == "__main__":
    main()

