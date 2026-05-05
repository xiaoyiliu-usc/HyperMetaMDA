"""
消融实验柱状图：风格同论文 classifier 对比图。
一张大图，每行一个数据集，每行 5 个子图 AUROC/AUPR/ACC/F1-score/Average。
每个子图横轴 = 消融方法，每根柱子一个消融，柱顶标数值，Y 轴 0.6~1.0。
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

EXP_LABELS = {
    "main_hgat": "HyperMetaMDA",
    "w_o_hypergraph": "w/o Hypergraph",
    "avg_fusion": "Avg Fusion",
    "w_o_path_hyperedge": "w/o Path Hyperedge",
    "knn_only_hyperedge": "KNN-only",
    "full_hyperedges": "Full Hyperedges",
    "random_hyperedges": "Random Hyperedges",
}
# 图中柱子从左到右：KNN-only 最左，MetaMDA (full) 最右，二者名字与颜色互换
ABLATION_DISPLAY_ORDER = [
    "knn_only_hyperedge", "w_o_hypergraph", "avg_fusion", "w_o_path_hyperedge", "main_hgat",
]

METRICS = ["auroc", "aupr", "acc", "f1"]
METRIC_LABELS = {"auroc": "AUROC", "aupr": "AUPR", "acc": "ACC", "f1": "F1-score"}
# 每个消融一种颜色（类似参考图里每个 classifier 一种颜色）
BAR_COLORS = [
    "#95a5a6", "#3498db", "#f1c40f", "#2ecc71",
    "#e74c3c", "#9b59b6", "#1abc9c", "#e67e22",
]
# 文件名里的 dataset 名 -> 图上的显示名
DATASET_LABELS = {"mdadh": "MDAD", "aBiofilm": "aBiofilm", "masi": "MASI"}


def load_ablation_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    if "exp_name" in df.columns:
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
            df = df.sort_values(["exp_name", "timestamp_utc"], kind="stable")
        df = df.drop_duplicates(subset=["exp_name"], keep="last")
    return df


def plot_ablation_like_reference(df_list, name_list, out_path: str):
    """
    一张图：N 行 × 5 列。每行 = 一个数据集的 AUROC/AUPR/ACC/F1-score/Average。
    每个子图：横轴 = 消融方法，一根柱子一个消融，柱顶标数值，Y 轴 0.6~1.0。
    """
    n_rows = len(df_list)
    assert n_rows == len(name_list) and n_rows >= 1
    n_cols = 5  # AUROC, AUPR, ACC, F1-score, Average
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, (df, dataset_name) in enumerate(zip(df_list, name_list)):
        # 按 ABLATION_DISPLAY_ORDER 排序，使 MetaMDA(full) 在最右、KNN-only 在最左
        order = [e for e in ABLATION_DISPLAY_ORDER if e in df["exp_name"].values]
        if order:
            df = df.set_index("exp_name").loc[order].reset_index()
        exp_names = df["exp_name"].tolist()
        labels = [EXP_LABELS.get(e, e) for e in exp_names]
        n = len(exp_names)
        x = np.arange(n)
        width = 0.65

        for col in range(n_cols):
            ax = axes[row, col]
            if col < 4:
                metric = METRICS[col]
                values = df[metric].values
                title = METRIC_LABELS[metric]
            else:
                values = df[METRICS].astype(float).mean(axis=1).values
                title = "Average"
            colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(n)]
            bars = ax.bar(x, values, width, color=colors, edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )
            if col == 0:
                ax.set_ylabel(f"{dataset_name}\nScore", fontsize=10, labelpad=8)
            else:
                ax.set_ylabel("Score", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
            # 纵轴按数据范围缩放，突出差异
            v_min, v_max = values.min(), values.max()
            margin = max(0.02, (v_max - v_min) * 0.15) if v_max > v_min else 0.02
            y_lo = max(0.0, v_min - margin)
            y_hi = min(1.02, v_max + margin)
            ax.set_ylim(y_lo, y_hi)
            ax.set_title(title, fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)

    plt.suptitle("Ablation Study (5-fold CV, seed=42)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        nargs="+",
        default=[
            "runs/ablation_MDAD_cv5_seed42.csv",
            "runs/ablation_aBiofilm_cv5_seed42.csv",
            "runs/ablation_masi_cv5_seed42.csv",
        ],
        help="CSV 列表，顺序即图中从上到下。MDAD 默认用 ablation_MDAD_cv5_seed42.csv（与 aBiofilm 同五类消融）",
    )
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--out", type=str, default=None, help="输出文件名，默认 ablation_bars.png")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_list = []
    name_list = []
    for p in args.csv:
        path = Path(p) if Path(p).is_absolute() else root / p
        df = load_ablation_csv(str(path))
        raw_name = path.stem.replace("ablation_", "").replace("_cv5_seed42", "")
        display_name = DATASET_LABELS.get(raw_name, raw_name)
        df_list.append(df)
        name_list.append(display_name)

    out_name = args.out or "ablation_bars.png"
    out_path = out_dir / out_name
    plot_ablation_like_reference(df_list, name_list, str(out_path))


if __name__ == "__main__":
    main()
