"""
主实验柱状图（Figure 2 风格）：仅 5-fold，MDAD + aBiofilm 两个数据集。
7 个基线（HMDAKATZ, Graph2MDA, LAGCN, GCNMDA, EGATMDA, SCSMDA, MetaMDA）+ 右侧 Ours。
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 从左到右：7 个基线 + 我们的方法名
METHOD_NAMES = [
    "HMDAKATZ", "Graph2MDA", "LAGCN", "GCNMDA", "EGATMDA", "SCSMDA", "MetaMDA", "HyperMetaMDA",
]
METRIC_NAMES = ["AUROC", "AUPR", "F1-score", "ACC", "Average"]
# 论文/图中 7 个基线的 5-fold 数据（你提供的）
MDAD_5FOLD = {
    "AUROC":   [0.9007, 0.8414, 0.8359, 0.9350, 0.9580, 0.9586, 0.9691],
    "AUPR":    [0.0543, 0.6189, 0.3522, 0.9254, 0.9250, 0.9488, 0.9700],
    "F1-score":[0.1455, 0.5900, 0.05,   0.7455, 0.8745, 0.8552, 0.9000],
    "ACC":     [0.9938, 0.9888, 0.9105, 0.7903, 0.8825, 0.8831, 0.8982],
    "Average": [0.5236, 0.7592, 0.5372, 0.8501, 0.9101, 0.9114, 0.9343],
}
ABIOFILM_5FOLD = {
    "AUROC":   [0.9399, 0.8292, 0.9435, 0.9474, 0.9580, 0.9567, 0.9758],
    "AUPR":    [0.1457, 0.3894, 0.2292, 0.9406, 0.9332, 0.9413, 0.9766],
    "F1-score":[0.2218, 0.6092, 0.1302, 0.7904, 0.8939, 0.8502, 0.9188],
    "ACC":     [0.9949, 0.9907, 0.9611, 0.8193, 0.8956, 0.8909, 0.9169],
    "Average": [0.5758, 0.7046, 0.5732, 0.8745, 0.9197, 0.9098, 0.9473],
}
# 7 个基线颜色 + Ours（GCNMDA 紫、MetaMDA 蓝便于区分）
BAR_COLORS = [
    "#87CEEB", "#FFA500", "#90EE90", "#9370DB", "#FFC0CB", "#1ABC9C", "#3498DB", "#E74C3C",
]


def _annotate_bars_smart(ax, bars, values, y_lo: float, y_hi: float) -> None:
    """
    柱顶数值智能避让：
    - 默认放柱顶上方
    - 若与附近标签过近则逐级抬高
    - 若接近上边界则放入柱内
    """
    y_range = max(1e-6, y_hi - y_lo)
    base_offset = 0.012 * y_range
    bump_step = 0.02 * y_range
    min_gap = 0.018 * y_range
    top_pad = 0.012 * y_range
    bottom_pad = 0.012 * y_range
    placed_y = []

    for i, (bar, val) in enumerate(zip(bars, values)):
        h = float(bar.get_height())
        y_text = h + base_offset
        va = "bottom"

        # 只和相邻几个标签比较即可，避免顶部密集重叠
        neighbors = placed_y[max(0, i - 3):]
        for _ in range(8):
            if all(abs(y_text - p) >= min_gap for p in neighbors):
                break
            y_text += bump_step

        # 若越界则改为柱内显示
        if y_text > y_hi - top_pad:
            y_text = h - 0.015 * y_range
            va = "top"

        # 双向兜底，保证不出图
        y_text = min(max(y_text, y_lo + bottom_pad), y_hi - top_pad)
        placed_y.append(y_text)

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_text,
            f"{val:.4f}",
            ha="center",
            va=va,
            fontsize=6.6,
            rotation=0,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.1),
            zorder=5,
        )


def load_ours_from_csv(mdad_csv: str, abiofilm_csv: str, exp_name: str = "main_hgat"):
    """从 ablation CSV 中读取 main_hgat 的 auroc, aupr, acc, f1，计算 average。"""
    root = Path(__file__).resolve().parents[1]
    out = {"MDAD": None, "aBiofilm": None}
    for name, path in [("MDAD", mdad_csv), ("aBiofilm", abiofilm_csv)]:
        p = Path(path) if Path(path).is_absolute() else root / path
        df = pd.read_csv(p)
        row = df[df["exp_name"] == exp_name]
        if len(row) == 0:
            raise FileNotFoundError(f"No row exp_name={exp_name!r} in {p}")
        row = row.iloc[0]
        auroc = float(row["auroc"])
        aupr = float(row["aupr"])
        acc = float(row["acc"])
        f1 = float(row["f1"])
        avg = (auroc + aupr + acc + f1) / 4.0
        out[name] = [auroc, aupr, f1, acc, avg]
    return out


def plot_figure2_main(ours_mdad, ours_abiofilm, out_path: str):
    """2 行（MDAD, aBiofilm）× 5 列（AUROC, AUPR, F1, ACC, Average），每子图 8 根柱子。"""
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    n_methods = 8
    x = np.arange(n_methods)
    width = 0.65

    # 组装每行数据：7 个基线 + Ours
    for row_idx, (dataset_name, baseline_data, ours_vals) in enumerate([
        ("MDAD", MDAD_5FOLD, ours_mdad),
        ("aBiofilm", ABIOFILM_5FOLD, ours_abiofilm),
    ]):
        for col_idx, m in enumerate(METRIC_NAMES):
            ax = axes[row_idx, col_idx]
            values = list(baseline_data[m]) + [ours_vals[col_idx]]
            values = np.array(values)
            colors = BAR_COLORS
            bars = ax.bar(x, values, width, color=colors, edgecolor="black", linewidth=0.5)
            if col_idx == 0:
                ax.set_ylabel(f"{dataset_name}\nScore", fontsize=10, labelpad=8)
            else:
                ax.set_ylabel("Score", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(METHOD_NAMES, rotation=25, ha="right", fontsize=8)
            v_min, v_max = values.min(), values.max()
            margin = max(0.02, (v_max - v_min) * 0.22) if v_max > v_min else 0.02
            y_lo = max(0.0, v_min - margin)
            y_hi = min(1.02, v_max + margin)
            ax.set_ylim(y_lo, y_hi)
            _annotate_bars_smart(ax, bars, values, y_lo, y_hi)
            ax.set_title(m, fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)

    plt.suptitle("Main Experiment (5-fold CV, seed=42)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdad_csv", type=str, default="runs/ablation_MDAD_cv5_seed42.csv")
    parser.add_argument("--abiofilm_csv", type=str, default="runs/ablation_aBiofilm_cv5_seed42.csv")
    parser.add_argument("--exp_name", type=str, default="main_hgat")
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--out", type=str, default="figure2_main_5fold.png")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ours = load_ours_from_csv(
        args.mdad_csv,
        args.abiofilm_csv,
        exp_name=args.exp_name,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out
    plot_figure2_main(ours["MDAD"], ours["aBiofilm"], str(out_path))


if __name__ == "__main__":
    main()
