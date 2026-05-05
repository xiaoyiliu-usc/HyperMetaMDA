#!/usr/bin/env python3
"""从 param_*.csv 生成 2x2 敏感性折线图（与论文风格类似）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


def _is_power_of_two_series(x: pd.Series) -> bool:
    v = x.astype(float).values
    if len(v) < 2 or (v <= 0).any():
        return False
    return bool(np.allclose(v, 2 ** np.round(np.log2(v)), rtol=1e-9))


def _ylim_with_padding(y: pd.Series, *, y_margin: float, clip_01: bool) -> tuple[float, float]:
    """Tight y-range around data so trends read clearly; small floor gap avoids '触底' illusion."""
    ymin, ymax = float(y.min()), float(y.max())
    span = max(ymax - ymin, 1e-6)
    # Proportional padding keeps amplitude visible; tiny absolute floor only clears the x-axis spine.
    bottom_floor = 0.005 if clip_01 else 0.002
    pad_bottom = span * y_margin + bottom_floor
    pad_top = span * y_margin * 0.5 + bottom_floor
    lo, hi = ymin - pad_bottom, ymax + pad_top
    if clip_01:
        lo = max(0.0, lo)
        hi = min(1.0, hi)
        if ymin - lo < bottom_floor:
            lo = max(0.0, ymin - bottom_floor)
    if lo >= hi:
        hi = min(1.0, ymax + 0.05) if clip_01 else ymax + 0.05
        lo = max(0.0, ymin - 0.05) if clip_01 else ymin - 0.05
        if clip_01 and ymin - lo < bottom_floor:
            lo = max(0.0, ymin - bottom_floor)
    return lo, hi


def plot_one(
    csv_path: Path,
    xlabel: str,
    out_png: Path,
    *,
    x_is_int: bool,
    force_log2_x: bool = False,
    no_auto_log2: bool = False,
    rotate_x: int = 0,
    y_margin: float = 0.08,
) -> None:
    df = pd.read_csv(csv_path)
    g = df.groupby("param_value", as_index=False)[
        ["auroc_mean", "aupr_mean", "acc_mean", "f1_mean"]
    ].mean()
    if x_is_int:
        g["param_value"] = g["param_value"].astype(int)
    else:
        g["param_value"] = g["param_value"].astype(float)
    g = g.sort_values("param_value")

    x = g["param_value"]
    auroc, aupr = g["auroc_mean"], g["aupr_mean"]
    acc, f1 = g["acc_mean"] / 100.0, g["f1_mean"]
    # NOTE: idxmax() returns the *index label*; after sorting, it may not match iloc positions.
    # Use the same row via .loc to avoid mis-marking the best x.
    best_idx = auroc.idxmax()
    best_x = float(g.loc[best_idx, "param_value"])

    auto_log2 = x_is_int and _is_power_of_two_series(x) and len(x) >= 4
    use_log2 = force_log2_x or (not no_auto_log2 and auto_log2)

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2))
    metrics = [("AUROC", auroc), ("AUPR", aupr), ("ACC", acc), ("F1-score", f1)]
    for ax, (title, y) in zip(axes.ravel(), metrics):
        ax.plot(x, y, marker="o", color="#1f77b4")
        ax.axvline(best_x, color="red", linestyle="--", linewidth=1)
        lo, hi = _ylim_with_padding(y, y_margin=y_margin, clip_01=True)
        ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(title)
        if use_log2:
            ax.set_xscale("log", base=2)
            ax.set_xticks(x.tolist())
            fmt = ScalarFormatter()
            fmt.set_scientific(False)
            ax.xaxis.set_major_formatter(fmt)
            ax.minorticks_off()
        elif x_is_int:
            ax.set_xticks(x.tolist())
            if rotate_x:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotate_x, ha="right")
        ax.grid(alpha=0.3)
    fig.suptitle(xlabel, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if rotate_x and not use_log2:
        fig.subplots_adjust(bottom=0.18)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print(f"saved: {out_png}  best_by_AUROC: {best_x}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--xlabel", type=str, required=True)
    p.add_argument("--x-int", action="store_true", help="横轴为整数刻度（dimension / window）")
    p.add_argument("--log2-x", action="store_true", help="强制横轴 log2")
    p.add_argument("--rotate-x", type=int, default=0, help="横轴标签旋转角度（非 log2 时）")
    p.add_argument("--no-auto-log2", action="store_true", help="关闭 64/128/… 自动 log2 横轴")
    p.add_argument(
        "--y-margin",
        type=float,
        default=0.08,
        help="纵轴在数据 min/max 外的留白比例（相对数据跨度），默认 0.08；需更松可加 --y-margin",
    )
    args = p.parse_args()
    plot_one(
        args.csv,
        args.xlabel,
        args.out,
        x_is_int=args.x_int,
        force_log2_x=args.log2_x,
        no_auto_log2=args.no_auto_log2,
        rotate_x=args.rotate_x,
        y_margin=args.y_margin,
    )


if __name__ == "__main__":
    main()
