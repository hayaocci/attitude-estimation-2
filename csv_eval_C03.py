#!/usr/bin/env python3
"""
C-03: 姿勢分布 vs 誤差解析スクリプト

comp_*.csv を読み込み、
  - true_roll を角度ビンに分け
  - 各ビンの誤差統計 (mean / median / count) を計算し、
  - グラフとCSVを出力します。

出力:
  - error_by_true_roll_bin.csv
  - error_by_true_roll_bin.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="C-03: 姿勢分布 vs 誤差解析")
    p.add_argument(
        "--csv",
        required=True,
        help="code_C で生成した comp_*.csv のパス",
    )
    p.add_argument(
        "--error_col",
        default="kf_err",
        help="使用する誤差列名 (default: kf_err)",
    )
    p.add_argument(
        "--bin_width",
        type=float,
        default=10.0,
        help="true_roll を分割するビン幅[deg] (default: 10.0)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"[INFO] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    for col in ["true_roll", args.error_col]:
        if col not in df.columns:
            raise KeyError(f"列 {col} がCSVに存在しません。列一覧: {df.columns.tolist()}")

    # 0〜360に正規化（必要に応じて）
    true_roll_norm = df["true_roll"] % 360.0

    bin_w = float(args.bin_width)
    # ビン index (0,1,2,...) を割り当て
    bin_idx = (true_roll_norm // bin_w).astype(int)
    df["roll_bin_idx"] = bin_idx

    # 各ビンの代表角度（中心）を計算
    df["roll_bin_center"] = (df["roll_bin_idx"] + 0.5) * bin_w

    # 集計
    grouped = (
        df.groupby("roll_bin_center")[args.error_col]
        .agg(["mean", "median", "std", "count", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{args.error_col}_mean",
                "median": f"{args.error_col}_median",
                "std": f"{args.error_col}_std",
                "count": "num_samples",
                "max": f"{args.error_col}_max",
            }
        )
        .sort_values("roll_bin_center")
    )

    out_csv = csv_path.with_name("error_by_true_roll_bin.csv")
    grouped.to_csv(out_csv, index=False)
    print(f"[INFO] Saved: {out_csv}")

    # グラフ作成
    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = grouped["roll_bin_center"].to_numpy()
    y_mean = grouped[f"{args.error_col}_mean"].to_numpy()
    y_median = grouped[f"{args.error_col}_median"].to_numpy()
    n_samples = grouped["num_samples"].to_numpy()

    # 左軸: 誤差 (mean, median)
    ax1.plot(x, y_mean, marker="o", label=f"{args.error_col} mean")
    ax1.plot(x, y_median, marker="s", linestyle="--", label=f"{args.error_col} median")
    ax1.set_xlabel("True Roll Bin Center [deg]")
    ax1.set_ylabel("Error [deg]")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{v:.0f}" for v in x], rotation=45)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper left")

    # 右軸: サンプル数
    ax2 = ax1.twinx()
    ax2.bar(x, n_samples, width=bin_w * 0.7, alpha=0.2, color="gray", label="num_samples")
    ax2.set_ylabel("Num Samples")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    out_png = csv_path.with_name("error_by_true_roll_bin.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved plot: {out_png}")


if __name__ == "__main__":
    main()
