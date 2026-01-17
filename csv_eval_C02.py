#!/usr/bin/env python3
"""
C-02: 画像難易度ヒートマップ作成スクリプト

comp_*.csv を読み込み、
  - filename × exp_id で平均誤差を計算し、
  - グローバルに難しい画像TopNを選び、
  - ヒートマップ画像として保存します。

出力:
  - image_difficulty_heatmap.png
  - image_difficulty_table.csv   (ヒートマップの元データ)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="C-02: 画像難易度ヒートマップ作成")
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
        "--topn",
        type=int,
        default=40,
        help="難しい画像TopNをヒートマップに表示 (default: 40)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"[INFO] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    for col in ["filename", "exp_id", args.error_col]:
        if col not in df.columns:
            raise KeyError(f"列 {col} がCSVに存在しません。列一覧: {df.columns.tolist()}")

    # filenameごとのグローバル平均誤差
    filename_mean = (
        df.groupby("filename")[args.error_col]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    # 難しい画像 TopN を抽出
    top_filenames = filename_mean["filename"].head(args.topn).tolist()
    print(f"[INFO] TopN filenames for heatmap (N={args.topn}): {len(top_filenames)}")

    df_top = df[df["filename"].isin(top_filenames)].copy()

    # exp_id × filename の平均誤差をpivot
    pivot = (
        df_top.groupby(["filename", "exp_id"])[args.error_col]
        .mean()
        .reset_index()
        .pivot(index="filename", columns="exp_id", values=args.error_col)
    )

    # exp_id の列順をソート
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # 欠損はNaN -> 可視化のために0 or 別値にしてもよいが、ここではNaNをそのまま保持し、
    # マスクして表示する
    data = pivot.to_numpy()
    mask = np.isnan(data)
    data_for_plot = np.ma.masked_array(data, mask=mask)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.4), max(6, len(pivot.index) * 0.25)))
    im = ax.imshow(data_for_plot, aspect="auto")

    # 軸ラベル
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("exp_id")
    ax.set_ylabel("filename")
    ax.set_title(f"Image Difficulty Heatmap ({args.error_col})")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(args.error_col)

    fig.tight_layout()

    out_png = csv_path.with_name("image_difficulty_heatmap.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved heatmap: {out_png}")

    # ヒートマップの元データも保存しておく
    out_csv = csv_path.with_name("image_difficulty_table.csv")
    pivot.to_csv(out_csv)
    print(f"[INFO] Saved heatmap table: {out_csv}")


if __name__ == "__main__":
    main()
