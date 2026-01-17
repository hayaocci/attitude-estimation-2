#!/usr/bin/env python3
"""
A-03: 画像単位の誤差安定性解析スクリプト

comp_*.csv を読み込み、
filename ごとに誤差を集計して「難しい画像ランキング」を作成します。

出力:
  - hard_images_by_filename.csv  (集計結果)
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="A-03: 画像単位の誤差安定性解析")
    p.add_argument(
        "--csv",
        required=True,
        help="code_C で生成した comp_*.csv のパス",
    )
    p.add_argument(
        "--error_col",
        default="kf_err",
        help="使用する誤差列名 (default: kf_err, 例: err_roll / kf_err)",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=50,
        help="コンソールに表示する上位画像数 (default: 50)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"[INFO] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    if args.error_col not in df.columns:
        raise KeyError(f"指定された誤差列 {args.error_col} が見つかりません。列一覧: {df.columns.tolist()}")

    # filename ごとに誤差を集計
    grouped = (
        df.groupby("filename")[args.error_col]
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
    )

    # 難しい(誤差が大きい)順にソート
    grouped_sorted = grouped.sort_values(
        by=f"{args.error_col}_mean", ascending=False
    ).reset_index(drop=True)

    out_path = csv_path.with_name("hard_images_by_filename.csv")
    grouped_sorted.to_csv(out_path, index=False)
    print(f"[INFO] Saved: {out_path}")

    # 上位だけコンソール表示
    print("\n[TOP HARD IMAGES]")
    print(
        grouped_sorted.head(args.topk)[
            ["filename", f"{args.error_col}_mean", f"{args.error_col}_max", "num_samples"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
