#!/usr/bin/env python3
# code_D_make_table.py
# -------------------------------------------------
# comp_XXX_YYY_kf.csv を読み込み、
#  id(=exp_id), INPUT_MODE, IMG_SIZE ごとに
#  指定メトリック列の mean / max を求め、
#  matplotlib の table で PNG を保存する
# -------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "comp_*.csv を読み込み、"
            "id(=exp_id), INPUT_MODE, IMG_SIZE ごとの mean / max を求めて "
            "matplotlib の表を PNG で保存するスクリプト"
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="code_C_compile_results.py で作成した comp_*.csv のパス",
    )
    parser.add_argument(
        "--metric",
        default="kf_err",
        help="mean / max を計算する列名 (default: kf_err)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="出力 PNG ファイル名 (省略時: CSV と同じ場所に metric_table.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV が見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)

    # --- id 列の決定 (exp_id があればそれを使う) ---
    if "exp_id" in df.columns:
        id_col = "exp_id"
    elif "id" in df.columns:
        id_col = "id"
    else:
        raise ValueError("id を表す列 (exp_id or id) が見つかりません。")

    metric_col = args.metric
    if metric_col not in df.columns:
        raise ValueError(
            f"指定メトリック列 '{metric_col}' が CSV に存在しません。\n"
            f"利用可能な列: {list(df.columns)}"
        )

    # INPUT_MODE / IMG_SIZE が無い場合はエラーにしておく
    for col in ["INPUT_MODE", "IMG_SIZE"]:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' が CSV に存在しません。")

    # --- groupby して mean / max ---
    group_cols = [id_col, "INPUT_MODE", "IMG_SIZE", "MAX_LR"]
    grouped = (
        df.groupby(group_cols)[metric_col]
        .agg(["mean", "max"])
        .reset_index()
    )

    # 少し丸めて見やすくする
    grouped["mean"] = grouped["mean"].round(3)
    grouped["max"] = grouped["max"].round(3)

    # --- table を matplotlib で描画 ---
    # 行数に応じて高さを調整
    n_rows = len(grouped)
    fig_height = max(2, 0.4 * n_rows + 1)  # 適当に調整
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.axis("off")

    table = ax.table(
        cellText=grouped.values,
        colLabels=grouped.columns,
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(grouped.columns))))

    plt.tight_layout()

    # 出力パス
    if args.out is not None:
        out_path = Path(args.out)
    else:
        out_path = csv_path.with_name(csv_path.stem + f"_{metric_col}_table.png")

    fig.savefig(out_path, dpi=200)
    print(f"✅ Saved table PNG: {out_path}")
    print(f"  Rows in table: {n_rows}")


if __name__ == "__main__":
    main()
