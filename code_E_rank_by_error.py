#!/usr/bin/env python3
# code_E_rank_by_error.py
# -------------------------------------------------
# comp_XXX_YYY_kf.csv を読み込み、
#   exp_id, INPUT_MODE, IMG_SIZE, TRAIN_DATASET_ROOT ごとに
#   指定メトリック列の mean / max を求め、
#   ・平均誤差ランキング Top10
#   ・最大誤差ランキング Top10
# をそれぞれ matplotlib の table で PNG 保存するスクリプト
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
            "exp_id, INPUT_MODE, IMG_SIZE, TRAIN_DATASET_ROOT ごとの "
            "mean / max を計算し、平均誤差・最大誤差のランキング表(Top10)を "
            "matplotlib で PNG 出力するスクリプト"
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
        "--topk",
        type=int,
        default=10,
        help="ランキングの上位何件までを表示するか (default: 10)",
    )
    parser.add_argument(
        "--out_prefix",
        default=None,
        help="出力PNGファイル名のプレフィックス (省略時: CSV名のstemを使用)",
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

    # 必要な列の存在確認
    required_cols = ["INPUT_MODE", "IMG_SIZE", "TRAIN_DATASET_ROOT"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' が CSV に存在しません。")

    # TRAIN_DATASET_ROOT を短縮形 (最後のパス要素) にしておく
    df["TRAIN_DATASET"] = df["TRAIN_DATASET_ROOT"].astype(str).apply(
        lambda s: s.split("/")[-1].split("\\")[-1]
    )

    # --- groupby して mean / max / count ---
    group_cols = [id_col, "INPUT_MODE", "IMG_SIZE", "TRAIN_DATASET"]
    grouped = (
        df.groupby(group_cols)[metric_col]
        .agg(["mean", "max", "count"])
        .reset_index()
    )

    # 少し丸めて見やすくする
    grouped["mean"] = grouped["mean"].round(3)
    grouped["max"] = grouped["max"].round(3)

    # 出力用名前
    if args.out_prefix is not None:
        prefix = Path(args.out_prefix)
    else:
        prefix = csv_path.with_suffix("")  # stem部分

    # ==============================
    # 1) 平均誤差ランキング TopK
    # ==============================
    ranked_mean = grouped.sort_values("mean", ascending=True).reset_index(drop=True)
    topk_mean = ranked_mean.head(args.topk).copy()

    # rank列を追加
    topk_mean.insert(0, "rank", range(1, len(topk_mean) + 1))

    # 表に出す列を選択
    cols_mean = [
        "rank",
        id_col,
        "INPUT_MODE",
        "IMG_SIZE",
        "TRAIN_DATASET",
        "mean",
        "max",
        "count",
    ]
    table_mean = topk_mean[cols_mean]

    # 図の高さは行数に応じて調整
    n_rows_mean = len(table_mean)
    fig_height_mean = max(2, 0.4 * n_rows_mean + 1)
    fig_mean, ax_mean = plt.subplots(figsize=(10, fig_height_mean))
    ax_mean.axis("off")

    tbl_mean = ax_mean.table(
        cellText=table_mean.values,
        colLabels=table_mean.columns,
        loc="center",
    )
    tbl_mean.auto_set_font_size(False)
    tbl_mean.set_fontsize(8)
    tbl_mean.auto_set_column_width(col=list(range(len(table_mean.columns))))
    ax_mean.set_title(f"Top {args.topk} Experiments by Mean {metric_col} (Lower is Better)")

    plt.tight_layout()
    out_mean_png = prefix.with_name(prefix.name + f"_{metric_col}_mean_rank_top{args.topk}.png")
    fig_mean.savefig(out_mean_png, dpi=200)
    plt.close(fig_mean)
    print(f"✅ Saved mean-ranking table PNG: {out_mean_png}")
    print(f"  Rows in mean-ranking table: {n_rows_mean}")

    # ==============================
    # 2) 最大誤差ランキング TopK
    # ==============================
    ranked_max = grouped.sort_values("max", ascending=True).reset_index(drop=True)
    topk_max = ranked_max.head(args.topk).copy()

    # rank列を追加
    topk_max.insert(0, "rank", range(1, len(topk_max) + 1))

    cols_max = [
        "rank",
        id_col,
        "INPUT_MODE",
        "IMG_SIZE",
        "TRAIN_DATASET",
        "mean",
        "max",
        "count",
    ]
    table_max = topk_max[cols_max]

    n_rows_max = len(table_max)
    fig_height_max = max(2, 0.4 * n_rows_max + 1)
    fig_max, ax_max = plt.subplots(figsize=(10, fig_height_max))
    ax_max.axis("off")

    tbl_max = ax_max.table(
        cellText=table_max.values,
        colLabels=table_max.columns,
        loc="center",
    )
    tbl_max.auto_set_font_size(False)
    tbl_max.set_fontsize(8)
    tbl_max.auto_set_column_width(col=list(range(len(table_max.columns))))
    ax_max.set_title(f"Top {args.topk} Experiments by Max {metric_col} (Lower is Better)")

    plt.tight_layout()
    out_max_png = prefix.with_name(prefix.name + f"_{metric_col}_max_rank_top{args.topk}.png")
    fig_max.savefig(out_max_png, dpi=200)
    plt.close(fig_max)
    print(f"✅ Saved max-ranking table PNG: {out_max_png}")
    print(f"  Rows in max-ranking table: {n_rows_max}")


if __name__ == "__main__":
    main()
