#!/usr/bin/env python3
# code_F_rank_datasets_by_rep_experiment.py
# -------------------------------------------------
# comp_XXX_YYY_kf.csv を読み込み、
#   TRAIN_DATASET ごとに全 exp をまとめて
#   各 exp の mean / max / count を計算。
#
# その後、
#   1) mean が最小の exp を「平均ベースの代表」として選び、
#      データセット間で mean 昇順に並べて PNG 出力。
#   2) max が最小の exp を「最大値ベースの代表」として選び、
#      データセット間で max 昇順に並べて PNG 出力。
#
# どちらも TopK で切らず「全データセット分」を表示する。
# -------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "comp_*.csv を読み込み、TRAIN_DATASET ごとに代表実験を選出し、"
            "mean / max ベースでデータセットランキング表 (PNG) を出力するスクリプト"
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
        "--out_prefix",
        default=None,
        help=(
            "出力PNGファイル名のプレフィックス "
            "(省略時: CSV名のstemを使用)"
        ),
    )
    return parser.parse_args()


def make_table_png(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """DataFrame を matplotlib.table で PNG 保存する小ヘルパー"""
    n_rows = len(df)
    fig_height = max(2, 0.4 * n_rows + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"✅ Saved table PNG: {out_path}")
    print(f"  Rows in table: {n_rows}")


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

    # 必須列チェック
    required_cols = ["INPUT_MODE", "IMG_SIZE", "TRAIN_DATASET_ROOT"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' が CSV に存在しません。")

    # TRAIN_DATASET_ROOT を短縮形 (最後のパス要素) にしておく
    df["TRAIN_DATASET"] = df["TRAIN_DATASET_ROOT"].astype(str).apply(
        lambda s: s.split("/")[-1].split("\\")[-1]
    )

    # -------------------------------------------------
    # 1) 各 exp (TRAIN_DATASET, id, INPUT_MODE, IMG_SIZE) ごとの mean / max / count
    # -------------------------------------------------
    group_cols = ["TRAIN_DATASET", id_col, "INPUT_MODE", "IMG_SIZE"]
    exp_stats = (
        df.groupby(group_cols)[metric_col]
        .agg(["mean", "max", "count"])
        .reset_index()
    )

    exp_stats["mean"] = exp_stats["mean"].round(3)
    exp_stats["max"] = exp_stats["max"].round(3)

    # 出力ファイル名のプレフィックス
    if args.out_prefix is not None:
        prefix = Path(args.out_prefix)
    else:
        # CSV の拡張子を除いた部分
        prefix = csv_path.with_suffix("")

    # -------------------------------------------------
    # 2) mean 最小の exp を各 TRAIN_DATASET の代表として選ぶ
    # -------------------------------------------------
    # TRAIN_DATASET 内で mean → max の順にソートしてから groupby.first()
    # こうすることで、mean が同値の場合は max が小さいほうを優先
    exp_sorted_by_mean = exp_stats.sort_values(
        by=["TRAIN_DATASET", "mean", "max"],
        ascending=[True, True, True],
    )
    rep_by_mean = (
        exp_sorted_by_mean
        .groupby("TRAIN_DATASET", as_index=False)
        .first()
    )

    # データセット間で mean 昇順に並べ替え
    rep_by_mean = rep_by_mean.sort_values("mean", ascending=True).reset_index(drop=True)
    rep_by_mean.insert(0, "rank", range(1, len(rep_by_mean) + 1))

    # 出力用に列順を整理
    cols_mean = [
        "rank",
        "TRAIN_DATASET",
        id_col,
        "INPUT_MODE",
        "IMG_SIZE",
        "mean",
        "max",
        "count",
    ]
    rep_by_mean_table = rep_by_mean[cols_mean]

    out_mean_png = prefix.with_name(prefix.name + f"_{metric_col}_dataset_rep_by_mean.png")
    make_table_png(
        rep_by_mean_table,
        out_mean_png,
        title=f"Dataset Representative by Mean {metric_col} (Lower is Better)"
    )

    # -------------------------------------------------
    # 3) max 最小の exp を各 TRAIN_DATASET の代表として選ぶ
    # -------------------------------------------------
    exp_sorted_by_max = exp_stats.sort_values(
        by=["TRAIN_DATASET", "max", "mean"],
        ascending=[True, True, True],
    )
    rep_by_max = (
        exp_sorted_by_max
        .groupby("TRAIN_DATASET", as_index=False)
        .first()
    )

    # データセット間で max 昇順に並べ替え
    rep_by_max = rep_by_max.sort_values("max", ascending=True).reset_index(drop=True)
    rep_by_max.insert(0, "rank", range(1, len(rep_by_max) + 1))

    cols_max = [
        "rank",
        "TRAIN_DATASET",
        id_col,
        "INPUT_MODE",
        "IMG_SIZE",
        "mean",
        "max",
        "count",
    ]
    rep_by_max_table = rep_by_max[cols_max]

    out_max_png = prefix.with_name(prefix.name + f"_{metric_col}_dataset_rep_by_max.png")
    make_table_png(
        rep_by_max_table,
        out_max_png,
        title=f"Dataset Representative by Max {metric_col} (Lower is Better)"
    )


if __name__ == "__main__":
    main()


