#!/usr/bin/env python3
# code_F_rank_by_inputmode_imgsize.py
# -------------------------------------------------
# comp_XXX_YYY_kf.csv を読み込み、
#   1) 各 exp (TRAIN_DATASET, id, INPUT_MODE, IMG_SIZE) ごとに
#      metric_col の mean / max / count を計算 (exp_stats)。
#
#   2) (INPUT_MODE, IMG_SIZE) ごとに
#      - mean が最小の exp を「mean代表」
#      - max が最小の exp を「max代表」
#      として選出し、代表実験一覧テーブルを PNG で保存。
#
#   3) 代表実験ごとに、元の CSV から全サンプルの metric を集めて
#      「代表 (INPUT_MODE × IMG_SIZE) ごとの誤差分布の箱ひげ図」を作成。
#      - mean代表用の箱ひげ図 PNG
#      - max代表用の箱ひげ図 PNG
#
# 箱ひげ図では：
#   - 外れ値は表示せず (showfliers=False)
#   - ヒゲを min〜max に設定 (whis=[0, 100])
#   として、すべての値をヒゲの範囲に含める。
# -------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 引数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "comp_*.csv を読み込み、(INPUT_MODE, IMG_SIZE) ごとに代表実験を選出し、"
            "代表実験テーブルと代表実験の箱ひげ図 (PNG) を出力するスクリプト"
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
        help="mean / max / 箱ひげ図を作る対象となる列名 (default: kf_err)",
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


# ============================================================
# テーブル出力ヘルパ
# ============================================================
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


# ============================================================
# 代表実験の箱ひげ図出力ヘルパ
# ============================================================
def make_rep_boxplot(
    df: pd.DataFrame,
    rep_df: pd.DataFrame,
    id_col: str,
    metric_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """
    (INPUT_MODE, IMG_SIZE) ごとの代表実験 (rep_df) に対して、
    元の df から該当実験の全サンプルの metric_col を取り出し、
    代表 (INPUT_MODE × IMG_SIZE) ごとの誤差分布を箱ひげ図で描画して PNG 保存する。

    rep_df: 各行が 1つの代表実験
        必須列: TRAIN_DATASET, id_col, INPUT_MODE, IMG_SIZE
    """
    data = []
    labels = []

    for _, row in rep_df.iterrows():
        train_dataset = row["TRAIN_DATASET"]
        exp_id = row[id_col]
        input_mode = row["INPUT_MODE"]
        img_size = row["IMG_SIZE"]

        mask = (
            (df["TRAIN_DATASET"] == train_dataset)
            & (df[id_col] == exp_id)
            & (df["INPUT_MODE"] == input_mode)
            & (df["IMG_SIZE"] == img_size)
        )

        values = df.loc[mask, metric_col].values

        if len(values) == 0:
            # 対応するサンプルが無い場合はスキップ
            print(
                f"⚠ WARN: No samples found for representative exp: "
                f"TRAIN_DATASET={train_dataset}, {id_col}={exp_id}, "
                f"INPUT_MODE={input_mode}, IMG_SIZE={img_size}"
            )
            continue

        data.append(values)
        # ラベルは 2行構成
        #  1行目: 色 × 画像サイズ
        #  2行目: id と TRAIN_DATASET (補足情報)
        label = f"{input_mode} × {img_size}\n{id_col}={exp_id}, ds={train_dataset}"
        labels.append(label)

    n_reps = len(data)
    if n_reps == 0:
        print("⚠ WARN: No representative experiments to plot. Skipping boxplot.")
        return

    fig_width = max(8, 0.8 * n_reps)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # 外れ値を表示せず (showfliers=False)、ヒゲを min〜max に (whis=[0,100])
    ax.boxplot(
        data,
        labels=labels,
        showmeans=False,
        showfliers=False,
        whis=[0, 100],
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("INPUT_MODE × IMG_SIZE (Representative EXP)")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"✅ Saved representative boxplot PNG: {out_path}")
    print(f"  Number of representative experiments: {n_reps}")


# ============================================================
# main
# ============================================================
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
    # 2) (INPUT_MODE, IMG_SIZE) ごとに代表実験を選ぶ (mean ベース)
    #    - 同じ (INPUT_MODE, IMG_SIZE) 内で mean → max の順にソート
    #    - mean が同値の場合は max が小さいほうを優先
    # -------------------------------------------------
    exp_sorted_by_mean = exp_stats.sort_values(
        by=["INPUT_MODE", "IMG_SIZE", "mean", "max"],
        ascending=[True, True, True, True],
    )
    rep_by_mean = (
        exp_sorted_by_mean
        .groupby(["INPUT_MODE", "IMG_SIZE"], as_index=False)
        .first()
    )

    # (INPUT_MODE, IMG_SIZE) 組の中で mean 昇順に並べ替え & rank 付与
    rep_by_mean = rep_by_mean.sort_values("mean", ascending=True).reset_index(drop=True)
    rep_by_mean.insert(0, "rank", range(1, len(rep_by_mean) + 1))

    cols_mean = [
        "rank",
        "INPUT_MODE",
        "IMG_SIZE",
        id_col,
        "TRAIN_DATASET",
        "mean",
        "max",
        "count",
    ]
    rep_by_mean_table = rep_by_mean[cols_mean]

    out_mean_png = prefix.with_name(prefix.name + f"_{metric_col}_rep_by_mean_inputmode_imgsize.png")
    make_table_png(
        rep_by_mean_table,
        out_mean_png,
        title=f"Representative by Mean {metric_col} per (INPUT_MODE, IMG_SIZE) (Lower is Better)"
    )

    # -------------------------------------------------
    # 3) (INPUT_MODE, IMG_SIZE) ごとに代表実験を選ぶ (max ベース)
    #    - 同じ (INPUT_MODE, IMG_SIZE) 内で max → mean の順にソート
    #    - max が同値の場合は mean が小さいほうを優先
    # -------------------------------------------------
    exp_sorted_by_max = exp_stats.sort_values(
        by=["INPUT_MODE", "IMG_SIZE", "max", "mean"],
        ascending=[True, True, True, True],
    )
    rep_by_max = (
        exp_sorted_by_max
        .groupby(["INPUT_MODE", "IMG_SIZE"], as_index=False)
        .first()
    )

    # (INPUT_MODE, IMG_SIZE) 組の中で max 昇順に並べ替え & rank 付与
    rep_by_max = rep_by_max.sort_values("max", ascending=True).reset_index(drop=True)
    rep_by_max.insert(0, "rank", range(1, len(rep_by_max) + 1))

    cols_max = [
        "rank",
        "INPUT_MODE",
        "IMG_SIZE",
        id_col,
        "TRAIN_DATASET",
        "mean",
        "max",
        "count",
    ]
    rep_by_max_table = rep_by_max[cols_max]

    out_max_png = prefix.with_name(prefix.name + f"_{metric_col}_rep_by_max_inputmode_imgsize.png")
    make_table_png(
        rep_by_max_table,
        out_max_png,
        title=f"Representative by Max {metric_col} per (INPUT_MODE, IMG_SIZE) (Lower is Better)"
    )

    # -------------------------------------------------
    # 4) 代表実験ごとの箱ひげ図を出力する
    #    - mean代表用 (rep_by_mean)
    #    - max代表用  (rep_by_max)
    #    どちらも元の df からサンプルを拾って箱ひげ図を作る
    # -------------------------------------------------
    # mean代表
    rep_mean_for_box = rep_by_mean[
        ["TRAIN_DATASET", id_col, "INPUT_MODE", "IMG_SIZE"]
    ].copy()
    out_box_mean = prefix.with_name(prefix.name + f"_{metric_col}_rep_by_mean_boxplot_inputmode_imgsize.png")
    make_rep_boxplot(
        df=df,
        rep_df=rep_mean_for_box,
        id_col=id_col,
        metric_col=metric_col,
        out_path=out_box_mean,
        title=f"Representative per (INPUT_MODE, IMG_SIZE) (by Mean {metric_col})",
        ylabel=f"{metric_col} (lower is better)",
    )

    # max代表
    rep_max_for_box = rep_by_max[
        ["TRAIN_DATASET", id_col, "INPUT_MODE", "IMG_SIZE"]
    ].copy()
    out_box_max = prefix.with_name(prefix.name + f"_{metric_col}_rep_by_max_boxplot_inputmode_imgsize.png")
    make_rep_boxplot(
        df=df,
        rep_df=rep_max_for_box,
        id_col=id_col,
        metric_col=metric_col,
        out_path=out_box_max,
        title=f"Representative per (INPUT_MODE, IMG_SIZE) (by Max {metric_col})",
        ylabel=f"{metric_col} (lower is better)",
    )


if __name__ == "__main__":
    main()
