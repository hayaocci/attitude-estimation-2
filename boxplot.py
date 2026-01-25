# #!/usr/bin/env python3
# # code_F_rank_experiments_top9.py
# # -------------------------------------------------
# # comp_XXX_YYY_kf.csv を読み込み、
# #
# #   1) 各 exp (TRAIN_DATASET, id, INPUT_MODE, IMG_SIZE) ごとに
# #      metric_col の mean / median / max / IQR / count を計算 (exp_stats)。
# #
# #   2) 「全実験 exp_stats」を使って、以下の 5 種類のテーブル PNG を出力：
# #      - 全実験テーブル
# #      - max が小さい順 TOP9 テーブル
# #      - mean が小さい順 TOP9 テーブル
# #      - median が小さい順 TOP9 テーブル
# #      - IQR が小さい順 TOP9 テーブル
# #
# #   3) 元の df からサンプル単位で metric_col を取り出して箱ひげ図を作成：
# #      - 全実験を mean 昇順に並べた箱ひげ図
# #      - max が小さい順 TOP9 の箱ひげ図
# #      - mean が小さい順 TOP9 の箱ひげ図
# #      - median が小さい順 TOP9 の箱ひげ図
# #      - IQR が小さい順 TOP9 の箱ひげ図
# #
# # 箱ひげ図では：
# #   - 外れ値は表示せず (showfliers=False)
# #   - ヒゲを min〜max に設定 (whis=[0, 100])
# #   として、すべての値をヒゲの範囲に含める。
# # -------------------------------------------------
# from __future__ import annotations
# import argparse
# from pathlib import Path

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt


# # ============================================================
# # 引数
# # ============================================================
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description=(
#             "comp_*.csv を読み込み、全実験を評価して "
#             "max/mean/median/IQR の TOP9 テーブルと箱ひげ図、"
#             "および全実験テーブル・箱ひげ図を出力するスクリプト"
#         )
#     )
#     parser.add_argument(
#         "--csv",
#         required=True,
#         help="code_C_compile_results.py で作成した comp_*.csv のパス",
#     )
#     parser.add_argument(
#         "--metric",
#         default="err_roll",  # ★ CNN 生誤差をデフォルトに
#         help="mean / max / 箱ひげ図を作る対象となる列名 (default: err_roll)",
#     )
#     parser.add_argument(
#         "--out_prefix",
#         default=None,
#         help=(
#             "出力PNGファイル名のプレフィックス "
#             "(省略時: CSV名のstemを使用)"
#         ),
#     )
#     return parser.parse_args()


# # ============================================================
# # テーブル出力ヘルパ
# # ============================================================
# def make_table_png(df: pd.DataFrame, out_path: Path, title: str) -> None:
#     """DataFrame を matplotlib.table で PNG 保存する小ヘルパー"""
#     n_rows = len(df)
#     fig_height = max(2, 0.4 * n_rows + 1)
#     fig, ax = plt.subplots(figsize=(12, fig_height))
#     ax.axis("off")

#     table = ax.table(
#         cellText=df.values,
#         colLabels=df.columns,
#         loc="center",
#     )
#     table.auto_set_font_size(False)
#     table.set_fontsize(8)
#     table.auto_set_column_width(col=list(range(len(df.columns))))

#     ax.set_title(title)
#     plt.tight_layout()
#     fig.savefig(out_path, dpi=200)
#     plt.close(fig)
#     print(f"✅ Saved table PNG: {out_path}")
#     print(f"  Rows in table: {n_rows}")


# # ============================================================
# # 部分集合の実験について箱ひげ図を作成
# # ============================================================
# def make_exp_boxplot(
#     df: pd.DataFrame,
#     exp_df: pd.DataFrame,
#     id_col: str,
#     metric_col: str,
#     out_path: Path,
#     title: str,
#     ylabel: str,
# ) -> None:
#     """
#     exp_df に含まれる各実験 (行) について、
#     元の df から (TRAIN_DATASET, id_col, INPUT_MODE, IMG_SIZE) でフィルタし、
#     metric_col の分布を箱ひげ図で描画する。
#     exp_df の行順をそのまま横軸の並び順として使用。
#     """
#     data = []
#     labels = []

#     for _, row in exp_df.iterrows():
#         train_ds = row["TRAIN_DATASET"]
#         exp_id = row[id_col]
#         input_mode = row["INPUT_MODE"]
#         img_size = row["IMG_SIZE"]

#         mask = (
#             (df["TRAIN_DATASET"] == train_ds)
#             & (df[id_col] == exp_id)
#             & (df["INPUT_MODE"] == input_mode)
#             & (df["IMG_SIZE"] == img_size)
#         )
#         values = df.loc[mask, metric_col].dropna().values
#         if len(values) == 0:
#             print(
#                 f"⚠ WARN: No samples found for exp: "
#                 f"TRAIN_DATASET={train_ds}, {id_col}={exp_id}, "
#                 f"INPUT_MODE={input_mode}, IMG_SIZE={img_size}"
#             )
#             continue

#         # ラベル用情報
#         color_mode = str(row.get("COLOR_MODE", ""))
#         valid_ds = str(row.get("VALID_DATASET", ""))
#         valid_blur = bool(row.get("VALID_BLUR", False))
#         vk = row.get("VALID_BLUR_KERNEL", "")
#         if pd.isna(vk):
#             vk = ""
#         blur_str = f"blur={int(vk)}" if valid_blur and vk != "" else "blur=None"

#         label = (
#             f"{input_mode}, {img_size}px, {color_mode}, {blur_str}\n"
#             f"{id_col}={exp_id}, train={train_ds}, valid={valid_ds}"
#         )

#         data.append(values)
#         labels.append(label)

#     n_groups = len(data)
#     if n_groups == 0:
#         print(f"⚠ WARN: No experiment groups to plot for {out_path}. Skipping.")
#         return

#     fig_width = max(10, 0.9 * n_groups)
#     fig, ax = plt.subplots(figsize=(fig_width, 6))

#     ax.boxplot(
#         data,
#         labels=labels,
#         showmeans=False,
#         showfliers=False,
#         whis=[0, 100],
#     )

#     ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.set_xlabel("Experiments")

#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)

#     plt.tight_layout()
#     fig.savefig(out_path, dpi=200)
#     plt.close(fig)

#     print(f"✅ Saved boxplot PNG: {out_path}")
#     print(f"  Number of experiments (boxplots): {n_groups}")


# # ============================================================
# # main
# # ============================================================
# def main():
#     args = parse_args()

#     csv_path = Path(args.csv)
#     if not csv_path.exists():
#         raise FileNotFoundError(f"CSV が見つかりません: {csv_path}")

#     df = pd.read_csv(csv_path)

#     # --- id 列の決定 (exp_id があればそれを使う) ---
#     if "exp_id" in df.columns:
#         id_col = "exp_id"
#     elif "id" in df.columns:
#         id_col = "id"
#     else:
#         raise ValueError("id を表す列 (exp_id or id) が見つかりません。")

#     metric_col = args.metric
#     if metric_col not in df.columns:
#         raise ValueError(
#             f"指定メトリック列 '{metric_col}' が CSV に存在しません。\n"
#             f"利用可能な列: {list(df.columns)}"
#         )

#     # 必須列チェック
#     required_cols = ["INPUT_MODE", "IMG_SIZE", "TRAIN_DATASET_ROOT"]
#     for col in required_cols:
#         if col not in df.columns:
#             raise ValueError(f"列 '{col}' が CSV に存在しません。")

#     # TRAIN / VALID の短縮名列を追加
#     df["TRAIN_DATASET"] = df["TRAIN_DATASET_ROOT"].astype(str).apply(
#         lambda s: s.split("/")[-1].split("\\")[-1]
#     )
#     if "VALID_DATASET_ROOT" in df.columns:
#         df["VALID_DATASET"] = df["VALID_DATASET_ROOT"].astype(str).apply(
#             lambda s: s.split("/")[-1].split("\\")[-1]
#         )
#     else:
#         df["VALID_DATASET"] = ""

#     # -------------------------------------------------
#     # 1) 各 exp (TRAIN_DATASET, id, INPUT_MODE, IMG_SIZE) ごとの
#     #    mean / median / max / IQR / count を計算
#     # -------------------------------------------------
#     group_cols = ["TRAIN_DATASET", id_col, "INPUT_MODE", "IMG_SIZE"]
#     grouped_metric = df.groupby(group_cols)[metric_col]

#     # 基本統計量
#     exp_stats = grouped_metric.agg(["mean", "median", "max", "count"]).reset_index()

#     # IQR（Q3 - Q1）を計算
#     iqrs = []
#     for keys, _ in grouped_metric:
#         sub = grouped_metric.get_group(keys).dropna().values
#         if len(sub) == 0:
#             iqrs.append(np.nan)
#         else:
#             q1 = np.percentile(sub, 25)
#             q3 = np.percentile(sub, 75)
#             iqrs.append(q3 - q1)
#     exp_stats["iqr"] = iqrs

#     # 追加情報（VALID_DATASET / COLOR_MODE / BLUR など）を付与
#     for col in ["VALID_DATASET", "COLOR_MODE", "VALID_BLUR", "VALID_BLUR_KERNEL"]:
#         if col in df.columns:
#             extra_series = (
#                 df.groupby(group_cols)[col]
#                 .agg(lambda x: x.iloc[0])
#                 .reset_index()[col]
#             )
#             exp_stats[col] = extra_series

#     exp_stats["mean"] = exp_stats["mean"].round(3)
#     exp_stats["median"] = exp_stats["median"].round(3)
#     exp_stats["max"] = exp_stats["max"].round(3)
#     exp_stats["iqr"] = exp_stats["iqr"].round(3)

#     # 出力ファイル名のプレフィックス
#     if args.out_prefix is not None:
#         prefix = Path(args.out_prefix)
#     else:
#         # CSV の拡張子を除いた部分
#         prefix = csv_path.with_suffix("")

#     # -------------------------------------------------
#     # 2) 全実験テーブル (All Experiments)
#     # -------------------------------------------------
#     all_cols = [
#         "TRAIN_DATASET",
#         "VALID_DATASET",
#         id_col,
#         "INPUT_MODE",
#         "IMG_SIZE",
#         "COLOR_MODE",
#         "VALID_BLUR",
#         "VALID_BLUR_KERNEL",
#         "mean",
#         "median",
#         "max",
#         "iqr",
#         "count",
#     ]
#     all_cols = [c for c in all_cols if c in exp_stats.columns]
#     all_exp_table = exp_stats[all_cols].sort_values(
#         by=["INPUT_MODE", "IMG_SIZE", "TRAIN_DATASET", id_col]
#     )

#     out_all_table = prefix.with_name(
#         prefix.name + f"_{metric_col}_all_experiments_table.png"
#     )
#     make_table_png(
#         all_exp_table,
#         out_all_table,
#         title=f"All Experiments Summary ({metric_col})",
#     )

#     # -------------------------------------------------
#     # 3) 各指標で TOP9 を抽出（max / mean / median / iqr）
#     # -------------------------------------------------
#     def top9_sorted_by(col_name: str) -> pd.DataFrame:
#         df_sorted = exp_stats.sort_values(col_name, ascending=True)
#         # NaN は一番後ろに行くように dropna→append でも可だが、
#         # ここでは NaN があってもそのまま末尾に来る想定。
#         return df_sorted.head(9).copy()

#     top9_by_max = top9_sorted_by("max")
#     top9_by_mean = top9_sorted_by("mean")
#     top9_by_median = top9_sorted_by("median")
#     top9_by_iqr = top9_sorted_by("iqr")

#     # rank 列を付与してテーブル用にする
#     def add_rank(df_rank: pd.DataFrame) -> pd.DataFrame:
#         df_rank = df_rank.copy().reset_index(drop=True)
#         df_rank.insert(0, "rank", range(1, len(df_rank) + 1))
#         cols = [
#             "rank",
#             "TRAIN_DATASET",
#             "VALID_DATASET",
#             id_col,
#             "INPUT_MODE",
#             "IMG_SIZE",
#             "COLOR_MODE",
#             "VALID_BLUR",
#             "VALID_BLUR_KERNEL",
#             "mean",
#             "median",
#             "max",
#             "iqr",
#             "count",
#         ]
#         cols = [c for c in cols if c in df_rank.columns]
#         return df_rank[cols]

#     top9_by_max_table = add_rank(top9_by_max)
#     top9_by_mean_table = add_rank(top9_by_mean)
#     top9_by_median_table = add_rank(top9_by_median)
#     top9_by_iqr_table = add_rank(top9_by_iqr)

#     # -------------------------------------------------
#     # 4) TOP9 テーブル PNG 出力
#     # -------------------------------------------------
#     out_top9_max_table = prefix.with_name(
#         prefix.name + f"_{metric_col}_top9_by_max_table.png"
#     )
#     make_table_png(
#         top9_by_max_table,
#         out_top9_max_table,
#         title=f"Top 9 Experiments (sorted by MAX {metric_col}, lower is better)",
#     )

#     out_top9_mean_table = prefix.with_name(
#         prefix.name + f"_{metric_col}_top9_by_mean_table.png"
#     )
#     make_table_png(
#         top9_by_mean_table,
#         out_top9_mean_table,
#         title=f"Top 9 Experiments (sorted by MEAN {metric_col}, lower is better)",
#     )

#     out_top9_median_table = prefix.with_name(
#         prefix.name + f"_{metric_col}_top9_by_median_table.png"
#     )
#     make_table_png(
#         top9_by_median_table,
#         out_top9_median_table,
#         title=f"Top 9 Experiments (sorted by MEDIAN {metric_col}, lower is better)",
#     )

#     out_top9_iqr_table = prefix.with_name(
#         prefix.name + f"_{metric_col}_top9_by_iqr_table.png"
#     )
#     make_table_png(
#         top9_by_iqr_table,
#         out_top9_iqr_table,
#         title=f"Top 9 Experiments (sorted by IQR {metric_col}, smaller = more stable)",
#     )

#     # -------------------------------------------------
#     # 5) 箱ひげ図：全実験（mean 昇順で並べる）
#     # -------------------------------------------------
#     all_for_box = exp_stats.sort_values("mean", ascending=True)
#     out_all_box = prefix.with_name(
#         prefix.name + f"_{metric_col}_all_experiments_boxplot.png"
#     )
#     make_exp_boxplot(
#         df=df,
#         exp_df=all_for_box,
#         id_col=id_col,
#         metric_col=metric_col,
#         out_path=out_all_box,
#         title=f"All Experiments (sorted by MEAN {metric_col})",
#         ylabel=f"{metric_col} (lower is better)",
#     )

#     # -------------------------------------------------
#     # 6) 箱ひげ図：各指標で TOP9
#     # -------------------------------------------------
#     # max
#     out_top9_max_box = prefix.with_name(
#         prefix.name + f"_{metric_col}_top9_by_max_boxplot.png"
#     )
#     make_exp_boxplot(
#         df=df,
#         exp_df=top9_by_max.sort_values("max", ascending=True),
#         id_col=id_col,
#         metric_col=metric_col,
#         out_path=out_top9_max_box,
#         title=f"Top 9 Experiments (sorted by MAX {metric_col})",
#         ylabel=f"{metric_col} (lower is better)",
#     )

#     # mean
#     out_top9_mean_box = prefix.with_name(
#         prefix.name + f"_{metric_col}_top9_by_mean_boxplot.png"
#     )
#     make_exp_boxplot(
#         df=df,
#         exp_df=top9_by_mean.sort_values("mean", ascending=True),
#         id_col=id_col,
#         metric_col=metric_col,
#         out_path=out_top9_mean_box,
#         title=f"Top 9 Experiments (sorted by MEAN {metric_col})",
#         ylabel=f"{metric_col} (lower is better)",
#     )

#     # median
#     out_top9_median_box = prefix.with_name(
#         prefix.name + f"_{metric_col}_top9_by_median_boxplot.png"
#     )
#     make_exp_boxplot(
#         df=df,
#         exp_df=top9_by_median.sort_values("median", ascending=True),
#         id_col=id_col,
#         metric_col=metric_col,
#         out_path=out_top9_median_box,
#         title=f"Top 9 Experiments (sorted by MEDIAN {metric_col})",
#         ylabel=f"{metric_col} (lower is better)",
#     )

#     # iqr
#     out_top9_iqr_box = prefix.with_name(
#         prefix.name + f"_{metric_col}_top9_by_iqr_boxplot.png"
#     )
#     make_exp_boxplot(
#         df=df,
#         exp_df=top9_by_iqr.sort_values("iqr", ascending=True),
#         id_col=id_col,
#         metric_col=metric_col,
#         out_path=out_top9_iqr_box,
#         title=f"Top 9 Experiments (sorted by IQR {metric_col})",
#         ylabel=f"{metric_col} (smaller = more stable)",
#     )


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# code_F_rank_experiments_topK.py
# -------------------------------------------------
# comp_XXX_YYY_kf.csv を読み込み、
#
#   1) 各 exp (TRAIN_DATASET, id, INPUT_MODE, IMG_SIZE) ごとに
#      metric_col の mean / median / max / IQR / count を計算 (exp_stats)。
#
#   2) 「全実験 exp_stats」を使って、以下の 5 種類のテーブル PNG を出力：
#      - 全実験テーブル
#      - max が小さい順 TOPK テーブル
#      - mean が小さい順 TOPK テーブル
#      - median が小さい順 TOPK テーブル
#      - IQR が小さい順 TOPK テーブル
#
#   3) 元の df からサンプル単位で metric_col を取り出して箱ひげ図を作成：
#      - 全実験を mean 昇順に並べた箱ひげ図
#      - max が小さい順 TOPK の箱ひげ図
#      - mean が小さい順 TOPK の箱ひげ図
#      - median が小さい順 TOPK の箱ひげ図
#      - IQR が小さい順 TOPK の箱ひげ図
#
#  すべての PNG は、prefix ベースの親フォルダを作成して
#  その中にまとめて保存する。
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
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 引数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "comp_*.csv を読み込み、全実験を評価して "
            "max/mean/median/IQR の TOPK テーブルと箱ひげ図、"
            "および全実験テーブル・箱ひげ図を出力するスクリプト"
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="code_C_compile_results.py で作成した comp_*.csv のパス",
    )
    parser.add_argument(
        "--metric",
        default="err_roll",  # ★ CNN 生誤差をデフォルトに
        help="mean / max / 箱ひげ図を作る対象となる列名 (default: err_roll)",
    )
    parser.add_argument(
        "--out_prefix",
        default=None,
        help=(
            "出力PNGファイル名のプレフィックス "
            "(省略時: CSV名のstemを使用)"
        ),
    )
    parser.add_argument(
        "--topK",
        type=int,
        default=9,
        help="ランキング出力で抽出する上位K件 (default: 9)",
    )
    return parser.parse_args()


# ============================================================
# テーブル出力ヘルパ
# ============================================================
def make_table_png(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """DataFrame を matplotlib.table で PNG 保存する小ヘルパー"""
    n_rows = len(df)
    fig_height = max(2, 0.4 * n_rows + 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))
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
# 部分集合の実験について箱ひげ図を作成
# ============================================================
def make_exp_boxplot(
    df: pd.DataFrame,
    exp_df: pd.DataFrame,
    id_col: str,
    metric_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """
    exp_df に含まれる各実験 (行) について、
    元の df から (TRAIN_DATASET, id_col, INPUT_MODE, IMG_SIZE) でフィルタし、
    metric_col の分布を箱ひげ図で描画する。
    exp_df の行順をそのまま横軸の並び順として使用。
    """
    data = []
    labels = []

    for _, row in exp_df.iterrows():
        train_ds = row["TRAIN_DATASET"]
        exp_id = row[id_col]
        input_mode = row["INPUT_MODE"]
        img_size = row["IMG_SIZE"]

        mask = (
            (df["TRAIN_DATASET"] == train_ds)
            & (df[id_col] == exp_id)
            & (df["INPUT_MODE"] == input_mode)
            & (df["IMG_SIZE"] == img_size)
        )
        values = df.loc[mask, metric_col].dropna().values
        if len(values) == 0:
            print(
                f"⚠ WARN: No samples found for exp: "
                f"TRAIN_DATASET={train_ds}, {id_col}={exp_id}, "
                f"INPUT_MODE={input_mode}, IMG_SIZE={img_size}"
            )
            continue

        # ラベル用情報
        color_mode = str(row.get("COLOR_MODE", ""))
        valid_ds = str(row.get("VALID_DATASET", ""))
        valid_blur = bool(row.get("VALID_BLUR", False))
        vk = row.get("VALID_BLUR_KERNEL", "")
        if pd.isna(vk):
            vk = ""
        blur_str = f"blur={int(vk)}" if valid_blur and vk != "" else "blur=None"

        label = (
            f"{input_mode}, {img_size}px, {color_mode}, {blur_str}\n"
            f"{id_col}={exp_id}, train={train_ds}, valid={valid_ds}"
        )

        data.append(values)
        labels.append(label)

    n_groups = len(data)
    if n_groups == 0:
        print(f"⚠ WARN: No experiment groups to plot for {out_path}. Skipping.")
        return

    fig_width = max(10, 0.9 * n_groups)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.boxplot(
        data,
        labels=labels,
        showmeans=False,
        showfliers=False,
        whis=[0, 100],
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Experiments")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"✅ Saved boxplot PNG: {out_path}")
    print(f"  Number of experiments (boxplots): {n_groups}")


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

    topK = args.topK
    if topK <= 0:
        raise ValueError("--topK は 1 以上にしてください。")

    # 必須列チェック
    required_cols = ["INPUT_MODE", "IMG_SIZE", "TRAIN_DATASET_ROOT"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' が CSV に存在しません。")

    # TRAIN / VALID の短縮名列を追加
    df["TRAIN_DATASET"] = df["TRAIN_DATASET_ROOT"].astype(str).apply(
        lambda s: s.split("/")[-1].split("\\")[-1]
    )
    if "VALID_DATASET_ROOT" in df.columns:
        df["VALID_DATASET"] = df["VALID_DATASET_ROOT"].astype(str).apply(
            lambda s: s.split("/")[-1].split("\\")[-1]
        )
    else:
        df["VALID_DATASET"] = ""

    # -------------------------------------------------
    # 1) 各 exp (TRAIN_DATASET, id, INPUT_MODE, IMG_SIZE) ごとの
    #    mean / median / max / IQR / count を計算
    # -------------------------------------------------
    group_cols = ["TRAIN_DATASET", id_col, "INPUT_MODE", "IMG_SIZE"]
    grouped_metric = df.groupby(group_cols)[metric_col]

    # 基本統計量
    exp_stats = grouped_metric.agg(["mean", "median", "max", "count"]).reset_index()

    # IQR（Q3 - Q1）を計算
    iqrs = []
    for keys, _ in grouped_metric:
        sub = grouped_metric.get_group(keys).dropna().values
        if len(sub) == 0:
            iqrs.append(np.nan)
        else:
            q1 = np.percentile(sub, 25)
            q3 = np.percentile(sub, 75)
            iqrs.append(q3 - q1)
    exp_stats["iqr"] = iqrs

    # 追加情報（VALID_DATASET / COLOR_MODE / BLUR など）を付与
    for col in ["VALID_DATASET", "COLOR_MODE", "VALID_BLUR", "VALID_BLUR_KERNEL"]:
        if col in df.columns:
            extra_series = (
                df.groupby(group_cols)[col]
                .agg(lambda x: x.iloc[0])
                .reset_index()[col]
            )
            exp_stats[col] = extra_series

    exp_stats["mean"] = exp_stats["mean"].round(3)
    exp_stats["median"] = exp_stats["median"].round(3)
    exp_stats["max"] = exp_stats["max"].round(3)
    exp_stats["iqr"] = exp_stats["iqr"].round(3)

    # 出力ファイル名のプレフィックス
    if args.out_prefix is not None:
        prefix = Path(args.out_prefix)
    else:
        # CSV の拡張子を除いた部分
        prefix = csv_path.with_suffix("")

    # 🔹 出力用の親フォルダを作成
    #    例) comp_type-8_kf.csv → comp_type-8_kf_err_roll_rank/
    out_dir = prefix.parent / f"{prefix.name}_{metric_col}_rank"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {out_dir}")

    # -------------------------------------------------
    # 2) 全実験テーブル (All Experiments)
    # -------------------------------------------------
    all_cols = [
        "TRAIN_DATASET",
        "VALID_DATASET",
        id_col,
        "INPUT_MODE",
        "IMG_SIZE",
        "COLOR_MODE",
        "VALID_BLUR",
        "VALID_BLUR_KERNEL",
        "mean",
        "median",
        "max",
        "iqr",
        "count",
    ]
    all_cols = [c for c in all_cols if c in exp_stats.columns]
    all_exp_table = exp_stats[all_cols].sort_values(
        by=["INPUT_MODE", "IMG_SIZE", "TRAIN_DATASET", id_col]
    )

    out_all_table = out_dir / f"{prefix.name}_{metric_col}_all_experiments_table.png"
    make_table_png(
        all_exp_table,
        out_all_table,
        title=f"All Experiments Summary ({metric_col})",
    )

    # -------------------------------------------------
    # 3) 各指標で TOPK を抽出（max / mean / median / iqr）
    # -------------------------------------------------
    def topK_sorted_by(col_name: str) -> pd.DataFrame:
        df_sorted = exp_stats.sort_values(col_name, ascending=True)
        # NaN は末尾に来る想定。そのまま head(K) でOK。
        return df_sorted.head(topK).copy()

    topK_by_max = topK_sorted_by("max")
    topK_by_mean = topK_sorted_by("mean")
    topK_by_median = topK_sorted_by("median")
    topK_by_iqr = topK_sorted_by("iqr")

    # rank 列を付与してテーブル用にする
    def add_rank(df_rank: pd.DataFrame) -> pd.DataFrame:
        df_rank = df_rank.copy().reset_index(drop=True)
        df_rank.insert(0, "rank", range(1, len(df_rank) + 1))
        cols = [
            "rank",
            "TRAIN_DATASET",
            "VALID_DATASET",
            id_col,
            "INPUT_MODE",
            "IMG_SIZE",
            "COLOR_MODE",
            "VALID_BLUR",
            "VALID_BLUR_KERNEL",
            "mean",
            "median",
            "max",
            "iqr",
            "count",
        ]
        cols = [c for c in cols if c in df_rank.columns]
        return df_rank[cols]

    topK_by_max_table = add_rank(topK_by_max)
    topK_by_mean_table = add_rank(topK_by_mean)
    topK_by_median_table = add_rank(topK_by_median)
    topK_by_iqr_table = add_rank(topK_by_iqr)

    # -------------------------------------------------
    # 4) TOPK テーブル PNG 出力
    # -------------------------------------------------
    out_topK_max_table = out_dir / f"{prefix.name}_{metric_col}_top{topK}_by_max_table.png"
    make_table_png(
        topK_by_max_table,
        out_topK_max_table,
        title=f"Top {topK} Experiments (sorted by MAX {metric_col}, lower is better)",
    )

    out_topK_mean_table = out_dir / f"{prefix.name}_{metric_col}_top{topK}_by_mean_table.png"
    make_table_png(
        topK_by_mean_table,
        out_topK_mean_table,
        title=f"Top {topK} Experiments (sorted by MEAN {metric_col}, lower is better)",
    )

    out_topK_median_table = out_dir / f"{prefix.name}_{metric_col}_top{topK}_by_median_table.png"
    make_table_png(
        topK_by_median_table,
        out_topK_median_table,
        title=f"Top {topK} Experiments (sorted by MEDIAN {metric_col}, lower is better)",
    )

    out_topK_iqr_table = out_dir / f"{prefix.name}_{metric_col}_top{topK}_by_iqr_table.png"
    make_table_png(
        topK_by_iqr_table,
        out_topK_iqr_table,
        title=f"Top {topK} Experiments (sorted by IQR {metric_col}, smaller = more stable)",
    )

    # -------------------------------------------------
    # 5) 箱ひげ図：全実験（mean 昇順で並べる）
    # -------------------------------------------------
    all_for_box = exp_stats.sort_values("mean", ascending=True)
    out_all_box = out_dir / f"{prefix.name}_{metric_col}_all_experiments_boxplot.png"
    make_exp_boxplot(
        df=df,
        exp_df=all_for_box,
        id_col=id_col,
        metric_col=metric_col,
        out_path=out_all_box,
        title=f"All Experiments (sorted by MEAN {metric_col})",
        ylabel=f"{metric_col} (lower is better)",
    )

    # -------------------------------------------------
    # 6) 箱ひげ図：各指標で TOPK
    # -------------------------------------------------
    # max
    out_topK_max_box = out_dir / f"{prefix.name}_{metric_col}_top{topK}_by_max_boxplot.png"
    make_exp_boxplot(
        df=df,
        exp_df=topK_by_max.sort_values("max", ascending=True),
        id_col=id_col,
        metric_col=metric_col,
        out_path=out_topK_max_box,
        title=f"Top {topK} Experiments (sorted by MAX {metric_col})",
        ylabel=f"{metric_col} (lower is better)",
    )

    # mean
    out_topK_mean_box = out_dir / f"{prefix.name}_{metric_col}_top{topK}_by_mean_boxplot.png"
    make_exp_boxplot(
        df=df,
        exp_df=topK_by_mean.sort_values("mean", ascending=True),
        id_col=id_col,
        metric_col=metric_col,
        out_path=out_topK_mean_box,
        title=f"Top {topK} Experiments (sorted by MEAN {metric_col})",
        ylabel=f"{metric_col} (lower is better)",
    )

    # median
    out_topK_median_box = out_dir / f"{prefix.name}_{metric_col}_top{topK}_by_median_boxplot.png"
    make_exp_boxplot(
        df=df,
        exp_df=topK_by_median.sort_values("median", ascending=True),
        id_col=id_col,
        metric_col=metric_col,
        out_path=out_topK_median_box,
        title=f"Top {topK} Experiments (sorted by MEDIAN {metric_col})",
        ylabel=f"{metric_col} (lower is better)",
    )

    # iqr
    out_topK_iqr_box = out_dir / f"{prefix.name}_{metric_col}_top{topK}_by_iqr_boxplot.png"
    make_exp_boxplot(
        df=df,
        exp_df=topK_by_iqr.sort_values("iqr", ascending=True),
        id_col=id_col,
        metric_col=metric_col,
        out_path=out_topK_iqr_box,
        title=f"Top {topK} Experiments (sorted by IQR {metric_col})",
        ylabel=f"{metric_col} (smaller = more stable)",
    )


if __name__ == "__main__":
    main()
