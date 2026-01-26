#!/usr/bin/env python3
"""
make_err_plots_from_logs_multi_csv.py
--------------------------------
- 2つ（以上も可）の「ログCSV」を指定して、以下の図を作成するスクリプト。

【前提とするCSV形式】
  filename,true_roll_deg,pred_roll_deg,err_deg,abs_err_deg
  0000.png,0.0,57.66,57.66,57.66
  ...

  - err_deg, abs_err_deg が無い場合は以下のように自動計算する:
      err_deg     = pred_roll_deg - true_roll_deg
      abs_err_deg = |err_deg|

【各CSVごとに出力する図】
  1) 散布図: True Roll vs Error (signed)
       → true_roll_deg を横軸、err_deg を縦軸
  2) ヒストグラム: Absolute Error
       → abs_err_deg を 0〜180 [deg] で集計

【すべてのCSVをまとめた全体図】
  3) 散布図: True Roll vs Error (signed, 全CSV)
       → 各CSVを scatter のみで描画（線で結ばない）
  4) ヒストグラム: Absolute Error (全CSV)
       → CSVが2つの場合、左y軸と右y軸を分けて描画し、
         それぞれ色を分ける

【出力先】
  - --out_dir を指定した場合: そこをルートに保存
  - 指定しない場合: 最初に指定したCSVの親ディレクトリに figs_logs/ を作成して保存

  例:
    python make_err_plots_from_logs_multi_csv.py \\
        --csvs logs/exp001.csv logs/exp002.csv \\
        --out_dir logs/figs_logs
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib; matplotlib.use("Agg")  # GUIなし環境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 引数
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="複数のログCSVから散布図・ヒストグラムを作成するスクリプト"
    )
    p.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        help="評価結果 CSV のパスを複数指定 (例: logs/exp001.csv logs/exp002.csv ...)",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="図の出力先ディレクトリ (指定しない場合は最初のCSVの親ディレクトリに figs_logs/ を作成)",
    )
    return p.parse_args()


# ============================================================
# ユーティリティ
# ============================================================
def choose_tick_step(y_max: int) -> int:
    """
    y_max に応じて「いい感じ」の目盛り間隔を返す。
    """
    if y_max <= 10:
        return 1
    if y_max <= 20:
        return 2
    if y_max <= 50:
        return 5
    if y_max <= 100:
        return 10
    if y_max <= 200:
        return 20
    return 50


# ============================================================
# CSV 読み込み & 前処理
# ============================================================
def load_and_prepare_df(csv_path: Path, exp_name: str) -> pd.DataFrame:
    """
    CSV を読み込み、err_deg / abs_err_deg が無ければ計算して付与し、
    さらに exp_name 列を追加して返す。
    """
    df = pd.read_csv(csv_path)

    # 必須カラムチェック
    required_cols = ["true_roll_deg", "pred_roll_deg"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{csv_path}: 必須カラム '{c}' が存在しません。")

    # err_deg が無ければ pred - true から計算
    if "err_deg" not in df.columns:
        df["err_deg"] = df["pred_roll_deg"] - df["true_roll_deg"]

    # abs_err_deg が無ければ |err_deg| から計算
    if "abs_err_deg" not in df.columns:
        df["abs_err_deg"] = df["err_deg"].abs()

    # exp_name 列を追加（どのCSV由来かを識別するため）
    df["exp_name"] = exp_name

    return df


# ============================================================
# 個別 CSV 用プロット
# ============================================================
def make_plots_for_one_csv(df_exp: pd.DataFrame, exp_name: str, out_dir: Path):
    """
    1つの CSV (1つの exp 相当) のみを含む DataFrame を対象に、
    散布図 + ヒストグラムを作成する。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- ① 散布図: True Roll vs Error (signed) --------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df_exp["true_roll_deg"], df_exp["err_deg"], s=12, alpha=0.6)
    ax.set_title(f"{exp_name}  True Roll vs Error (signed)")
    ax.set_xlabel("True Roll [deg]")
    ax.set_ylabel("Error [deg]")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_true_vs_err.png", dpi=150)
    plt.close(fig)

    # -------- ② ヒストグラム: Absolute Error --------
    # 0〜180度の範囲で集計（必要に応じて変更可）
    bins = np.arange(0, 181, 10)

    fig, ax = plt.subplots(figsize=(7, 5))
    Y, _bin_edges, _ = ax.hist(
        df_exp["abs_err_deg"],
        bins=bins,
        range=(0, 180),
        alpha=0.7,
        edgecolor="black",
        linewidth=1.0,
    )
    ax.set_title(f"{exp_name}  Absolute Error Histogram")
    ax.set_xlabel("Error [deg]")
    ax.set_ylabel("Count")
    ax.set_xticks(np.arange(0, 181, 20))

    y_max = int(max(Y)) + 1 if len(Y) > 0 else 1
    step = choose_tick_step(y_max)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, step))

    fig.tight_layout()
    fig.savefig(out_dir / "hist_abs_err.png", dpi=150)
    plt.close(fig)


# ============================================================
# 全 CSV をまとめたプロット
# ============================================================
def make_plots_all_csvs(df_all: pd.DataFrame, out_dir: Path):
    """
    全ての CSV を1つにまとめた DataFrame df_all から、
    - 散布図 (True vs Error, CSVごとに scatter)
    - ヒストグラム (Absolute Error, 全CSV)
      → CSVが2つの場合は左右別y軸、それ以外は共通y軸
    を作成する。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # グループ分け (exp_name ごと)
    if "exp_name" not in df_all.columns:
        groups = {"all": df_all}
    else:
        groups = {name: g for name, g in df_all.groupby("exp_name")}

    # -------- (全体) 散布図: True Roll vs Error (signed) --------
    fig, ax = plt.subplots(figsize=(7, 5))
    handles = []
    labels = []

    for name, g in groups.items():
        g_sorted = g.sort_values("true_roll_deg")
        h = ax.scatter(
            g_sorted["true_roll_deg"].values,
            g_sorted["err_deg"].values,
            s=12,
            alpha=0.7,
            label=name,
        )
        handles.append(h)
        labels.append(name)

    ax.set_title("True Roll vs Error (signed) - ALL CSVs")
    ax.set_xlabel("Theta [deg]")
    ax.set_ylabel("Error [deg]")
    if handles:
        ax.legend(handles, labels, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_true_vs_err_all.png", dpi=150)
    plt.close(fig)

    # -------- (全体) ヒストグラム: Absolute Error --------
    bins = np.arange(0, 181, 10)
    group_items = list(groups.items())

    # CSV が 2つのときは左右で縦軸を分ける（色も分ける）
    if len(group_items) == 2:
        (name1, g1), (name2, g2) = group_items

        fig, ax1 = plt.subplots(figsize=(7, 5))

        # 左側 (ax1) に1つ目
        Y1, _bin_edges1, _ = ax1.hist(
            g1["abs_err_deg"],
            bins=bins,
            range=(0, 180),
            alpha=0.5,
            edgecolor="black",
            linewidth=0.8,
            label=name1,
            color="tab:blue",
        )
        ax1.set_xlabel("Error [deg]")
        ax1.set_ylabel(f"Count ({name1})", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_xticks(np.arange(0, 181, 20))

        y_max1 = int(max(Y1)) + 1 if len(Y1) > 0 else 1
        step1 = choose_tick_step(y_max1)
        ax1.set_ylim(0, y_max1)
        ax1.set_yticks(np.arange(0, y_max1 + 1, step1))

        # 右側 (ax2) に2つ目
        ax2 = ax1.twinx()
        Y2, _bin_edges2, _ = ax2.hist(
            g2["abs_err_deg"],
            bins=bins,
            range=(0, 180),
            alpha=0.5,
            edgecolor="black",
            linewidth=0.8,
            label=name2,
            color="tab:orange",
        )
        ax2.set_ylabel(f"Count ({name2})", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        y_max2 = int(max(Y2)) + 1 if len(Y2) > 0 else 1
        step2 = choose_tick_step(y_max2)
        ax2.set_ylim(0, y_max2)
        ax2.set_yticks(np.arange(0, y_max2 + 1, step2))

        # 凡例は左右で分ける
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1:
            ax1.legend(h1, l1, loc="upper left", fontsize=8)
        if h2:
            ax2.legend(h2, l2, loc="upper right", fontsize=8)

        ax1.set_title("Absolute Error Histogram - ALL CSVs (left/right axes)")
        fig.tight_layout()
        fig.savefig(out_dir / "hist_abs_err_all_dual_axis.png", dpi=150)
        plt.close(fig)

    else:
        # 2つ以外のときは共通y軸で重ね書き（色は自動で分かれる）
        fig, ax = plt.subplots(figsize=(7, 5))
        y_max_all = 0

        for name, g in group_items:
            Y, _bin_edges, _ = ax.hist(
                g["abs_err_deg"],
                bins=bins,
                range=(0, 180),
                alpha=0.4,
                edgecolor="black",
                linewidth=0.7,
                label=name,
            )
            if len(Y) > 0:
                y_max_all = max(y_max_all, int(max(Y)))

        ax.set_title("Absolute Error Histogram - ALL CSVs")
        ax.set_xlabel("Absolute Error [deg]")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.set_xticks(np.arange(0, 181, 20))

        y_max = y_max_all + 1 if y_max_all > 0 else 1
        step = choose_tick_step(y_max)
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.arange(0, y_max + 1, step))

        fig.tight_layout()
        fig.savefig(out_dir / "hist_abs_err_all.png", dpi=150)
        plt.close(fig)


# ============================================================
# main
# ============================================================
def main():
    args = parse_args()
    csv_paths = [Path(p) for p in args.csvs]

    # 出力先ルート
    if args.out_dir is None:
        first_parent = csv_paths[0].parent
        figs_root = first_parent / "figs_logs"
    else:
        figs_root = Path(args.out_dir)
    figs_root.mkdir(parents=True, exist_ok=True)

    print("CSV files:")
    for p in csv_paths:
        print("  -", p)

    # 全CSVを読み込み & 結合
    df_all_list = []

    for csv_path in csv_paths:
        if not csv_path.exists():
            print("WARNING: CSV が見つかりません:", csv_path)
            continue

        exp_name = csv_path.stem  # ファイル名(拡張子除く)をラベルとして使う
        print(f"Read CSV: {csv_path}  (exp_name = {exp_name})")
        df_exp = load_and_prepare_df(csv_path, exp_name)

        # 個別図
        exp_dir = figs_root / exp_name
        print(f"  Make plots for {exp_name} -> {exp_dir}")
        make_plots_for_one_csv(df_exp, exp_name, exp_dir)

        df_all_list.append(df_exp)

    if not df_all_list:
        print("有効なCSVがありませんでした。終了します。")
        return

    df_all = pd.concat(df_all_list, axis=0, ignore_index=True)

    # 全CSV統合の図
    print("  Make ALL-CSV plots ->", figs_root)
    make_plots_all_csvs(df_all, figs_root)

    print("Done.")


if __name__ == "__main__":
    main()
