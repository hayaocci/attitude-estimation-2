#!/usr/bin/env python3
"""
plot_roll_errors.py
===================
生成物
  /fig/
      roll_error_scatter.png       … 3×3 散布図
      roll_error_scatter_all.png   … 全データ散布図
      roll_error_stats.png         … 平均・最大バー図
      roll_error_hist.png          … 3×3 ヒストグラム
  combined_roll_errors.csv         … 9 CSV 結合版
  err_roll_stats_table.csv         … ★NEW  平均・最大値の一覧表
"""

from __future__ import annotations
import argparse, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATTERN = re.compile(r"^(rgb|gray|bin4)_sz(56|112|224)\.csv$", re.IGNORECASE)
CHANNELS = ["rgb", "gray", "bin4"]
SIZES = [56, 112, 224]


# ────────────────────── 入力処理 ─────────────────────────
def find_csv_files(folder: Path) -> list[Path]:
    csvs = [p for p in folder.glob("*.csv") if PATTERN.match(p.name)]
    if len(csvs) != 9:
        raise SystemExit(f"[ERROR] {folder} に必要な 9 CSV が {len(csvs)} 件しかありません。")
    return csvs


def load_and_concat(paths: list[Path]) -> pd.DataFrame:
    dfs, ref = [], None
    for p in paths:
        ch, sz = PATTERN.match(p.name).groups()
        sz = int(sz)

        df = pd.read_csv(p)
        req = {"filename", "true_roll", "err_roll"}
        if not req.issubset(df.columns):
            raise SystemExit(f"[ERROR] {p} に列 {req} がありません。")

        if ref is None:
            ref = df["filename"].tolist()
        elif df["filename"].tolist() != ref:
            raise SystemExit(f"[ERROR] {p} の filename 順序が他ファイルと異なります。")

        df = df.loc[:, ["filename", "true_roll", "err_roll"]].copy()
        df["channel"] = ch.lower()
        df["size"] = sz
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ────────────────────── 可視化 ───────────────────────────
def scatter_grid(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
    for i, ch in enumerate(CHANNELS):
        for j, sz in enumerate(SIZES):
            ax = axes[i, j]
            sub = df[(df["channel"] == ch) & (df["size"] == sz)]
            ax.scatter(sub["true_roll"], sub["err_roll"], s=10, alpha=0.6, lw=0)
            ax.set_title(f"{ch.upper()}  Sz{sz}", fontsize=10)
            if i == 2:
                ax.set_xlabel("true theta [deg]")
            if j == 0:
                ax.set_ylabel("err theta [deg]")
            ax.grid(True, ls="--", lw=0.3)
    fig.suptitle("True Roll vs. Error Roll (3×3)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=200)
    print(f"[INFO] Scatter grid → {out_png}")


def scatter_all(df: pd.DataFrame, out_png: Path) -> None:
    color_map = {"rgb": "tab:red", "gray": "tab:green", "bin4": "tab:blue"}
    marker_map = {56: "o", 112: "^", 224: "s"}

    fig, ax = plt.subplots(figsize=(10, 7))
    for ch in CHANNELS:
        for sz in SIZES:
            sub = df[(df["channel"] == ch) & (df["size"] == sz)]
            ax.scatter(
                sub["true_roll"], sub["err_roll"],
                s=25, alpha=0.6, lw=0,
                c=color_map[ch],
                marker=marker_map[sz],
                label=f"{ch.upper()} Sz{sz}"
            )
    ax.set_xlabel("true theta [deg]")
    ax.set_ylabel("err theta [deg]")
    ax.set_title("True Roll vs. Error Roll (All Data)")
    ax.grid(True, ls="--", lw=0.3)
    ax.legend(ncol=3, fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(out_png, dpi=200)
    print(f"[INFO] Scatter all  → {out_png}")


def stats_bar(stats: pd.DataFrame, out_png: Path) -> None:
    stats = stats.copy()  # 引数を書き換えない
    stats["label"] = stats["channel"].str.upper() + "-" + stats["size"].astype(str)
    x = np.arange(len(stats)); w = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x-w/2, stats["mean"], w, label="Mean |err|")
    ax.bar(x+w/2, stats["max"],  w, label="Max |err|")
    ax.set_xticks(x); ax.set_xticklabels(stats["label"], rotation=45, ha="right")
    ax.set_ylabel("err_theta [deg]")
    # ax.set_title("Mean & Max err_roll per Channel / Size")
    ax.legend(); ax.grid(axis="y", ls="--", lw=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=200)
    print(f"[INFO] Stats bar    → {out_png}")


def hist_grid(df: pd.DataFrame, out_png: Path) -> None:
    glob_min, glob_max = df["err_roll"].min(), df["err_roll"].max()
    bins = np.linspace(glob_min, glob_max, 31)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
    for i, ch in enumerate(CHANNELS):
        for j, sz in enumerate(SIZES):
            ax = axes[i, j]
            sub = df[(df["channel"] == ch) & (df["size"] == sz)]
            ax.hist(sub["err_roll"], bins=bins, range=(glob_min, glob_max),
                    alpha=0.75, edgecolor="black", lw=0.3)
            ax.set_title(f"{ch.upper()}  Sz{sz}", fontsize=10)
            if i == 2:
                ax.set_xlabel("err_roll [deg]")
            if j == 0:
                ax.set_ylabel("Frequency")
            ax.grid(True, ls="--", lw=0.3)
    for ax in axes.ravel():
        ax.set_xlim(glob_min, glob_max)

    fig.suptitle("err_roll Histogram (common x-axis)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=200)
    print(f"[INFO] Histogram    → {out_png}")

# ★ NEW ────────────────────────────────────────────────
def stats_table_plot(stats: pd.DataFrame, out_png: Path) -> None:
    """
    channel×size の平均値・最大値を『数値だけ』で表示する PNG を出力。
    matplotlib の table 機能で見やすい表にする。
    """
    # 全列を丸めて見やすく
    tbl = stats.copy()
    tbl["mean"] = tbl["mean"].round(2)
    tbl["max"]  = tbl["max"].round(2)

    fig, ax = plt.subplots(figsize=(6, 2.5 + 0.2*len(tbl)))
    ax.axis("off")  # 軸は非表示

    table = ax.table(
        cellText   = tbl.values,
        colLabels  = tbl.columns,
        cellLoc    = "center",
        loc        = "center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)  # 行間を少し拡張

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"[INFO] Stats table PNG → {out_png}")


# ────────────────────── メイン ─────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Plot roll-error graphs from 9 CSVs")
    ap.add_argument("--dir", required=True, help="CSV が 9 枚あるフォルダを指定")
    ap.add_argument("--out_csv", default="combined_roll_errors.csv",
                    help="結合 CSV のファイル名")
    ap.add_argument("--out_stats_csv", default="err_roll_stats_table.csv",
                    help="平均・最大値テーブルの CSV 名")  # ★NEW
    args = ap.parse_args()

    folder = Path(args.dir).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"[ERROR] {folder} は存在しないかディレクトリではありません。")

    fig_dir = folder / "fig"; fig_dir.mkdir(exist_ok=True)

    # ---------- データ読み込み ----------
    df = load_and_concat(find_csv_files(folder))

    # ---------- 結合 CSV ----------
    (folder / args.out_csv).write_text(df.to_csv(index=False))
    print(f"[INFO] Combined CSV → {folder/args.out_csv}")

    # ---------- 統計値 DataFrame ----------
    stats = (
        df.groupby(["channel", "size"])["err_roll"]
        .agg(mean="mean", max="max")
        .reset_index()
        .sort_values(["channel", "size"])
    )

    # ---------- 統計値 CSV ★NEW ----------
    stats.to_csv(folder / args.out_stats_csv, index=False)
    print(f"[INFO] Stats table  → {folder/args.out_stats_csv}")

    # ---------- グラフ出力 ----------
    scatter_grid(df, fig_dir / "roll_error_scatter.png")
    scatter_all(df,  fig_dir / "roll_error_scatter_all.png")
    stats_bar(stats,  fig_dir / "roll_error_stats.png")
    hist_grid(df,     fig_dir / "roll_error_hist.png")
    stats_table_plot(stats, fig_dir / "roll_error_stats_table.png")   # ★ NEW


if __name__ == "__main__":
    main()
