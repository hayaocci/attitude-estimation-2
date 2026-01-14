#!/usr/bin/env python3
"""
plot_roll_errors_flexible.py
============================

1～9 CSV に柔軟対応した Roll Error 可視化スクリプト
生成物:
  /fig/
      roll_error_scatter.png       … グリッド散布図
      roll_error_scatter_all.png   … 全データ散布図
      roll_error_stats.png         … 平均・最大バー図
      roll_error_hist.png          … ヒストグラム
      roll_error_stats_table.png   … 平均・最大値表
  combined_roll_errors.csv         … 結合 CSV
  kf_err_stats_table.csv         … 平均・最大値 CSV
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATTERN = re.compile(r"^(rgb|gray|bin4)_sz(56|112|224)\.csv$", re.IGNORECASE)
CHANNELS = ["rgb", "gray", "bin4"]
SIZES = [56, 112, 224]

# ────────────────────── 入力処理 ─────────────────────────
def find_csv_files(folder: Path) -> list[Path]:
    csvs = [p for p in folder.glob("*.csv") if PATTERN.match(p.name)]
    if len(csvs) == 0:
        raise SystemExit(f"[ERROR] {folder} に対象の CSV が存在しません。")
    return csvs

def load_and_concat(paths: list[Path]) -> pd.DataFrame:
    dfs, ref = [], None
    for p in paths:
        ch, sz = PATTERN.match(p.name).groups()
        sz = int(sz)

        df = pd.read_csv(p)
        req = {"filename", "kf_roll", "kf_err"}
        if not req.issubset(df.columns):
            raise SystemExit(f"[ERROR] {p} に列 {req} がありません。")

        if ref is None:
            ref = df["filename"].tolist()
        elif df["filename"].tolist() != ref:
            raise SystemExit(f"[ERROR] {p} の filename 順序が他ファイルと異なります。")

        df = df.loc[:, ["filename", "kf_roll", "kf_err"]].copy()
        df["channel"] = ch.lower()
        df["size"] = sz
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

# ────────────────────── 可視化関数 ───────────────────────────
def scatter_grid(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
    for i, ch in enumerate(CHANNELS):
        for j, sz in enumerate(SIZES):
            ax = axes[i, j]
            sub = df[(df["channel"] == ch) & (df["size"] == sz)]
            if not sub.empty:
                ax.scatter(sub["kf_roll"], sub["kf_err"], s=10, alpha=0.6, lw=0)
            ax.set_title(f"{ch.upper()}  Sz{sz}", fontsize=10)
            if i == 2:
                ax.set_xlabel("kf_roll [deg]")
            if j == 0:
                ax.set_ylabel("kf_err [deg]")
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
            if not sub.empty:
                ax.scatter(
                    sub["kf_roll"], sub["kf_err"],
                    s=25, alpha=0.6, lw=0,
                    c=color_map[ch],
                    marker=marker_map[sz],
                    label=f"{ch.upper()} Sz{sz}"
                )
    ax.set_xlabel("kf_roll [deg]")
    ax.set_ylabel("kf_err [deg]")
    ax.set_title("True Roll vs. Error Roll (All Data)")
    ax.grid(True, ls="--", lw=0.3)
    ax.legend(ncol=3, fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(out_png, dpi=200)
    print(f"[INFO] Scatter all  → {out_png}")

def stats_bar(stats: pd.DataFrame, out_png: Path) -> None:
    stats = stats.copy()
    stats["label"] = stats["channel"].str.upper() + "-" + stats["size"].astype(str)
    x = np.arange(len(stats)); w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x-w/2, stats["mean"], w, label="Mean |err|")
    ax.bar(x+w/2, stats["max"],  w, label="Max |err|")
    ax.set_xticks(x); ax.set_xticklabels(stats["label"], rotation=45, ha="right")
    ax.set_ylabel("kf_err [deg]")
    ax.set_title("Mean & Max kf_err per Channel / Size")
    ax.legend(); ax.grid(axis="y", ls="--", lw=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=200)
    print(f"[INFO] Stats bar    → {out_png}")

def hist_grid(df: pd.DataFrame, out_png: Path) -> None:
    if df["kf_err"].empty:
        print(f"[WARN] Histogram skipped: no data")
        return
    glob_min, glob_max = df["kf_err"].min(), df["kf_err"].max()
    bins = np.linspace(glob_min, glob_max, 31)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
    for i, ch in enumerate(CHANNELS):
        for j, sz in enumerate(SIZES):
            ax = axes[i, j]
            sub = df[(df["channel"] == ch) & (df["size"] == sz)]
            if not sub.empty:
                ax.hist(sub["kf_err"], bins=bins, range=(glob_min, glob_max),
                        alpha=0.75, edgecolor="black", lw=0.3)
            ax.set_title(f"{ch.upper()}  Sz{sz}", fontsize=10)
            if i == 2:
                ax.set_xlabel("kf_err [deg]")
            if j == 0:
                ax.set_ylabel("Frequency")
            ax.grid(True, ls="--", lw=0.3)
    for ax in axes.ravel():
        ax.set_xlim(glob_min, glob_max)

    fig.suptitle("kf_err Histogram (common x-axis)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=200)
    print(f"[INFO] Histogram    → {out_png}")

def stats_table_plot(stats: pd.DataFrame, out_png: Path) -> None:
    tbl = stats.copy()
    tbl["mean"] = tbl["mean"].round(2)
    tbl["max"]  = tbl["max"].round(2)

    fig, ax = plt.subplots(figsize=(6, 2.5 + 0.2*len(tbl)))
    ax.axis("off")

    table = ax.table(
        cellText   = tbl.values,
        colLabels  = tbl.columns,
        cellLoc    = "center",
        loc        = "center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"[INFO] Stats table PNG → {out_png}")

# ────────────────────── main ─────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Plot roll-error graphs from CSVs (1～9 CSV対応)")
    ap.add_argument("--dir", required=True, help="CSV が入ったフォルダを指定")
    ap.add_argument("--out_csv", default="combined_roll_errors.csv",
                    help="結合 CSV のファイル名")
    ap.add_argument("--out_stats_csv", default="kf_err_stats_table.csv",
                    help="平均・最大値テーブルの CSV 名")
    args = ap.parse_args()

    folder = Path(args.dir).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"[ERROR] {folder} は存在しないかディレクトリではありません。")

    fig_dir = folder / "fig"; fig_dir.mkdir(exist_ok=True)

    # ---------- データ読み込み ----------
    csv_paths = find_csv_files(folder)
    df = load_and_concat(csv_paths)

    # ---------- 結合 CSV ----------
    (folder / args.out_csv).write_text(df.to_csv(index=False))
    print(f"[INFO] Combined CSV → {folder/args.out_csv}")

    # ---------- 統計値 DataFrame ----------
    stats = (
        df.groupby(["channel", "size"])["kf_err"]
        .agg(mean="mean", max="max")
        .reset_index()
        .sort_values(["channel", "size"])
    )

    # ---------- 統計値 CSV ----------
    stats.to_csv(folder / args.out_stats_csv, index=False)
    print(f"[INFO] Stats table  → {folder/args.out_stats_csv}")

    # ---------- グラフ出力 ----------
    scatter_grid(df, fig_dir / "roll_error_scatter.png")
    scatter_all(df,  fig_dir / "roll_error_scatter_all.png")
    stats_bar(stats,  fig_dir / "roll_error_stats.png")
    hist_grid(df,     fig_dir / "roll_error_hist.png")
    stats_table_plot(stats, fig_dir / "roll_error_stats_table.png")

if __name__ == "__main__":
    main()
