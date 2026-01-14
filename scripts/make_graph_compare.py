#!/usr/bin/env python3
"""
plot_roll_errors_perfile_and_overview.py
=======================================

各 CSV（例: rgb_sz224.csv, gray_sz112.csv, ...）ごとに個別解析を行い、
カルマンフィルタ適用前(CNN err_roll) と適用後(KF kf_err) を比較した図を生成する。

さらに、9通り（rgb/gray/bin4 × 56/112/224）すべてをまとめた
mean/max の棒グラフ（CNN/KF 各1枚）も作成する。

出力例:
  fig/
      rgb_sz224/
          scatter.png
          hist.png
          stats_bar.png
          stats_table.png
      gray_sz112/
          ...
      overview/
          cnn_stats_bar.png   ← 9条件CNN一覧
          kf_stats_bar.png    ← 9条件KF一覧
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 対象CSV名の正規表現
PATTERN = re.compile(r"^(rgb|gray|bin4)_sz(56|112|224)\.csv$", re.IGNORECASE)


ORDER = [
    ("rgb", 224),
    ("rgb", 112),
    ("rgb", 56),
    ("gray", 224),
    ("gray", 112),
    ("gray", 56),
    ("bin4", 224),
    ("bin4", 112),
    ("bin4", 56),
]


# ---------------------- CSV 読み込み ----------------------
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    req = {"filename", "true_roll", "err_roll", "kf_roll", "kf_err"}
    if not req.issubset(df.columns):
        raise SystemExit(f"[ERROR] {path} に必要な列 {req} が含まれていません。")

    return df


# ---------------------- 図作成：散布図 ----------------------
def make_scatter(df: pd.DataFrame, save_path: Path):
    plt.figure(figsize=(8, 6))
    plt.scatter(df["true_roll"], df["err_roll"], s=20, alpha=0.5, label="CNN Error")
    plt.scatter(df["true_roll"], df["kf_err"],  s=20, alpha=0.5, label="KF Error")
    plt.xlabel("true_roll [deg]")
    plt.ylabel("Error [deg]")
    plt.title("CNN vs KF Error")
    plt.grid(True, ls="--", lw=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print("[INFO] scatter →", save_path)


# ---------------------- 図作成：ヒストグラム ----------------------
def make_hist(df: pd.DataFrame, save_path: Path):
    plt.figure(figsize=(8, 6))

    bins = np.linspace(df[["err_roll", "kf_err"]].min().min(),
                       df[["err_roll", "kf_err"]].max().max(),
                       40)

    plt.hist(df["err_roll"], bins=bins, alpha=0.6,
             label="CNN Error", edgecolor="black", lw=0.3)
    plt.hist(df["kf_err"], bins=bins, alpha=0.6,
             label="KF Error", edgecolor="black", lw=0.3)

    plt.xlabel("Error [deg]")
    plt.ylabel("Frequency")
    plt.title("Histogram: CNN vs KF Error")
    plt.grid(True, ls="--", lw=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print("[INFO] hist →", save_path)


# ---------------------- 図作成：個別 stats bar ----------------------
def make_stats_bar(stats: dict, save_path: Path, label: str):
    x = np.arange(1)
    w = 0.35

    plt.figure(figsize=(6, 6))
    plt.bar(x - w/2, stats["cnn_mean"], w, label="CNN Mean")
    plt.bar(x + w/2, stats["kf_mean"],  w, label="KF Mean")

    plt.ylabel("Error [deg]")
    plt.title(f"Error Mean Comparison ({label})")
    plt.xticks([])
    plt.grid(axis="y", ls="--", lw=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print("[INFO] stats bar →", save_path)



# ---------------------- 図作成：個別 stats table ----------------------
def make_stats_table(stats: dict, save_path: Path):
    tbl = pd.DataFrame([stats]).round(3)

    fig, ax = plt.subplots(figsize=(4, 1.6))
    ax.axis("off")

    t = ax.table(
        cellText=tbl.values,
        colLabels=tbl.columns,
        cellLoc="center",
        loc="center"
    )

    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close()
    print("[INFO] stats table →", save_path)



# ---------------------- 図作成：9条件 CNN 一覧バー ----------------------
def plot_cnn_stats_overview(stats_df: pd.DataFrame, save_path: Path):
    df = stats_df.copy()

    # ★ ORDER に従ってソート
    df["sort_key"] = df.apply(lambda r: ORDER.index((r["channel"], r["size"])), axis=1)
    df = df.sort_values("sort_key")

    df["label"] = df["channel"].str.upper() + "-Sz" + df["size"].astype(str)

    x = np.arange(len(df))
    w = 0.35

    plt.figure(figsize=(12, 9))
    plt.bar(x - w/2, df["cnn_mean"], w, label="CNN Mean")
    plt.bar(x + w/2, df["cnn_max"],  w, label="CNN Max")

    plt.xticks(x, df["label"], rotation=45, ha="right")
    plt.ylabel("Error [deg]")
    plt.title("CNN Error (Pre-KF): Mean & Max (All 9 conditions)")
    plt.grid(axis="y", ls="--", lw=0.3)
    # plt.legend()
    # plt.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.15))
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        framealpha=0.8
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print("[INFO] CNN overview bar →", save_path)



# ---------------------- 図作成：9条件 KF 一覧バー ----------------------
def plot_kf_stats_overview(stats_df: pd.DataFrame, save_path: Path):
    df = stats_df.copy()

    # ★ ORDER に従ってソート
    df["sort_key"] = df.apply(lambda r: ORDER.index((r["channel"], r["size"])), axis=1)
    df = df.sort_values("sort_key")

    df["label"] = df["channel"].str.upper() + "-Sz" + df["size"].astype(str)

    x = np.arange(len(df))
    w = 0.35

    plt.figure(figsize=(12, 9))
    plt.bar(x - w/2, df["kf_mean"], w, label="KF Mean")
    plt.bar(x + w/2, df["kf_max"],  w, label="KF Max")

    plt.xticks(x, df["label"], rotation=45, ha="right")
    plt.ylabel("Error [deg]")
    plt.title("KF Error (Post-KF): Mean & Max (All 9 conditions)")
    plt.grid(axis="y", ls="--", lw=0.3)
    # plt.legend()
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        framealpha=0.8
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print("[INFO] KF overview bar →", save_path)


def make_angle_plot(df: pd.DataFrame, save_path: Path, label: str):
    """true_roll, pred_roll（CNN）, kf_roll（KF）の角度比較グラフを生成"""
    plt.figure(figsize=(10, 6))

    # 真値
    plt.plot(df["true_roll"], label="True", color="black", linewidth=2)

    # CNN（KFなし）
    plt.plot(df["pred_roll"], label="CNN (No KF)", linestyle="--", color="blue", linewidth=1.5)

    # KFあり
    plt.plot(df["kf_roll"], label="KF Applied", linestyle=":", color="red", linewidth=1.5)

    plt.xlabel("Index (image order)")
    plt.ylabel("Roll Angle [deg]")
    plt.title(f"Roll Angle Comparison ({label})")
    plt.grid(True, ls="--", lw=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print("[INFO] angle plot →", save_path)


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser(description="Per-file CNN/KF roll error comparison + overview")
    ap.add_argument("--dir", required=True, help="CSV フォルダ")
    args = ap.parse_args()

    folder = Path(args.dir).resolve()
    fig_root = folder / "fig"
    fig_root.mkdir(exist_ok=True)

    csv_files = [p for p in folder.glob("*.csv") if PATTERN.match(p.name)]
    if len(csv_files) == 0:
        raise SystemExit("[ERROR] 対象 CSV がありません。")

    all_stats_list = []

    for csv_path in csv_files:
        ch, sz = PATTERN.match(csv_path.name).groups()
        label = f"{ch}_sz{sz}"

        print(f"\n===== Processing {label} =====")

        # 出力フォルダ
        out_dir = fig_root / label
        out_dir.mkdir(exist_ok=True)

        # CSV 読み込み
        df = load_csv(csv_path)

        # 個別統計
        stats = {
            "channel": ch,
            "size": int(sz),
            "cnn_mean": df["err_roll"].mean(),
            "cnn_max":  df["err_roll"].max(),
            "kf_mean":  df["kf_err"].mean(),
            "kf_max":   df["kf_err"].max(),
        }

        all_stats_list.append(stats)

        # --- 図の生成 ---
        make_scatter(df, out_dir / "scatter.png")
        make_hist(df, out_dir / "hist.png")
        make_stats_bar(stats, out_dir / "stats_bar.png", label)
        make_stats_table(stats, out_dir / "stats_table.png")
        make_angle_plot(df, out_dir / "angle_plot.png", label)


        # 個別 CSV も保存（必要なら）
        df.to_csv(out_dir / f"{label}_combined.csv", index=False)

        print(f"[INFO] Completed {label}")

    # =========================
    # 9 条件まとめ棒グラフ作成
    # =========================
    overview_dir = fig_root / "overview"
    overview_dir.mkdir(exist_ok=True)

    stats_df = pd.DataFrame(all_stats_list)

    plot_cnn_stats_overview(stats_df, overview_dir / "cnn_stats_bar.png")
    plot_kf_stats_overview(stats_df, overview_dir / "kf_stats_bar.png")

    print("\n[INFO] 全 CSV の処理完了")




if __name__ == "__main__":
    main()
