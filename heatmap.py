#!/usr/bin/env python3
"""
analyze_preprocess_experiments_inputmode_v3.py

1つのcsv(全expまとめ)を読み込み、

- expごとの評価指標サマリを作成
- heatmap（INPUT_MODE×IMG_SIZE, INPUT_MODE×BLUR_MODE, IMG_SIZE×BLUR_MODE）
    * 各 heatmap を複数指標で出力:
        - mae_deg              : 平均絶対誤差
        - std_deg              : 標準偏差
        - p95_deg              : 95パーセンタイル誤差
        - max_err_deg          : 最大誤差
        - median_deg           : 中央値 |err|
        - center_width90_deg   : 中心90%区間幅（p95 - p5）
        - trimmed_mae95_deg    : 上位5%を除いたMAE
        - iqr_deg              : 四分位範囲（p75 - p25）
      → 指標ごとにサブフォルダを分けて保存
- Top/Bottom散布図（mae_vs_p95）
- 付録用サマリ表(csv)

出力先:
    ・--out_dir を指定しなければ、
      入力csvと同じ階層に <csv_stem>_analysis/ フォルダを作成して保存

使い方:
    python analyze_preprocess_experiments_inputmode_v3.py \
        --csv /path/to/all_experiments.csv

    # 出力先を明示したい場合
    python analyze_preprocess_experiments_inputmode_v3.py \
        --csv /path/to/all_experiments.csv \
        --out_dir /path/to/custom_out_dir
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------
# 引数
# --------------------------------------------------------
def parse_args():
    pa = argparse.ArgumentParser(
        description="前処理27通りの評価結果を可視化（heatmap, scatter, 表）するスクリプト（INPUT_MODE版 v3）"
    )
    pa.add_argument(
        "--csv",
        required=True,
        help="全expが含まれている評価csvのパス",
    )
    pa.add_argument(
        "--out_dir",
        default=None,
        help=(
            "図やサマリcsvを出力するディレクトリ。"
            "未指定の場合は、csvと同じ階層に <csv_stem>_analysis/ を作成して出力"
        ),
    )
    pa.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top/Bottom散布図でハイライトする上位/下位の件数（デフォルト5）",
    )
    return pa.parse_args()


# --------------------------------------------------------
# True/False or 文字列 → bool 変換
# --------------------------------------------------------
def to_bool_like(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "1", "t", "yes", "y"):
        return True
    return False


# --------------------------------------------------------
# exp単位のサマリ作成（INPUT_MODE, IMG_SIZE, BLUR_MODE を含める）
# --------------------------------------------------------
def aggregate_by_exp(df: pd.DataFrame) -> pd.DataFrame:
    if "exp_id" not in df.columns:
        raise ValueError("csv に 'exp_id' 列がありません。")

    if "err_roll" not in df.columns:
        raise ValueError("csv に 'err_roll' 列がありません。")

    has_kf = "kf_err" in df.columns

    rows = []
    for exp_id, g in df.groupby("exp_id"):
        # 入力モード（rgb / gray / bin4）
        input_mode = g["INPUT_MODE"].iloc[0] if "INPUT_MODE" in g.columns else "unknown"

        # 画像サイズ
        img_size = int(g["IMG_SIZE"].iloc[0]) if "IMG_SIZE" in g.columns else -1

        # BLUR_MODE を作るための情報
        if "TRAIN_BLUR" in g.columns:
            train_blur_raw = g["TRAIN_BLUR"].iloc[0]
            train_blur = to_bool_like(train_blur_raw)
        else:
            train_blur = False

        if "TRAIN_BLUR_KERNEL" in g.columns:
            blur_kernel_raw = g["TRAIN_BLUR_KERNEL"].iloc[0]
        else:
            blur_kernel_raw = np.nan

        # BLUR_MODE のカテゴリ化
        #   - blurなし      → "none"
        #   - kernel <= 3   → "weak"
        #   - kernel >= 5等 → "strong"
        if not train_blur or pd.isna(blur_kernel_raw):
            blur_mode = "none"
        else:
            try:
                k = int(blur_kernel_raw)
            except Exception:
                k = None
            if k is None:
                blur_mode = "none"
            elif k <= 3:
                blur_mode = "weak"
            else:
                blur_mode = "strong"

        # err_roll ベースの指標
        abs_err = g["err_roll"].abs()

        # 基本指標
        mae_deg = float(abs_err.mean())
        std_deg = float(abs_err.std(ddof=0))
        p95 = float(abs_err.quantile(0.95))
        p95_deg = p95
        max_err_deg = float(abs_err.max())

        # (A) median(|err|)
        median_deg = float(abs_err.median())

        # (E) 中心区間の幅（ここでは中心90%: p95 - p5）
        p05 = float(abs_err.quantile(0.05))
        center_width90_deg = p95 - p05

        # trimmed MAE（上位5%除外）: |err| <= p95 の平均
        trimmed_mask = abs_err <= p95
        if trimmed_mask.any():
            trimmed_mae95_deg = float(abs_err[trimmed_mask].mean())
        else:
            trimmed_mae95_deg = np.nan

        # IQR（四分位範囲）: p75 - p25
        q25 = float(abs_err.quantile(0.25))
        q75 = float(abs_err.quantile(0.75))
        iqr_deg = q75 - q25

        # kf_err があれば同様に計算（なければ NaN）
        (
            mae_kf_deg,
            std_kf_deg,
            p95_kf_deg,
            max_kf_deg,
            median_kf_deg,
            center_width90_kf_deg,
            trimmed_mae95_kf_deg,
            iqr_kf_deg,
        ) = (np.nan,) * 8

        if has_kf:
            abs_kf = g["kf_err"].abs()

            mae_kf_deg = float(abs_kf.mean())
            std_kf_deg = float(abs_kf.std(ddof=0))
            p95_kf = float(abs_kf.quantile(0.95))
            p95_kf_deg = p95_kf
            max_kf_deg = float(abs_kf.max())

            median_kf_deg = float(abs_kf.median())
            p05_kf = float(abs_kf.quantile(0.05))
            center_width90_kf_deg = p95_kf - p05_kf

            trimmed_mask_kf = abs_kf <= p95_kf
            if trimmed_mask_kf.any():
                trimmed_mae95_kf_deg = float(abs_kf[trimmed_mask_kf].mean())
            else:
                trimmed_mae95_kf_deg = np.nan

            q25_kf = float(abs_kf.quantile(0.25))
            q75_kf = float(abs_kf.quantile(0.75))
            iqr_kf_deg = q75_kf - q25_kf

        rows.append(
            {
                "exp_id": exp_id,
                "INPUT_MODE": input_mode,
                "IMG_SIZE": img_size,
                "BLUR_MODE": blur_mode,
                # 元の指標
                "mae_deg": mae_deg,
                "std_deg": std_deg,
                "p95_deg": p95_deg,
                "max_err_deg": max_err_deg,
                # 追加指標（非KF）
                "median_deg": median_deg,
                "center_width90_deg": center_width90_deg,
                "trimmed_mae95_deg": trimmed_mae95_deg,
                "iqr_deg": iqr_deg,
                # KF指標
                "mae_kf_deg": mae_kf_deg,
                "std_kf_deg": std_kf_deg,
                "p95_kf_deg": p95_kf_deg,
                "max_kf_deg": max_kf_deg,
                # 追加指標（KF）
                "median_kf_deg": median_kf_deg,
                "center_width90_kf_deg": center_width90_kf_deg,
                "trimmed_mae95_kf_deg": trimmed_mae95_kf_deg,
                "iqr_kf_deg": iqr_kf_deg,
                "n_samples": len(g),
            }
        )

    summary_df = pd.DataFrame(rows)
    # MAEの小さい順に並べておく（Top/Bottomで使う）
    summary_df = summary_df.sort_values("mae_deg").reset_index(drop=True)
    return summary_df


# --------------------------------------------------------
# heatmap を描画（因子×因子）
# --------------------------------------------------------
def plot_heatmap_factor_factor(
    summary_df: pd.DataFrame,
    factor_row: str,
    factor_col: str,
    metric: str,
    out_path: Path,
    title: Optional[str] = None,
    row_order: Optional[List] = None,
    col_order: Optional[List] = None,
):
    """
    例:
      factor_row="INPUT_MODE", factor_col="IMG_SIZE", metric="mae_deg",
      row_order=["rgb", "gray", "bin4"], col_order=[56, 112, 224]
    """
    # Pivot table
    pivot = summary_df.pivot_table(
        index=factor_row,
        columns=factor_col,
        values=metric,
        aggfunc="mean",
    )

    # 行・列の順番を指定（orderが指定されていればそれに合わせる）
    if row_order is None:
        row_order = list(pivot.index)
    if col_order is None:
        col_order = list(pivot.columns)

    pivot = pivot.reindex(index=row_order, columns=col_order)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pivot.values, cmap="viridis")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticklabels([str(x) for x in pivot.index])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # セル内に値を表示
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                text = "-"
            else:
                # 単位がdeg前提なので小数1桁くらいが見やすい
                text = f"{val:.1f}"
            ax.text(j, i, text, ha="center", va="center", color="white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    if title is None:
        title = f"{metric} heatmap ({factor_row} × {factor_col})"
    ax.set_title(title)

    ax.set_xlabel(factor_col)
    ax.set_ylabel(factor_row)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Heatmap を保存: {out_path}")


# --------------------------------------------------------
# Top/Bottom 散布図
# --------------------------------------------------------
def plot_top_bottom_scatter(
    summary_df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    rank_metric: str,
    topk: int,
    out_path: Path,
    title: Optional[str] = None,
):
    """
    例:
      x_metric="mae_deg", y_metric="p95_deg", rank_metric="mae_deg"
    """
    df = summary_df.copy()

    top = df.nsmallest(topk, rank_metric)
    bottom = df.nlargest(topk, rank_metric)

    fig, ax = plt.subplots(figsize=(6, 5))

    # その他の点
    others_mask = ~df["exp_id"].isin(top["exp_id"]) & ~df["exp_id"].isin(bottom["exp_id"])
    others = df[others_mask]
    ax.scatter(
        others[x_metric],
        others[y_metric],
        s=20,
        c="lightgray",
        label="others",
    )

    # Top
    ax.scatter(
        top[x_metric],
        top[y_metric],
        s=60,
        c="tab:blue",
        label=f"Top {topk}",
        edgecolors="black",
    )
    for _, row in top.iterrows():
        ax.annotate(
            row["exp_id"],
            (row[x_metric], row[y_metric]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="tab:blue",
        )

    # Bottom
    ax.scatter(
        bottom[x_metric],
        bottom[y_metric],
        s=60,
        c="tab:red",
        label=f"Bottom {topk}",
        edgecolors="black",
    )
    for _, row in bottom.iterrows():
        ax.annotate(
            row["exp_id"],
            (row[x_metric], row[y_metric]),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=8,
            color="tab:red",
        )

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    if title is None:
        title = f"Top/Bottom scatter ({x_metric} vs {y_metric}, ranked by {rank_metric})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Top/Bottom散布図を保存: {out_path}")


# --------------------------------------------------------
# メイン
# --------------------------------------------------------
def main():
    args = parse_args()
    csv_path = Path(args.csv).resolve()

    # 出力先ディレクトリ:
    #   --out_dir 未指定なら、csvと同じ階層に <csv_stem>_analysis を作る
    if args.out_dir is None:
        out_dir = csv_path.parent / f"{csv_path.stem}_analysis"
    else:
        out_dir = Path(args.out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Output directory: {out_dir}")

    df = pd.read_csv(csv_path)

    # 1. expごとのサマリ
    summary_df = aggregate_by_exp(df)

    # 付録用の表（そのまま修論の表に使える）
    summary_csv_path = out_dir / "preprocess_summary_table.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[OK] サマリ表(csv)出力: {summary_csv_path}")

    # ----------------------------------------------------
    # 2. heatmap 3種類（INPUT×SIZE, INPUT×BLUR, SIZE×BLUR）
    #    各指標ごとにサブフォルダを分けて出力
    # ----------------------------------------------------

    # INPUT_MODE：rgb, gray, bin4 の順で並べたい
    input_order = [m for m in ["rgb", "gray", "bin4"] if m in summary_df["INPUT_MODE"].unique()]

    # IMG_SIZE：候補を 56,112,224 と想定（必要に応じてここを変える）
    size_candidates = [56, 112, 224]
    unique_sizes = list(summary_df["IMG_SIZE"].unique())
    size_order = [s for s in size_candidates if s in unique_sizes]
    # もし候補外のサイズがあれば一応後ろに付ける
    for s in unique_sizes:
        if s not in size_order:
            size_order.append(s)

    # BLUR_MODE：none, weak, strong の順
    blur_order = [b for b in ["none", "weak", "strong"] if b in summary_df["BLUR_MODE"].unique()]

    # heatmap に使う指標たち
    metrics_for_heatmap = [
        ("mae_deg", "mae"),
        ("std_deg", "std"),
        ("p95_deg", "p95"),
        ("max_err_deg", "max_err"),
        ("median_deg", "median"),
        ("center_width90_deg", "width90"),
        ("trimmed_mae95_deg", "tmae95"),
        ("iqr_deg", "iqr"),
    ]

    for metric, folder_name in metrics_for_heatmap:
        metric_dir = out_dir / folder_name
        metric_dir.mkdir(parents=True, exist_ok=True)

        mshort = folder_name  # フォルダ名をそのまま短縮表記にも使う

        # INPUT × SIZE
        heatmap_is = metric_dir / f"INPUT_MODE_x_IMG_SIZE_{mshort}.png"
        plot_heatmap_factor_factor(
            summary_df,
            factor_row="INPUT_MODE",
            factor_col="IMG_SIZE",
            metric=metric,
            out_path=heatmap_is,
            title=f"{metric} heatmap (INPUT_MODE × IMG_SIZE)",
            row_order=input_order,
            col_order=size_order,
        )

        # INPUT × BLUR
        heatmap_ib = metric_dir / f"INPUT_MODE_x_BLUR_MODE_{mshort}.png"
        plot_heatmap_factor_factor(
            summary_df,
            factor_row="INPUT_MODE",
            factor_col="BLUR_MODE",
            metric=metric,
            out_path=heatmap_ib,
            title=f"{metric} heatmap (INPUT_MODE × BLUR_MODE)",
            row_order=input_order,
            col_order=blur_order,
        )

        # SIZE × BLUR
        heatmap_sb = metric_dir / f"IMG_SIZE_x_BLUR_MODE_{mshort}.png"
        plot_heatmap_factor_factor(
            summary_df,
            factor_row="IMG_SIZE",
            factor_col="BLUR_MODE",
            metric=metric,
            out_path=heatmap_sb,
            title=f"{metric} heatmap (IMG_SIZE × BLUR_MODE)",
            row_order=size_order,
            col_order=blur_order,
        )

    # ----------------------------------------------------
    # 3. Top/Bottom散布図（x=mae_deg, y=p95_deg, rank=mae_deg）
    # ----------------------------------------------------
    scatter_path = out_dir / "scatter_top_bottom_mae_vs_p95.png"
    plot_top_bottom_scatter(
        summary_df,
        x_metric="mae_deg",
        y_metric="p95_deg",
        rank_metric="mae_deg",
        topk=args.topk,
        out_path=scatter_path,
        title="Top/Bottom scatter (MAE vs P95, ranked by MAE)",
    )

    print("\n[DONE] 解析完了")


if __name__ == "__main__":
    main()
