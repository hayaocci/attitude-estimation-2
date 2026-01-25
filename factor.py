#!/usr/bin/env python3
"""
analyze_preprocess_experiments_main_effects_v4.py

目的:
  27通り（INPUT_MODE × IMG_SIZE × BLUR_MODE）の前処理実験について、
  1) exp単位に集約した指標（mae/p95/iqr...）で主効果を分析（棒＋箱ひげ）
  2) 元データ（サンプル単位の |err_roll| / |kf_err| ）でも、同じ切り口で箱ひげ図を作成

v4 変更点（ユーザ指定）:
  - すべての箱ひげ図（exp指標 + raw）を min–max 仕様に統一
      * whis=(0, 100)  : ひげを最小値〜最大値に
      * showfliers=False: 外れ値（〇）を表示しない
  - デフォルト出力フォルダ名に "effect" を追加:
      out_dir = <csv_stem>_analysis_effect/
  - 棒グラフはバーのみ（エラーバー線は不要）

入力:
  - 全expまとめCSV（サンプル単位の行。少なくとも exp_id, err_roll を含む）
  - INPUT_MODE, IMG_SIZE, TRAIN_BLUR, TRAIN_BLUR_KERNEL があれば利用して BLUR_MODE を復元
  - kf_err があれば KF 指標も計算 & raw箱ひげ図も追加

出力:
  --out_dir 未指定なら、入力csvと同じ階層に <csv_stem>_analysis_effect/ を作成
  1) exp単位のサマリ: preprocess_summary_table.csv
  2) exp指標の主効果（CSV + 図）: out_dir/<metric_short>/
       - main_effect_*.png（バーのみ）
       - boxplot_*.png（exp指標の箱ひげ：min-max）
       - main_effects_table_*.csv
  3) raw（サンプル単位）の箱ひげ図: out_dir/raw_boxplots/
       - raw_boxplot_INPUT_MODE_abs_err.png
       - raw_boxplot_IMG_SIZE_abs_err.png
       - raw_boxplot_BLUR_MODE_abs_err.png
       - (KFがあれば) raw_boxplot_*_abs_kf_err.png
  4) 影響幅まとめ: main_effects_delta_ranges_all_metrics*.csv

使い方:
  python analyze_preprocess_experiments_main_effects_v4.py --csv /path/to/all_experiments.csv
  python analyze_preprocess_experiments_main_effects_v4.py --csv /path/to/all_experiments.csv --out_dir /path/to/out

注意:
  - exp指標の箱ひげ図: exp単位の metric 値の分布
  - raw箱ひげ図: サンプル単位の |err_roll|（または |kf_err|）分布
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------
# 引数
# --------------------------------------------------------
def parse_args():
    pa = argparse.ArgumentParser(
        description="前処理27通りの主効果（exp指標: バー＋箱ひげ）＋ raw誤差の箱ひげ図（min-max）を出力"
    )
    pa.add_argument("--csv", required=True, help="全expが含まれている評価csvのパス")
    pa.add_argument(
        "--out_dir",
        default=None,
        help=(
            "図やサマリcsvを出力するディレクトリ。"
            "未指定の場合は、csvと同じ階層に <csv_stem>_analysis_effect/ を作成して出力"
        ),
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
    return s in ("true", "1", "t", "yes", "y")


# --------------------------------------------------------
# exp単位のBLUR_MODEを決める（提示コード踏襲）
# --------------------------------------------------------
def compute_blur_mode_from_group(g: pd.DataFrame) -> str:
    train_blur = to_bool_like(g["TRAIN_BLUR"].iloc[0]) if "TRAIN_BLUR" in g.columns else False
    blur_kernel_raw = g["TRAIN_BLUR_KERNEL"].iloc[0] if "TRAIN_BLUR_KERNEL" in g.columns else np.nan

    if (not train_blur) or pd.isna(blur_kernel_raw):
        return "none"

    try:
        k = int(blur_kernel_raw)
    except Exception:
        k = None

    if k is None:
        return "none"
    if k <= 3:
        return "weak"
    return "strong"


# --------------------------------------------------------
# exp単位のサマリ作成（提示コード踏襲）
# --------------------------------------------------------
def aggregate_by_exp(df: pd.DataFrame) -> pd.DataFrame:
    if "exp_id" not in df.columns:
        raise ValueError("csv に 'exp_id' 列がありません。")
    if "err_roll" not in df.columns:
        raise ValueError("csv に 'err_roll' 列がありません。")

    has_kf = "kf_err" in df.columns

    rows = []
    for exp_id, g in df.groupby("exp_id"):
        input_mode = g["INPUT_MODE"].iloc[0] if "INPUT_MODE" in g.columns else "unknown"
        img_size = int(g["IMG_SIZE"].iloc[0]) if "IMG_SIZE" in g.columns else -1
        blur_mode = compute_blur_mode_from_group(g)

        abs_err = g["err_roll"].abs()

        mae_deg = float(abs_err.mean())
        std_deg = float(abs_err.std(ddof=0))
        p95 = float(abs_err.quantile(0.95))
        max_err_deg = float(abs_err.max())

        median_deg = float(abs_err.median())
        p05 = float(abs_err.quantile(0.05))
        center_width90_deg = p95 - p05

        trimmed_mask = abs_err <= p95
        trimmed_mae95_deg = float(abs_err[trimmed_mask].mean()) if trimmed_mask.any() else np.nan

        q25 = float(abs_err.quantile(0.25))
        q75 = float(abs_err.quantile(0.75))
        iqr_deg = q75 - q25

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
            trimmed_mae95_kf_deg = float(abs_kf[trimmed_mask_kf].mean()) if trimmed_mask_kf.any() else np.nan

            q25_kf = float(abs_kf.quantile(0.25))
            q75_kf = float(abs_kf.quantile(0.75))
            iqr_kf_deg = q75_kf - q25_kf

        rows.append(
            {
                "exp_id": exp_id,
                "INPUT_MODE": input_mode,
                "IMG_SIZE": img_size,
                "BLUR_MODE": blur_mode,
                "mae_deg": mae_deg,
                "std_deg": std_deg,
                "p95_deg": p95,
                "max_err_deg": max_err_deg,
                "median_deg": median_deg,
                "center_width90_deg": center_width90_deg,
                "trimmed_mae95_deg": trimmed_mae95_deg,
                "iqr_deg": iqr_deg,
                "mae_kf_deg": mae_kf_deg,
                "std_kf_deg": std_kf_deg,
                "p95_kf_deg": p95_kf_deg,
                "max_kf_deg": max_kf_deg,
                "median_kf_deg": median_kf_deg,
                "center_width90_kf_deg": center_width90_kf_deg,
                "trimmed_mae95_kf_deg": trimmed_mae95_kf_deg,
                "iqr_kf_deg": iqr_kf_deg,
                "n_samples": len(g),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("mae_deg").reset_index(drop=True)
    return summary_df


# --------------------------------------------------------
# raw（サンプル単位）用に、各行へ BLUR_MODE を付与する
# --------------------------------------------------------
def add_blur_mode_to_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "exp_id" not in df.columns:
        raise ValueError("csv に 'exp_id' 列がありません。")

    blur_map: Dict[str, str] = {}
    for exp_id, g in df.groupby("exp_id"):
        blur_map[exp_id] = compute_blur_mode_from_group(g)

    out = df.copy()
    out["BLUR_MODE"] = out["exp_id"].map(blur_map).fillna("none")
    return out


# --------------------------------------------------------
# 主効果テーブル（exp単位で集計）
# --------------------------------------------------------
def make_main_effect_table(
    summary_df: pd.DataFrame,
    factor: str,
    metric: str,
    order: Optional[List] = None,
) -> pd.DataFrame:
    g = summary_df.groupby(factor)[metric]
    out = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0), "n_exp": g.count()}).reset_index()

    if order is not None:
        order_map = {v: i for i, v in enumerate(order)}
        out["_order"] = out[factor].map(lambda x: order_map.get(x, 10_000))
        out = out.sort_values(["_order", factor]).drop(columns=["_order"]).reset_index(drop=True)
    else:
        out = out.sort_values("mean").reset_index(drop=True)
    return out


def make_effect_delta_table(main_effect_df: pd.DataFrame, factor: str) -> pd.DataFrame:
    means = main_effect_df["mean"].values
    if len(means) == 0:
        return pd.DataFrame(
            [{"factor": factor, "n_levels": 0, "best_mean": np.nan, "worst_mean": np.nan, "range(best-worst)": np.nan}]
        )
    best = float(np.nanmin(means))
    worst = float(np.nanmax(means))
    return pd.DataFrame(
        [{"factor": factor, "n_levels": int(len(means)), "best_mean": best, "worst_mean": worst, "range(best-worst)": worst - best}]
    )


# --------------------------------------------------------
# 図：棒グラフ（バーのみ、エラーバーなし）
# --------------------------------------------------------
def plot_main_effect_bar_only(
    main_effect_df: pd.DataFrame,
    factor: str,
    metric: str,
    out_path: Path,
    title: Optional[str] = None,
):
    labels = main_effect_df[factor].astype(str).values
    y = main_effect_df["mean"].values

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(labels)), y)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    ax.set_ylabel(metric)
    ax.set_title(title or f"Main effect (bar): {factor} - {metric}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    for i, v in enumerate(y):
        ax.text(i, v, "-" if np.isnan(v) else f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Bar-only plot saved: {out_path}")


# --------------------------------------------------------
# 図：箱ひげ図（exp単位の指標分布）— min-max ひげで統一
# --------------------------------------------------------
def plot_boxplot_exp_metric_minmax(
    summary_df: pd.DataFrame,
    factor: str,
    metric: str,
    out_path: Path,
    order: Optional[List] = None,
    title: Optional[str] = None,
):
    levels = order if order is not None else sorted(summary_df[factor].dropna().unique().tolist())

    data = []
    labels = []
    ns = []
    for lv in levels:
        vals = summary_df.loc[summary_df[factor] == lv, metric].dropna().values
        data.append(vals)
        labels.append(str(lv))
        ns.append(len(vals))

    fig, ax = plt.subplots(figsize=(6, 4))

    if all(len(d) == 0 for d in data):
        ax.text(0.5, 0.5, "No data (all NaN)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Exp-metric boxplot saved (empty): {out_path}")
        return

    # min-max whisker, no fliers
    ax.boxplot(
        data,
        labels=[f"{lab}\n(n={n})" for lab, n in zip(labels, ns)],
        showfliers=False,
        whis=(0, 100),
    )
    ax.set_ylabel(metric)
    ax.set_title(title or f"Boxplot (min-max, exp-metric): {factor} - {metric}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Exp-metric boxplot saved: {out_path}")


# --------------------------------------------------------
# 図：箱ひげ図（raw: サンプル単位の |err| 分布）— min-max ひげで統一
# --------------------------------------------------------
def plot_raw_boxplot_abs_error_minmax(
    df_rows: pd.DataFrame,
    factor: str,
    err_col: str,
    out_path: Path,
    order: Optional[List] = None,
    title: Optional[str] = None,
):
    if err_col not in df_rows.columns:
        raise ValueError(f"raw boxplot: df に '{err_col}' 列がありません。")

    levels = order if order is not None else sorted(df_rows[factor].dropna().unique().tolist())

    data = []
    labels = []
    ns = []
    for lv in levels:
        v = df_rows.loc[df_rows[factor] == lv, err_col].abs().dropna().values
        data.append(v)
        labels.append(str(lv))
        ns.append(len(v))

    fig, ax = plt.subplots(figsize=(7, 4))

    if all(len(d) == 0 for d in data):
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Raw boxplot saved (empty): {out_path}")
        return

    ax.boxplot(
        data,
        labels=[f"{lab}\n(n={n})" for lab, n in zip(labels, ns)],
        showfliers=False,
        whis=(0, 100),
    )
    ax.set_ylabel(f"|{err_col}| (deg)")
    ax.set_title(title or f"Raw boxplot (min-max): {factor} - |{err_col}|")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Raw boxplot saved: {out_path}")


# --------------------------------------------------------
# メイン
# --------------------------------------------------------
def main():
    args = parse_args()
    csv_path = Path(args.csv).resolve()

    # ★修正点: effect を追加
    out_dir = (csv_path.parent / f"{csv_path.stem}_analysis_effect") if args.out_dir is None else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Output directory: {out_dir}")

    df = pd.read_csv(csv_path)

    # exp単位サマリ
    summary_df = aggregate_by_exp(df)
    summary_df.to_csv(out_dir / "preprocess_summary_table.csv", index=False)
    print(f"[OK] Summary table saved: {out_dir / 'preprocess_summary_table.csv'}")

    # 因子順（提示コード踏襲）
    input_order = [m for m in ["rgb", "gray", "bin4"] if m in summary_df["INPUT_MODE"].unique()]
    size_candidates = [56, 112, 224]
    unique_sizes = list(summary_df["IMG_SIZE"].unique())
    size_order = [s for s in size_candidates if s in unique_sizes] + [s for s in unique_sizes if s not in size_candidates]
    blur_order = [b for b in ["none", "weak", "strong"] if b in summary_df["BLUR_MODE"].unique()]

    # 指標一覧（非KF + KF）
    metrics = [
        ("mae_deg", "mae"),
        ("std_deg", "std"),
        ("p95_deg", "p95"),
        ("max_err_deg", "max_err"),
        ("median_deg", "median"),
        ("center_width90_deg", "width90"),
        ("trimmed_mae95_deg", "tmae95"),
        ("iqr_deg", "iqr"),
    ]
    has_kf = ("kf_err" in df.columns) and (not summary_df["mae_kf_deg"].isna().all())
    if has_kf:
        metrics += [
            ("mae_kf_deg", "mae_kf"),
            ("std_kf_deg", "std_kf"),
            ("p95_kf_deg", "p95_kf"),
            ("max_kf_deg", "max_kf"),
            ("median_kf_deg", "median_kf"),
            ("center_width90_kf_deg", "width90_kf"),
            ("trimmed_mae95_kf_deg", "tmae95_kf"),
            ("iqr_kf_deg", "iqr_kf"),
        ]

    # --------------- exp指標の主効果（バーのみ＋箱ひげ(min-max)） ---------------
    delta_rows = []

    for metric, short in metrics:
        metric_dir = out_dir / short
        metric_dir.mkdir(parents=True, exist_ok=True)

        # INPUT_MODE
        me_input = make_main_effect_table(summary_df, "INPUT_MODE", metric, order=input_order)
        me_input.to_csv(metric_dir / f"main_effects_table_INPUT_MODE_{short}.csv", index=False)
        plot_main_effect_bar_only(
            me_input,
            factor="INPUT_MODE",
            metric=metric,
            out_path=metric_dir / f"main_effect_INPUT_MODE_{short}.png",
            title=f"Main effect (INPUT_MODE) - {metric}",
        )
        plot_boxplot_exp_metric_minmax(
            summary_df,
            factor="INPUT_MODE",
            metric=metric,
            order=input_order,
            out_path=metric_dir / f"boxplot_INPUT_MODE_{short}.png",
            title=f"Boxplot (min-max) by INPUT_MODE - {metric}",
        )
        delta_rows.append(make_effect_delta_table(me_input, factor=f"INPUT_MODE ({metric})"))

        # IMG_SIZE
        me_size = make_main_effect_table(summary_df, "IMG_SIZE", metric, order=size_order)
        me_size.to_csv(metric_dir / f"main_effects_table_IMG_SIZE_{short}.csv", index=False)
        plot_main_effect_bar_only(
            me_size,
            factor="IMG_SIZE",
            metric=metric,
            out_path=metric_dir / f"main_effect_IMG_SIZE_{short}.png",
            title=f"Main effect (IMG_SIZE) - {metric}",
        )
        plot_boxplot_exp_metric_minmax(
            summary_df,
            factor="IMG_SIZE",
            metric=metric,
            order=size_order,
            out_path=metric_dir / f"boxplot_IMG_SIZE_{short}.png",
            title=f"Boxplot (min-max) by IMG_SIZE - {metric}",
        )
        delta_rows.append(make_effect_delta_table(me_size, factor=f"IMG_SIZE ({metric})"))

        # BLUR_MODE
        me_blur = make_main_effect_table(summary_df, "BLUR_MODE", metric, order=blur_order)
        me_blur.to_csv(metric_dir / f"main_effects_table_BLUR_MODE_{short}.csv", index=False)
        plot_main_effect_bar_only(
            me_blur,
            factor="BLUR_MODE",
            metric=metric,
            out_path=metric_dir / f"main_effect_BLUR_MODE_{short}.png",
            title=f"Main effect (BLUR_MODE) - {metric}",
        )
        plot_boxplot_exp_metric_minmax(
            summary_df,
            factor="BLUR_MODE",
            metric=metric,
            order=blur_order,
            out_path=metric_dir / f"boxplot_BLUR_MODE_{short}.png",
            title=f"Boxplot (min-max) by BLUR_MODE - {metric}",
        )
        delta_rows.append(make_effect_delta_table(me_blur, factor=f"BLUR_MODE ({metric})"))

        # metricごとに因子まとめCSV
        combined = []
        tmp = me_input.copy()
        tmp.insert(0, "factor", "INPUT_MODE")
        combined.append(tmp)
        tmp = me_size.copy()
        tmp.insert(0, "factor", "IMG_SIZE")
        combined.append(tmp)
        tmp = me_blur.copy()
        tmp.insert(0, "factor", "BLUR_MODE")
        combined.append(tmp)
        pd.concat(combined, axis=0, ignore_index=True).to_csv(metric_dir / f"main_effects_table_ALL_FACTORS_{short}.csv", index=False)

        print(f"[OK] Metric outputs saved under: {metric_dir}")

    # 影響幅まとめ
    if len(delta_rows) > 0:
        delta_df = pd.concat(delta_rows, axis=0, ignore_index=True)
        delta_df.to_csv(out_dir / "main_effects_delta_ranges_all_metrics.csv", index=False)
        delta_df.sort_values("range(best-worst)", ascending=False).reset_index(drop=True).to_csv(
            out_dir / "main_effects_delta_ranges_all_metrics_sorted.csv", index=False
        )
        print(f"[OK] Effect-range summaries saved under: {out_dir}")

    # --------------- raw（元データ）箱ひげ図（min-max） ---------------
    raw_dir = out_dir / "raw_boxplots"
    raw_dir.mkdir(parents=True, exist_ok=True)

    df_rows = add_blur_mode_to_rows(df)

    if "INPUT_MODE" not in df_rows.columns:
        df_rows["INPUT_MODE"] = "unknown"
    if "IMG_SIZE" not in df_rows.columns:
        df_rows["IMG_SIZE"] = -1

    # raw: err_roll
    plot_raw_boxplot_abs_error_minmax(
        df_rows,
        factor="INPUT_MODE",
        err_col="err_roll",
        order=input_order if len(input_order) > 0 else None,
        out_path=raw_dir / "raw_boxplot_INPUT_MODE_abs_err.png",
        title="Raw |err_roll| boxplot (min-max) by INPUT_MODE",
    )
    plot_raw_boxplot_abs_error_minmax(
        df_rows,
        factor="IMG_SIZE",
        err_col="err_roll",
        order=size_order if len(size_order) > 0 else None,
        out_path=raw_dir / "raw_boxplot_IMG_SIZE_abs_err.png",
        title="Raw |err_roll| boxplot (min-max) by IMG_SIZE",
    )
    plot_raw_boxplot_abs_error_minmax(
        df_rows,
        factor="BLUR_MODE",
        err_col="err_roll",
        order=blur_order if len(blur_order) > 0 else None,
        out_path=raw_dir / "raw_boxplot_BLUR_MODE_abs_err.png",
        title="Raw |err_roll| boxplot (min-max) by BLUR_MODE",
    )

    # raw: kf_err（あれば）
    if "kf_err" in df_rows.columns:
        plot_raw_boxplot_abs_error_minmax(
            df_rows,
            factor="INPUT_MODE",
            err_col="kf_err",
            order=input_order if len(input_order) > 0 else None,
            out_path=raw_dir / "raw_boxplot_INPUT_MODE_abs_kf_err.png",
            title="Raw |kf_err| boxplot (min-max) by INPUT_MODE",
        )
        plot_raw_boxplot_abs_error_minmax(
            df_rows,
            factor="IMG_SIZE",
            err_col="kf_err",
            order=size_order if len(size_order) > 0 else None,
            out_path=raw_dir / "raw_boxplot_IMG_SIZE_abs_kf_err.png",
            title="Raw |kf_err| boxplot (min-max) by IMG_SIZE",
        )
        plot_raw_boxplot_abs_error_minmax(
            df_rows,
            factor="BLUR_MODE",
            err_col="kf_err",
            order=blur_order if len(blur_order) > 0 else None,
            out_path=raw_dir / "raw_boxplot_BLUR_MODE_abs_kf_err.png",
            title="Raw |kf_err| boxplot (min-max) by BLUR_MODE",
        )

    print("\n[DONE] Main-effect analysis completed (bar-only + min-max boxplots for exp metrics and raw errors).")


if __name__ == "__main__":
    main()
