#!/usr/bin/env python3
"""
make_err_plots_multi_exp.py
--------------------------------
- 複数の exp_id を含む CSV を読み込み、
  各 exp_id ごとに以下の図を出力する:

  1) 散布図: true_roll vs err_roll (CNN)
  2) 散布図: true_roll vs err_roll/kf_err (CNN vs EKF) ※kf_err 列がある場合
  3) ヒストグラム: err_roll (CNN)
  4) ヒストグラム: err_roll vs kf_err (CNN vs EKF) ※kf_err 列がある場合

- さらに、すべての exp をまとめた:

  5) 散布図 (全 exp 統合, CNN) ※各 exp ごとに直線で接続
     → 凡例は別画像として保存
  6) 散布図 (全 exp 統合, CNN vs EKF) ※各 exp ごとに直線で接続
     → 凡例は別画像として保存
  7) ヒストグラム (全 exp 統合, CNN のみ or CNN vs EKF)

- 出力先:
    CSV が foo/bar/results.csv のとき
      個別:  foo/bar/figs/exp183/error_vs_true_cnn.png など
      全体:  foo/bar/figs/error_vs_true_all_cnn.png など

使い方例:
    python make_err_plots_multi_exp.py --csv path/to/comp_all.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 引数
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="exp_id ごと & 全体の散布図・ヒストグラム作成スクリプト"
    )
    p.add_argument(
        "--csv",
        required=True,
        help="評価結果 CSV のパス (true_roll, err_roll, kf_err, exp_id 等を含む)",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="図の出力先ディレクトリ (指定しない場合は CSV と同じ場所の figs/)",
    )
    return p.parse_args()


# ============================================================
# メタ情報（INPUT_MODE, IMG_SIZE, BLUR）生成
# ============================================================
def _to_bool_like(x):
    """'True' / 'False' / 1 / 0 などをざっくり bool に寄せる"""
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f", ""):
        return False
    # よくわからない場合はいちおう True 扱い
    return True


def blur_level_from_row(row: pd.Series) -> str:
    """
    TRAIN_BLUR / TRAIN_BLUR_KERNEL → ブラー強度を英語表記に変換
    - None
    - Small (k=3)
    - Large (k=7)
    - k={value} (その他)
    """
    # TRAIN 用を優先し、無い場合 VALID を見る
    blur_flag = row.get("TRAIN_BLUR", row.get("VALID_BLUR", False))

    # 真偽判定（True/False/0/1/None対応）
    if isinstance(blur_flag, str):
        blur_flag = blur_flag.strip().lower() in ("true", "1", "yes", "y", "t")
    else:
        blur_flag = bool(blur_flag)

    # ブラーが無しなら即 None
    if not blur_flag:
        return "None"

    # kernel を取得
    kernel = row.get("TRAIN_BLUR_KERNEL", row.get("VALID_BLUR_KERNEL", None))

    # kernel が取れない or NaN の場合は fallback
    try:
        if pd.isna(kernel):
            return "k=NA"
    except Exception:
        pass

    # kernel を数値判定
    try:
        k = int(kernel)
    except Exception:
        return f"k={kernel}"

    # ルール適用
    if k == 3:
        return "Small"
    elif k == 7:
        return "Large"
    else:
        return f"k={k}"  # その他の値


def get_meta_info(df_exp: pd.DataFrame) -> tuple[str, str, str, str]:
    """
    その exp の代表行を 1 行取り、メタ情報を返す。
    戻り値: (input_mode, img_size_str, blur_txt, meta_label)
      - meta_label は "rgb, 112, Small" のような短い文字列
    """
    row = df_exp.iloc[0]

    input_mode = str(row.get("INPUT_MODE", "")).strip()
    img_size = row.get("IMG_SIZE", "")
    img_size_str = str(img_size)
    try:
        if not pd.isna(img_size):
            img_size_str = str(int(img_size))
    except Exception:
        img_size_str = str(img_size)

    blur_txt = blur_level_from_row(row)
    meta_label = f"{input_mode}, {img_size_str}, {blur_txt}"
    return input_mode, img_size_str, blur_txt, meta_label


# ============================================================
# 個別 exp 用プロット（凡例なし、タイトルにメタ情報を含める）
# ============================================================
def make_plots_for_one_exp(df_exp: pd.DataFrame, exp_id: str, out_dir: Path):
    """
    df_exp: 1 つの exp_id のみを含む DataFrame
    out_dir: figs/expXXX/ のような出力フォルダ
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 必須カラムチェック
    for c in ["true_roll", "err_roll"]:
        if c not in df_exp.columns:
            raise ValueError(f"CSV に '{c}' 列が存在しません。")

    has_kf = "kf_err" in df_exp.columns
    input_mode, img_size_str, blur_txt, meta_label = get_meta_info(df_exp)
    meta_bracket = f"[input={input_mode}, size={img_size_str}, blur={blur_txt}]"

    # -------- ① 散布図: Error vs True (CNN) --------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df_exp["true_roll"], df_exp["err_roll"], s=8, alpha=0.5)
    ax.set_title(f"{exp_id}  Error vs True (CNN) {meta_bracket}")
    ax.set_xlabel("True Roll [deg]")
    ax.set_ylabel("Error [deg]")
    fig.tight_layout()
    fig.savefig(out_dir / "error_vs_true_cnn.png", dpi=150)
    plt.close(fig)

    # -------- ② 散布図: Error vs True (CNN vs EKF) --------
    if has_kf:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df_exp["true_roll"], df_exp["err_roll"],
                   s=8, alpha=0.5)
        ax.scatter(df_exp["true_roll"], df_exp["kf_err"],
                   s=8, alpha=0.5)
        ax.set_title(f"{exp_id}  Error vs True (CNN vs EKF) {meta_bracket}")
        ax.set_xlabel("True Roll [deg]")
        ax.set_ylabel("Error [deg]")
        fig.tight_layout()
        fig.savefig(out_dir / "error_vs_true_cnn_vs_kf.png", dpi=150)
        plt.close(fig)

    # -------- ③ ヒストグラム: Error (CNN) --------
    bins = np.arange(0, 181, 10)  # [0,10,20,...,180]

    fig, ax = plt.subplots(figsize=(7, 5))
    Y_cnn, bin_edges, _ = ax.hist(
        df_exp["err_roll"],
        bins=bins,
        alpha=0.7,
        range=(0, 180),
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_title(f"{exp_id}  Error Histogram (CNN) {meta_bracket}")
    ax.set_xlabel("Error [deg]")
    ax.set_ylabel("Count")

    ax.set_xticks(np.arange(0, 181, 20))

    y_max = int(max(Y_cnn)) + 1 if len(Y_cnn) > 0 else 1
    if y_max <= 10:
        step = 1
    elif y_max <= 40:
        step = 2
    else:
        step = 5
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, step))

    fig.tight_layout()
    fig.savefig(out_dir / "error_hist_cnn.png", dpi=150)
    plt.close(fig)

    # -------- ④ ヒストグラム: Error (CNN vs EKF) --------
    if has_kf:
        fig, ax = plt.subplots(figsize=(7, 5))
        Y1, bin_edges1, _ = ax.hist(
            df_exp["err_roll"],
            bins=bins,
            alpha=0.5,
            range=(0, 180),
            edgecolor="black",
            linewidth=1.0,
        )
        Y2, bin_edges2, _ = ax.hist(
            df_exp["kf_err"],
            bins=bins,
            alpha=0.5,
            range=(0, 180),
            edgecolor="black",
            linewidth=1.0,
        )

        ax.set_title(f"{exp_id}  Error Histogram (CNN vs EKF) {meta_bracket}")
        ax.set_xlabel("Error [deg]")
        ax.set_ylabel("Count")

        ax.set_xticks(np.arange(0, 181, 20))

        y_max = int(max(max(Y1), max(Y2))) + 1 if len(Y1) > 0 else 1
        if y_max <= 10:
            step = 1
        elif y_max <= 40:
            step = 2
        else:
            step = 5
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.arange(0, y_max + 1, step))

        fig.tight_layout()
        fig.savefig(out_dir / "error_hist_cnn_vs_kf.png", dpi=150)
        plt.close(fig)


# ============================================================
# 全 exp をまとめたプロット
#   - 統合版散布図: 各 exp を直線で接続
#   - 統合版散布図の凡例は別画像として保存
# ============================================================
def make_plots_all_exps(df: pd.DataFrame, out_dir: Path):
    """
    全 exp を一枚の図にまとめて表示。
    - 散布図: exp ごとに true_roll でソートして直線で接続 (CNN, CNN vs EKF)
    - ヒストグラム: exp ごとにヒストグラム重ね描画
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if "exp_id" not in df.columns:
        print("WARNING: 'exp_id' 列が無いので、全体図は 1 シリーズとして描画します。")
        exp_groups = {"all": df}
    else:
        exp_groups = {eid: g for eid, g in df.groupby("exp_id")}

    has_kf = "kf_err" in df.columns
    bins = np.arange(0, 181, 10)

    # -------- (全体) 散布図 CNN (線で接続) --------
    fig, ax = plt.subplots(figsize=(7, 5))
    lines = []
    labels = []

    for eid, g in exp_groups.items():
        # true_roll でソートして、線で接続
        g_sorted = g.sort_values("true_roll")
        _, _, _, meta_label = get_meta_info(g_sorted)
        line, = ax.plot(
            g_sorted["true_roll"].values,
            g_sorted["err_roll"].values,
            marker="o",
            markersize=3,
            linewidth=1.0,
            alpha=0.8,
        )
        lines.append(line)
        labels.append(meta_label)

    ax.set_title("Error vs True (CNN) - ALL exps")
    ax.set_xlabel("True Roll [deg]")
    ax.set_ylabel("Error [deg]")

    fig.tight_layout()
    fig.savefig(out_dir / "error_vs_true_all_cnn.png", dpi=150)
    plt.close(fig)

    # 別画像として凡例のみを保存
    if lines:
        fig_leg, ax_leg = plt.subplots(figsize=(6, max(1.5, 0.3 * len(labels))))
        ax_leg.axis("off")
        leg = ax_leg.legend(
            lines,
            labels,
            loc="center",
            frameon=False,
        )
        fig_leg.savefig(out_dir / "error_vs_true_all_cnn_legend.png",
                        dpi=150, bbox_inches="tight")
        plt.close(fig_leg)

    # -------- (全体) 散布図 CNN vs EKF (線で接続) --------
    if has_kf:
        fig, ax = plt.subplots(figsize=(7, 5))
        lines_cnn = []
        lines_kf = []
        labels_cnn = []
        labels_kf = []

        for eid, g in exp_groups.items():
            g_sorted = g.sort_values("true_roll")
            _, _, _, meta_label = get_meta_info(g_sorted)

            line_cnn, = ax.plot(
                g_sorted["true_roll"].values,
                g_sorted["err_roll"].values,
                marker="o",
                markersize=3,
                linewidth=1.0,
                alpha=0.7,
            )
            line_kf, = ax.plot(
                g_sorted["true_roll"].values,
                g_sorted["kf_err"].values,
                marker="x",
                markersize=3,
                linewidth=1.0,
                alpha=0.7,
            )
            lines_cnn.append(line_cnn)
            lines_kf.append(line_kf)
            labels_cnn.append(f"{meta_label} (CNN)")
            labels_kf.append(f"{meta_label} (EKF)")

        ax.set_title("Error vs True (CNN vs EKF) - ALL exps")
        ax.set_xlabel("True Roll [deg]")
        ax.set_ylabel("Error [deg]")

        fig.tight_layout()
        fig.savefig(out_dir / "error_vs_true_all_cnn_vs_kf.png", dpi=150)
        plt.close(fig)

        # 凡例だけ別画像に保存
        if lines_cnn or lines_kf:
            fig_leg, ax_leg = plt.subplots(
                figsize=(7, max(2.0, 0.3 * (len(labels_cnn) + len(labels_kf))))
            )
            ax_leg.axis("off")
            all_lines = lines_cnn + lines_kf
            all_labels = labels_cnn + labels_kf
            leg = ax_leg.legend(
                all_lines,
                all_labels,
                loc="center",
                frameon=False,
                ncol=2,
            )
            fig_leg.savefig(out_dir / "error_vs_true_all_cnn_vs_kf_legend.png",
                            dpi=150, bbox_inches="tight")
            plt.close(fig_leg)

    # -------- (全体) ヒストグラム CNN --------
    fig, ax = plt.subplots(figsize=(7, 5))
    y_max_all = 0

    for eid, g in exp_groups.items():
        _, _, _, meta_label = get_meta_info(g)
        Y, bin_edges, _ = ax.hist(
            g["err_roll"],
            bins=bins,
            alpha=0.4,
            range=(0, 180),
            label=meta_label,
            edgecolor="black",
            linewidth=0.7,
        )
        if len(Y) > 0:
            y_max_all = max(y_max_all, int(max(Y)))

    ax.set_title("Error Histogram (CNN) - ALL exps")
    ax.set_xlabel("Error [deg]")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    ax.set_xticks(np.arange(0, 181, 20))

    y_max = y_max_all + 1 if y_max_all > 0 else 1
    if y_max <= 10:
        step = 1
    elif y_max <= 40:
        step = 2
    else:
        step = 5
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, step))

    fig.tight_layout()
    fig.savefig(out_dir / "error_hist_all_cnn.png", dpi=150)
    plt.close(fig)

    # -------- (全体) ヒストグラム CNN vs EKF --------
    if has_kf:
        fig, ax = plt.subplots(figsize=(7, 5))
        y_max_all = 0

        for eid, g in exp_groups.items():
            _, _, _, meta_label = get_meta_info(g)
            Y1, _, _ = ax.hist(
                g["err_roll"],
                bins=bins,
                alpha=0.3,
                range=(0, 180),
                label=f"{meta_label} (CNN)",
                edgecolor="black",
                linewidth=0.7,
            )
            Y2, _, _ = ax.hist(
                g["kf_err"],
                bins=bins,
                alpha=0.3,
                range=(0, 180),
                label=f"{meta_label} (EKF)",
                edgecolor="black",
                linewidth=0.7,
            )
            if len(Y1) > 0:
                y_max_all = max(y_max_all, int(max(Y1)))
            if len(Y2) > 0:
                y_max_all = max(y_max_all, int(max(Y2)))

        ax.set_title("Error Histogram (CNN vs EKF) - ALL exps")
        ax.set_xlabel("Error [deg]")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7, ncol=2)

        ax.set_xticks(np.arange(0, 181, 20))

        y_max = y_max_all + 1 if y_max_all > 0 else 1
        if y_max <= 10:
            step = 1
        elif y_max <= 40:
            step = 2
        else:
            step = 5
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.arange(0, y_max + 1, step))

        fig.tight_layout()
        fig.savefig(out_dir / "error_hist_all_cnn_vs_kf.png", dpi=150)
        plt.close(fig)


# ============================================================
# main
# ============================================================
def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print("CSV が見つかりません:", csv_path)
        return

    if args.out_dir is None:
        figs_root = csv_path.parent / "figs"
    else:
        figs_root = Path(args.out_dir)
    figs_root.mkdir(parents=True, exist_ok=True)

    print("Read CSV:", csv_path)
    df = pd.read_csv(csv_path)

    if "exp_id" not in df.columns:
        print("WARNING: 'exp_id' 列がありません。全体のみ描画します。")
        make_plots_all_exps(df, figs_root)
        print("Done.")
        return

    # --- exp_id ごとに個別図を作成 ---
    for exp_id, df_exp in df.groupby("exp_id"):
        exp_dir = figs_root / str(exp_id)
        print(f"  Make plots for {exp_id} -> {exp_dir}")
        make_plots_for_one_exp(df_exp, str(exp_id), exp_dir)

    # --- 全 exp 統合の図 ---
    print("  Make ALL-exps plots ->", figs_root)
    make_plots_all_exps(df, figs_root)

    print("Done.")


if __name__ == "__main__":
    main()
