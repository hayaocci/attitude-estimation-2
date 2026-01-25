#!/usr/bin/env python3
# code_C_compile_results.py (AUTO DATASET / BLUR INFO 対応版)
from __future__ import annotations
import argparse
from pathlib import Path
import re

import pandas as pd
import yaml

# ============================================================
# コマンドライン引数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "lab_logs/expXX/eval_<dataset>_<split>_kf/eval_results_kf.csv と "
            "config_used.yaml を集約して1つのCSVにまとめるスクリプト "
            "(dataset名は config_used.yaml から自動検出)"
        )
    )

    parser.add_argument(
        "--split",
        default="valid",
        choices=["train", "valid", "test"],
        help="評価に使った split 名 (default: valid)",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="lab_logs",
        help="lab_logs のルートディレクトリ (default: lab_logs)",
    )

    # -------- 追加: compile 対象の exp を絞り込むオプション --------
    parser.add_argument(
        "--exp_from",
        type=str,
        default=None,
        help="この exp から (例: exp03 / 3)。exp_ids 指定時は無視されます。",
    )
    parser.add_argument(
        "--exp_to",
        type=str,
        default=None,
        help="この exp まで (例: exp10 / 10)。exp_ids 指定時は無視されます。",
    )
    parser.add_argument(
        "--exp_ids",
        type=str,
        default=None,
        help=(
            "compile 対象とする exp をカンマ区切りで指定 (例: exp01,exp03,exp08 または 1,3,8)。"
            "指定された場合は exp_from / exp_to より優先されます。"
        ),
    )

    return parser.parse_args()


# ============================================================
# TRAIN_DATASET_ROOT / VALID_DIR から datasets/XXX の XXX 部分だけ抜き出す
# 例:
#   "datasets/type-2_lab_blur_7" -> "type-2_lab_blur_7"
#   "datasets/type-7_lab/cache/bin4/sz56_area/valid/imgs" -> "type-7_lab"
# ============================================================
def extract_dataset_name(path_like_str: str) -> str:
    """
    与えられたパス文字列から 'datasets/XXX' の XXX 部分だけを取り出す。
    見つからなければ元の文字列をそのまま返す。
    """
    s = str(path_like_str)

    # Path オブジェクトとして辿りながら、親が datasets になっているノードを探す
    p = Path(s)
    for node in [p] + list(p.parents):
        if node.parent.name == "datasets":
            return node.name

    # うまく見つからなければ、"datasets/" で文字列検索して、その次の要素だけ返す
    if "datasets/" in s:
        tail = s.split("datasets/", 1)[1].lstrip("/")
        return tail.split("/", 1)[0]

    # 最後の手段として元の文字列
    return s


# ============================================================
# blur_カーネルサイズ の検出
#   例: "type-2_lab_blur_7" -> (True, 7)
#       "type-8_lab"        -> (False, None)
# ============================================================
def parse_blur_info(name_or_path: str) -> tuple[bool, int | None]:
    s = str(name_or_path)
    m = re.search(r"blur_(\d+)", s)
    if m:
        return True, int(m.group(1))
    return False, None


# ============================================================
# exp 名から数値部分を取り出すユーティリティ (exp03 -> 3)
# うまくパースできない場合は None
# ============================================================
def parse_exp_number(exp_name: str) -> int | None:
    m = re.match(r"exp(\d+)$", exp_name)
    if m:
        return int(m.group(1))
    return None


# ============================================================
# ユーザー入力から exp 名 (expXX) を正規化
# 例: "3" -> "exp03", "exp7" -> "exp07"
# ============================================================
def normalize_exp_name(token: str) -> str:
    token = token.strip()
    # すでに expXX 形式の場合
    if token.startswith("exp"):
        # exp の後ろを数値としてゼロ埋め
        num_part = token[3:]
        try:
            num = int(num_part)
            return f"exp{num:02d}"
        except ValueError:
            # 数値にできなければそのまま返す
            return token
    else:
        # 純粋な数字だけの場合
        try:
            num = int(token)
            return f"exp{num:02d}"
        except ValueError:
            # それ以外はそのまま返す
            return token


# ============================================================
# exp_dirs を exp_ids / exp_from / exp_to で絞り込む
# ============================================================
def filter_exp_dirs(exp_dirs: list[Path], args) -> list[Path]:
    # 1. exp_ids が指定されている場合は最優先
    if args.exp_ids:
        tokens = [t for t in args.exp_ids.split(",") if t.strip()]
        target_names = {normalize_exp_name(t) for t in tokens}
        filtered = [p for p in exp_dirs if p.name in target_names]
        print(
            f"[INFO] exp_ids により {len(filtered)} 個の exp を対象とします: "
            f"{sorted(p.name for p in filtered)}"
        )
        return filtered

    # 2. exp_from / exp_to による範囲指定
    exp_from_num: int | None = None
    exp_to_num: int | None = None

    if args.exp_from:
        name_from = normalize_exp_name(args.exp_from)
        n = parse_exp_number(name_from)
        if n is not None:
            exp_from_num = n
        else:
            print(f"[WARN] exp_from '{args.exp_from}' をパースできませんでした。無視します。")

    if args.exp_to:
        name_to = normalize_exp_name(args.exp_to)
        n = parse_exp_number(name_to)
        if n is not None:
            exp_to_num = n
        else:
            print(f"[WARN] exp_to '{args.exp_to}' をパースできませんでした。無視します。")

    # どちらも指定されていない場合はそのまま
    if exp_from_num is None and exp_to_num is None:
        return exp_dirs

    filtered: list[Path] = []
    for p in exp_dirs:
        n = parse_exp_number(p.name)
        if n is None:
            # expXX 形式でなければ一応スキップ
            print(f"[WARN] exp ディレクトリ名 '{p.name}' から番号を取得できません。スキップします。")
            continue

        if exp_from_num is not None and n < exp_from_num:
            continue
        if exp_to_num is not None and n > exp_to_num:
            continue

        filtered.append(p)

    print(
        f"[INFO] exp_from/exp_to により {len(filtered)} 個の exp を対象とします: "
        f"{sorted(p.name for p in filtered)}"
    )
    return filtered


# ============================================================
# 出力用 analysis サブディレクトリ名を決める
#   - フィルタ指定なし: "all"
#   - exp_ids 指定:      "ids_1-3-8" など
#   - range 指定:        "range_from3_to7" など
# => 最終ディレクトリ: compilation/analysis-<name>/
# ============================================================
def get_analysis_name(args) -> str:
    # 1. exp_ids 優先
    if args.exp_ids:
        tokens = [t for t in args.exp_ids.split(",") if t.strip()]
        norm_names = [normalize_exp_name(t) for t in tokens]

        # 数値部分を優先して短くする (exp03 -> 3)。取れない場合はそのまま。
        parts: list[str] = []
        for name in norm_names:
            n = parse_exp_number(name)
            if n is not None:
                parts.append(str(n))
            else:
                parts.append(name)
        if parts:
            return "ids_" + "-".join(parts)
        return "ids"

    # 2. range (exp_from / exp_to)
    exp_from_num: int | None = None
    exp_to_num: int | None = None

    if args.exp_from:
        name_from = normalize_exp_name(args.exp_from)
        n = parse_exp_number(name_from)
        if n is not None:
            exp_from_num = n

    if args.exp_to:
        name_to = normalize_exp_name(args.exp_to)
        n = parse_exp_number(name_to)
        if n is not None:
            exp_to_num = n

    if exp_from_num is None and exp_to_num is None:
        return "all"

    parts: list[str] = []
    if exp_from_num is not None:
        parts.append(f"from{exp_from_num}")
    if exp_to_num is not None:
        parts.append(f"to{exp_to_num}")

    if not parts:
        return "all"

    return "range_" + "_".join(parts)


# ============================================================
# 1つのexpの結果を読み込んで DataFrame を作る
#   - dataset名は config_used.yaml から自動抽出
#   - TRAIN/VALID 側それぞれ blur 情報も付与
# ============================================================
def load_one_exp(exp_dir: Path, split: str) -> pd.DataFrame | None:
    cfg_path = exp_dir / "config_used.yaml"

    if not cfg_path.exists():
        print(f"  [SKIP] config_used.yaml not found: {cfg_path}")
        return None

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # -------- TRAIN / VALID データセット情報を抽出 --------
    # TRAIN は TRAIN_DATASET_ROOT もしくは _TRAIN_ROOT から
    train_root_raw = cfg.get("TRAIN_DATASET_ROOT", cfg.get("_TRAIN_ROOT", ""))
    train_root_short = extract_dataset_name(train_root_raw)

    # VALID のメインは VALID_DIR / VALID_DATASET_ROOT / _VALID_ROOTS[0] から
    valid_dir = cfg.get("VALID_DIR", "")
    valid_roots = cfg.get("_VALID_ROOTS", [])
    if valid_roots:
        valid_root_raw = valid_roots[0]
    else:
        # VALID_DIR から dataset 名を推定 (datasets/XXX/cache/...) 形式を想定
        valid_root_raw = valid_dir
    valid_root_short = extract_dataset_name(valid_root_raw)

    # -------- blur 情報を解析 --------
    train_blur, train_blur_kernel = parse_blur_info(train_root_short)
    valid_blur, valid_blur_kernel = parse_blur_info(valid_root_short)

    # -------- eval_results_kf.csv の場所を決定 --------
    # eval_<valid_root_short>_<split>_kf/eval_results_kf.csv を想定
    eval_dir = exp_dir / f"eval_{valid_root_short}_{split}_kf"
    eval_csv = eval_dir / "eval_results_kf.csv"

    if not eval_csv.exists():
        print(f"  [SKIP] eval_results_kf.csv not found: {eval_csv}")
        return None

    try:
        df = pd.read_csv(eval_csv)
    except Exception as e:
        print(f"  [ERROR] Failed to read {eval_csv}: {e}")
        return None

    # -------- config からその他の情報を取得 --------
    batch_size = cfg.get("BATCH_SIZE", None)
    dropout_p = cfg.get("DROPOUT_P", None)
    img_size = cfg.get("IMG_SIZE", [None, None])
    img_size_scalar = img_size[0] if isinstance(img_size, (list, tuple)) and img_size else img_size
    input_mode = cfg.get("INPUT_MODE", None)
    color_mode = cfg.get("COLOR_MODE", None)
    max_lr = cfg.get("MAX_LR", None)
    weight_decay = cfg.get("WEIGHT_DECAY", None)
    exp_id = cfg.get("id", exp_dir.name)

    # -------- 全行に同じ情報を付与 --------
    df["exp_id"] = exp_id
    df["BATCH_SIZE"] = batch_size
    df["DROPOUT_P"] = dropout_p
    df["IMG_SIZE"] = img_size_scalar
    df["INPUT_MODE"] = input_mode
    df["COLOR_MODE"] = color_mode
    df["MAX_LR"] = max_lr
    df["WEIGHT_DECAY"] = weight_decay

    # TRAIN 側データセット情報
    df["TRAIN_DATASET_ROOT"] = train_root_short
    df["TRAIN_BLUR"] = train_blur
    df["TRAIN_BLUR_KERNEL"] = train_blur_kernel

    # VALID 側データセット情報
    df["VALID_DATASET_ROOT"] = valid_root_short
    df["VALID_BLUR"] = valid_blur
    df["VALID_BLUR_KERNEL"] = valid_blur_kernel

    return df


# ============================================================
# メイン：全 exp* を走査して一つのCSVにまとめる
#   - dataset名は各expごとに自動判定（VALID_DIRベース）
#   - TRAIN/VALIDごとの blur 情報もCSVに保存
#   - 出力先: lab_logs/compilation/analysis-XX/comp_results_<split>_kf.csv
# ============================================================
def main():
    args = parse_args()
    log_root = Path(args.log_root)
    if not log_root.exists():
        print(f"[ERROR] log_root not found: {log_root}")
        return

    # exp* ディレクトリ一覧取得
    exp_dirs = sorted(
        [p for p in log_root.iterdir() if p.is_dir() and p.name.startswith("exp")]
    )
    if not exp_dirs:
        print(f"[WARN] No exp* directories found under {log_root}")
        return

    # exp_ids / exp_from / exp_to による絞り込み
    exp_dirs = filter_exp_dirs(exp_dirs, args)
    if not exp_dirs:
        print("[WARN] 条件にマッチする exp ディレクトリがありません。処理を終了します。")
        return

    split = args.split
    print(f"Compiling results for split='{split}' from {len(exp_dirs)} experiments")

    all_dfs: list[pd.DataFrame] = []
    for exp_dir in exp_dirs:
        print(f"\n=== Processing {exp_dir.name} ===")
        df = load_one_exp(exp_dir, split)
        if df is not None and not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("[WARN] 有効な eval_results_kf.csv が見つかりませんでした。Nothing to compile.")
        return

    df_all = pd.concat(all_dfs, axis=0, ignore_index=True)

    # -------- 出力ディレクトリを決定: compilation/analysis-XX --------
    analysis_name = get_analysis_name(args)
    compilation_dir = log_root / "compilation" / f"analysis-{analysis_name}"
    compilation_dir.mkdir(parents=True, exist_ok=True)

    out_csv = compilation_dir / f"comp_results_{split}_kf.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\n✅ Saved compiled CSV: {out_csv}")
    print(f"  Rows: {len(df_all)}")

if __name__ == "__main__":
    main()
