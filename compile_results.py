#!/usr/bin/env python3
# code_C_compile_results.py
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import yaml

# ============================================================
# コマンドライン引数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "lab_logs/expXX/eval_<dataset>_<split>_kf/eval_results_kf.csv と "
            "config_used.yaml を集約して1つのCSVにまとめるスクリプト"
        )
    )
    parser.add_argument("--dataset", required=True, help="評価対象の datasets/ 配下のフォルダ名 (例: type-8)")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"],
                        help="評価に使った split 名 (default: valid)")
    parser.add_argument("--log_root", type=str, default="lab_logs",
                        help="lab_logs のルートディレクトリ (default: lab_logs)")
    return parser.parse_args()

# ============================================================
# TRAIN_DATASET_ROOT から datasets/XXX の XXX 部分だけ抜き出す
# ============================================================
def extract_dataset_name(train_root_value: str) -> str:
    # 例: "datasets/type-2_aug-v2" -> "type-2_aug-v2"
    p = Path(train_root_value)
    # 'datasets' 直下を想定
    if p.name and p.parent.name == "datasets":
        return p.name
    # 保険として "datasets/" でsplit
    if "datasets/" in train_root_value:
        return train_root_value.split("datasets/")[-1].strip("/")

    # うまくパースできない場合は、そのまま返す
    return train_root_value

# ============================================================
# 1つのexpの結果を読み込んで DataFrame を作る
# ============================================================
def load_one_exp(exp_dir: Path, dataset: str, split: str) -> pd.DataFrame | None:
    cfg_path = exp_dir / "config_used.yaml"
    eval_dir = exp_dir / f"eval_{dataset}_{split}_kf"
    eval_csv = eval_dir / "eval_results_kf.csv"

    if not cfg_path.exists():
        print(f"  [SKIP] config_used.yaml not found: {cfg_path}")
        return None
    if not eval_csv.exists():
        print(f"  [SKIP] eval_results_kf.csv not found: {eval_csv}")
        return None

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    try:
        df = pd.read_csv(eval_csv)
    except Exception as e:
        print(f"  [ERROR] Failed to read {eval_csv}: {e}")
        return None

    # 必要な項目だけ config から取り出す
    batch_size = cfg.get("BATCH_SIZE", None)
    dropout_p = cfg.get("DROPOUT_P", None)
    img_size = cfg.get("IMG_SIZE", [None, None])
    img_size_scalar = img_size[0] if isinstance(img_size, (list, tuple)) and img_size else img_size
    input_mode = cfg.get("INPUT_MODE", None)
    max_lr = cfg.get("MAX_LR", None)
    weight_decay = cfg.get("WEIGHT_DECAY", None)
    train_root_raw = cfg.get("TRAIN_DATASET_ROOT", "")
    train_root_short = extract_dataset_name(train_root_raw)

    exp_id = cfg.get("id", exp_dir.name)

    # 全行に同じ情報を付与
    df["exp_id"] = exp_id
    df["BATCH_SIZE"] = batch_size
    df["DROPOUT_P"] = dropout_p
    df["IMG_SIZE"] = img_size_scalar
    df["INPUT_MODE"] = input_mode
    df["MAX_LR"] = max_lr
    df["WEIGHT_DECAY"] = weight_decay
    df["TRAIN_DATASET_ROOT"] = train_root_short  # 短縮形 (XXX のみ)

    return df

# ============================================================
# メイン：全 exp* を走査して一つのCSVにまとめる
# ============================================================
def main():
    args = parse_args()
    log_root = Path(args.log_root)
    if not log_root.exists():
        print(f"[ERROR] log_root not found: {log_root}")
        return

    exp_dirs = sorted(
        [p for p in log_root.iterdir() if p.is_dir() and p.name.startswith("exp")]
    )
    if not exp_dirs:
        print(f"[WARN] No exp* directories found under {log_root}")
        return

    dataset = args.dataset
    split = args.split
    print(f"Compiling results for dataset='{dataset}', split='{split}'")

    all_dfs = []
    for exp_dir in exp_dirs:
        print(f"\n=== Processing {exp_dir.name} ===")
        df = load_one_exp(exp_dir, dataset, split)
        if df is not None and not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("[WARN] No valid eval_results_kf.csv found. Nothing to compile.")
        return

    df_all = pd.concat(all_dfs, axis=0, ignore_index=True)

    compilation_dir = log_root / "compilation" / dataset
    compilation_dir.mkdir(parents=True, exist_ok=True)
    out_csv = compilation_dir / f"comp_{dataset}_{split}_kf.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\n✅ Saved compiled CSV: {out_csv}")
    print(f"  Rows: {len(df_all)}")

if __name__ == "__main__":
    main()
