#!/usr/bin/env python3
"""
augment_cache_dataset.py
------------------------
中央 X 固定・縦スライド BOX で部分隠しを行う拡張スクリプト。

■ 入力 (--path) は柔軟
    datasets/                               → 全 type-* を処理
    datasets/type-3                         → その type-3 だけ
    datasets/type-3/cache/rgb               → その channel 以下
    datasets/type-3/cache/rgb/sz224_area    → そのサイズ 1 個

■ 出力（必ず datasets 直下）
    datasets/type-3_occ_<suffix>/cache/...
          <suffix> = gray / black / white / noise / custom
"""
from __future__ import annotations
import argparse, random
from pathlib import Path
import cv2, numpy as np, pandas as pd
from tqdm import tqdm
import re


# ────── 色パース ─────────────────────────────────────────
def parse_color(arg):
    """戻り値: (mode, color_tuple_or_None, suffix_str)"""
    kw = arg[0].lower()
    if kw in ("gray", "black", "white", "noise"):
        table = {"gray": (128,), "black": (0,), "white": (255,)}
        return ("noise", None, "noise") if kw == "noise" else ("solid", table[kw], kw)
    nums = tuple(int(x) for x in arg)
    if len(nums) not in (1, 3):
        raise ValueError("--color はキーワードか数値 1〜3 個")
    return "solid", nums, "custom"


# ────── BOX 塗り ────────────────────────────────────────
def positions(h, side, k):
    return [(h-side)//2] if k == 1 else [int(round(i*(h-side)/(k-1))) for i in range(k)]


def fill(img, x1, y1, side, mode, col):
    x2, y2 = x1+side, y1+side
    if mode == "solid":
        if img.ndim == 2:
            img[y1:y2, x1:x2] = col[0]
        else:
            img[y1:y2, x1:x2] = col if len(col) == 3 else (col[0],)*3
    else:  # noise
        img[y1:y2, x1:x2] = np.random.randint(0, 256, img[y1:y2, x1:x2].shape, dtype=img.dtype)


# ────── split(train/valid) 拡張 ─────────────────────────
def augment_split(src_split: Path, dst_split: Path,
                  num_aug: int, frac: float, mode: str, col):
    img_dir, csv_path = src_split/"imgs", src_split/"labels.csv"
    if not img_dir.is_dir() or not csv_path.is_file():
        return False

    dst_img = dst_split/"imgs"
    dst_img.mkdir(parents=True, exist_ok=True)

    # 元画像コピー
    for p in img_dir.glob("*.png"):
        cv2.imwrite(str(dst_img/p.name),
                    cv2.imread(str(p), cv2.IMREAD_UNCHANGED))

    df = pd.read_csv(csv_path); aug_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=str(src_split.relative_to(src_split.parents[3])), leave=False):
        im = cv2.imread(str(img_dir/row["filename"]), cv2.IMREAD_UNCHANGED)
        if im is None: continue
        h, w = im.shape[:2]; side = int(frac*min(h, w)); x1 = (w-side)//2
        for i, y1 in enumerate(positions(h, side, num_aug)):
            aug = im.copy(); fill(aug, x1, y1, side, mode, col)
            new_name = f"{Path(row['filename']).stem}_occ{i}.png"
            cv2.imwrite(str(dst_img/new_name), aug)
            aug_rows.append({"filename": new_name,
                             "roll": row["roll"], "pitch": row["pitch"], "yaw": row["yaw"]})
    pd.concat([df, pd.DataFrame(aug_rows)], ignore_index=True).to_csv(
        dst_split/"labels.csv", index=False)
    return True


# ────── szXYZ_area 列挙 ───────────────────────────────
def gather_sz_dirs(top: Path):
    # szXYZ_area
    if (top/"train").is_dir() or (top/"valid").is_dir(): return [top]
    # channel
    if any(p.is_dir() and p.name.startswith("sz") for p in top.iterdir()):
        return [p for p in top.iterdir() if p.is_dir()]
    # type-x
    if (top/"cache").is_dir():
        return [sz for ch in (top/"cache").iterdir() for sz in ch.iterdir() if sz.is_dir()]
    # datasets
    return [sz for t in top.glob("type-*")
              for ch in (t/"cache").iterdir()
              for sz in ch.iterdir() if sz.is_dir()]

def split_type_rel(sz_path: Path) -> tuple[Path, Path]:
    """
    sz_path (= …/cache/<channel>/szXYZ_area) から
       type_dir = …/type-x
       rel_path = <channel>/szXYZ_area
    を返す。階層が深くても（train/valid 指定でも）OK。
    """
    # ① cache ディレクトリを探す
    cache_dir = next((p for p in sz_path.parents if p.name == "cache"), None)
    if cache_dir is None:
        raise ValueError(f"'cache' が見つかりません: {sz_path}")

    # ② type-x を探す（cache の 1 つ上に必ずある前提）
    type_dir = cache_dir.parent
    if not re.match(r"type-\d+", type_dir.name):
        raise ValueError(f"type-* フォルダが不正: {type_dir}")

    # ③ rel_path = channel/szXYZ_area
    rel_path = sz_path.relative_to(cache_dir)
    return type_dir, rel_path

# ────── main ──────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True,
        help="datasets 〜 szXYZ_area まで柔軟に指定")
    ap.add_argument("--num_aug", type=int, default=5)
    ap.add_argument("--box_frac", type=float, default=0.15)
    ap.add_argument("--color", nargs="+", default=["gray"],
        help='"gray" "black" "white" "noise" か数値 1〜3 個')
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    mode, col, suffix = parse_color(args.color)
    sz_dirs = gather_sz_dirs(Path(args.path).resolve())

    for sz in sz_dirs:
        type_dir, rel_path = split_type_rel(sz)          # ★ 変更
        ds_root   = type_dir.parent                      # datasets/
        out_cache = ds_root / f"{type_dir.name}_occ_{suffix}" / "cache"
        out_sz    = out_cache / rel_path                 # channel/szXYZ_area

        for split in ("train", "valid"):
            augment_split(sz/split, out_sz/split,
                          num_aug=args.num_aug, frac=args.box_frac,
                          mode=mode, col=col)


if __name__ == "__main__":
    main()
