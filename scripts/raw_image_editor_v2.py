#!/usr/bin/env python3
# batch_crop_center.py
# -------------------------------------------
# フォルダ内の全画像を「画像中心」基準で正方形クロップし
# <folder>_crop/ に同名で保存
# -------------------------------------------

import cv2
from pathlib import Path
import argparse


# ---------- 正方形クロップ ----------
def crop_square(img, center, side):
    """画像 img を中心 center, 一辺 side の正方形でクロップして返す"""
    h, w = img.shape[:2]
    cx, cy = center
    half = side // 2

    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + side, y1 + side

    # はみ出し調整（サイズを維持しつつ範囲をスライド）
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        shift = x2 - w
        x1 -= shift
        x2 = w
        if x1 < 0:
            x1 = 0
    if y2 > h:
        shift = y2 - h
        y1 -= shift
        y2 = h
        if y1 < 0:
            y1 = 0

    return img[y1:y2, x1:x2]


def main(folder_path: Path, side: int):
    src_folder = folder_path
    dst_folder = folder_path.with_name(folder_path.name + "_crop")
    dst_folder.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 入力フォルダ: {src_folder}")
    print(f"[INFO] 保存先フォルダ: {dst_folder}")
    print(f"[INFO] クロップ一辺: {side}px（画像中心基準）")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for img_path in src_folder.iterdir():
        if img_path.suffix.lower() not in exts:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 読み込めないファイルをスキップ: {img_path.name}")
            continue

        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2  # ★画像中心
        cropped = crop_square(img, (cx, cy), side)

        cv2.imwrite(str(dst_folder / img_path.name), cropped)
        print(f"  -> {img_path.name} をクロップして保存（center=({cx},{cy})）")

    print("[DONE] すべて完了しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="フォルダ内画像を画像中心基準で正方形クロップ"
    )
    parser.add_argument("folder", type=str, help="処理対象フォルダ")
    parser.add_argument("--side", type=int, required=True,
                        help="クロップする正方形の一辺[pixel]（例: 800）")
    args = parser.parse_args()

    main(Path(args.folder).resolve(), args.side)
