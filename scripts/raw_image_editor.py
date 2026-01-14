#!/usr/bin/env python3
# batch_crop_circle.py
# -------------------------------------------
# 1) <folder>/0000.jpg から円を検出
# 2) その中心でフォルダ内の全画像をクロップ
# 3) <folder>_crop/ に同名で保存
# -------------------------------------------

import cv2
import numpy as np
import math
from pathlib import Path
import argparse


# ---------- 円検出（0000.jpg にだけ実行） ----------
def detect_circle_center_radius(img_path: Path,
                                blur_ksize: int = 5,
                                canny_low: int = 50,
                                canny_high: int = 150):
    """最大面積輪郭に外接する円を返す (cx, cy, radius)"""
    color = cv2.imread(str(img_path))
    if color is None:
        raise FileNotFoundError(f"画像が読み込めません: {img_path}")
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges   = cv2.Canny(blurred, canny_low, canny_high)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("円が検出できませんでした。")

    largest = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(largest)
    return int(round(x)), int(round(y)), int(round(r))


# ---------- 正方形クロップ ----------
def crop_square(img, center, side):
    """画像 img を中心 center, 一辺 side の正方形でクロップして返す"""
    h, w = img.shape[:2]
    cx, cy = center
    half = side // 2

    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + side, y1 + side

    # はみ出し調整
    if x1 < 0:
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > w:
        shift = x2 - w; x1 -= shift; x2 = w
        if x1 < 0: x1 = 0
    if y2 > h:
        shift = y2 - h; y1 -= shift; y2 = h
        if y1 < 0: y1 = 0

    return img[y1:y2, x1:x2]


# ---------- メイン処理 ----------
def main(folder_path: Path):
    src_folder  = folder_path
    dst_folder  = folder_path.with_name(folder_path.name + "_crop")
    dst_folder.mkdir(parents=True, exist_ok=True)

    ref_img = src_folder / "0000.png"
    cx, cy, radius = detect_circle_center_radius(ref_img)
    margin_scale = 3.3
    side = int(math.ceil(2 * radius * margin_scale))

    print(f"[INFO] 検出中心: ({cx}, {cy}), 半径: {radius}, クロップ一辺: {side}px")
    print(f"[INFO] 保存先フォルダ: {dst_folder}")

    # 対象拡張子（必要に応じて追加）
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for img_path in src_folder.iterdir():
        if img_path.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 読み込めないファイルをスキップ: {img_path.name}")
            continue

        cropped = crop_square(img, (cx, cy), side)
        cv2.imwrite(str(dst_folder / img_path.name), cropped)
        print(f"  -> {img_path.name} をクロップして保存")

    print("[DONE] すべて完了しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="フォルダ内画像を円中心基準で正方形クロップ")
    parser.add_argument("folder", type=str,
                        help="処理対象フォルダ（0000.jpg を含む）")
    args = parser.parse_args()

    main(Path(args.folder).resolve())
# -------------------------------------------