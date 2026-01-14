#!/usr/bin/env python3
# scripts/datasets_converter.py
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import shutil

SIZES    = [56, 112, 224]
CHANNELS  = ["rgb", "gray", "bin4"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

INTERPOLATION_MAP = {
    "nearest": cv2.INTER_NEAREST,
    "linear":  cv2.INTER_LINEAR,
    "area":    cv2.INTER_AREA,
    "cubic":   cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}

def max_pool_resize(img, target_size):
    h, w = img.shape[:2]
    th, tw = target_size
    pad_h = (th - h % th) % th
    pad_w = (tw - w % tw) % tw
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    H, W = img_padded.shape[:2]
    sh, sw = H // th, W // tw
    if img.ndim == 2:
        pooled = img_padded.reshape(th, sh, tw, sw).max(axis=(1, 3))
    else:
        pooled = img_padded.reshape(th, sh, tw, sw, 3).max(axis=(1, 3))
    return pooled.astype(img.dtype)

def resize_square(img: np.ndarray, size: int, method: str) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    if method == "maxpool":
        resized = max_pool_resize(img, (nh, nw))
    else:
        interpolation = INTERPOLATION_MAP[method]
        resized = cv2.resize(img, (nw, nh), interpolation=interpolation)

    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def to_gray(img_bgr): return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
def to_bin4(gray): return (np.digitize(gray, [3, 30, 120], right=False) * 85).astype(np.uint8)

def save_if_not_exists(img: np.ndarray, path: Path):
    if path.exists():
        return  # スキップ
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

def process_image(path: Path, raw_root: Path, cache_root: Path, resize_mode: str):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] 読み込み失敗: {path}")
        return

    split = path.parts[-3]
    if split not in {"train", "valid"}:
        print(f"[SKIP] train/valid 以外: {path}")
        return

    stem = path.stem
    gray = to_gray(img)
    bin4 = to_bin4(gray)

    for sz in SIZES:
        postfix = f"sz{sz}_{resize_mode}"
        for ch, data in zip(
            ["rgb", "gray", "bin4"],
            [img, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.cvtColor(bin4, cv2.COLOR_GRAY2BGR)]
        ):
            base = cache_root / ch / postfix / split / "imgs"
            out_path = base / f"{stem}.png"
            resized = resize_square(data, sz, resize_mode)
            save_if_not_exists(resized, out_path)

def copy_labels(raw_root: Path, cache_root: Path, resize_mode: str):
    for split in ("train", "valid"):
        src = raw_root / split / "labels.csv"
        if not src.exists():
            print(f"[WARN] labels.csv が見つかりません: {src}")
            continue
        for sz in SIZES:
            postfix = f"sz{sz}_{resize_mode}"
            for ch in CHANNELS:
                dst_dir = cache_root / ch / postfix / split
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst_dir / "labels.csv")

def convert(dataset_dir: Path, resize_mode: str):
    raw_root = dataset_dir / "raw"
    cache_root = dataset_dir / "cache"
    imgs = [p for p in raw_root.rglob("*") if p.suffix.lower() in IMG_EXTS]

    print(f"対象画像: {len(imgs)} 枚")
    for p in tqdm(imgs, desc=f"Converting ({resize_mode})"):
        process_image(p, raw_root, cache_root, resize_mode)

    copy_labels(raw_root, cache_root, resize_mode)
    print("✅ 完了")

def main():
    ap = argparse.ArgumentParser(
        description="Convert raw images to rgb/gray/bin4 caches with specified resize method."
    )
    ap.add_argument("dataset_dir", type=Path, help="datasets/type-1 など")
    ap.add_argument("--resize-mode", type=str, default="maxpool", choices=["nearest", "linear", "area", "cubic", "lanczos", "maxpool"],
                    help="リサイズ手法（default: area）")
    args = ap.parse_args()
    convert(args.dataset_dir.resolve(), args.resize_mode)

if __name__ == "__main__":
    main()
