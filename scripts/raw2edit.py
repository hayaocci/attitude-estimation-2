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
    """Max poolingによるリサイズ（修正版）"""
    h, w = img.shape[:2]
    th, tw = target_size
    
    # スケールファクターを計算
    scale_h, scale_w = th / h, tw / w
    
    if scale_h >= 1.0 and scale_w >= 1.0:
        # 拡大の場合は通常の補間を使用
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_NEAREST)
    
    # 縮小の場合のみmax pooling
    # プーリングサイズを計算
    pool_h = max(1, int(h / th))
    pool_w = max(1, int(w / tw))
    
    # 実際の出力サイズを計算
    out_h = h // pool_h
    out_w = w // pool_w
    
    if img.ndim == 2:
        # グレースケール
        pooled = np.zeros((out_h, out_w), dtype=img.dtype)
        for i in range(out_h):
            for j in range(out_w):
                region = img[i*pool_h:(i+1)*pool_h, j*pool_w:(j+1)*pool_w]
                pooled[i, j] = np.max(region)
    else:
        # カラー
        pooled = np.zeros((out_h, out_w, img.shape[2]), dtype=img.dtype)
        for i in range(out_h):
            for j in range(out_w):
                region = img[i*pool_h:(i+1)*pool_h, j*pool_w:(j+1)*pool_w]
                pooled[i, j] = np.max(region, axis=(0, 1))
    
    # 目標サイズと異なる場合は最終調整
    if (out_h, out_w) != (th, tw):
        pooled = cv2.resize(pooled, (tw, th), interpolation=cv2.INTER_NEAREST)
    
    return pooled

def resize_square(img: np.ndarray, size: int, method: str) -> np.ndarray:
    """アスペクト比を保持して正方形にリサイズ（561x561対応版）"""
    h, w = img.shape[:2]
    is_grayscale = img.ndim == 2
    
    # 既に正方形で目標サイズと同じ場合はそのまま返す
    if h == w == size:
        return img
    
    # アスペクト比を保持したリサイズ（高精度計算）
    if h == w:
        # 正方形の場合は直接リサイズ
        if method == "maxpool":
            resized = max_pool_resize(img, (size, size))
        else:
            interpolation = INTERPOLATION_MAP[method]
            resized = cv2.resize(img, (size, size), interpolation=interpolation)
        return resized
    else:
        # 長方形の場合
        scale = size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        
        # リサイズ実行
        if method == "maxpool":
            resized = max_pool_resize(img, (nh, nw))
        else:
            interpolation = INTERPOLATION_MAP[method]
            resized = cv2.resize(img, (nw, nh), interpolation=interpolation)
        
        # 実際のリサイズ後のサイズを取得（OpenCVの実装による微調整対応）
        actual_h, actual_w = resized.shape[:2]
        
        # 正方形キャンバス作成（元画像の次元に合わせる）
        if is_grayscale:
            canvas = np.zeros((size, size), dtype=img.dtype)
        else:
            canvas = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
        
        # 中央配置の座標計算（実際のサイズを使用）
        y0 = (size - actual_h) // 2
        x0 = (size - actual_w) // 2
        
        # 境界チェック
        y1 = min(y0 + actual_h, size)
        x1 = min(x0 + actual_w, size)
        actual_h = y1 - y0
        actual_w = x1 - x0
        
        # 画像を中央に配置
        canvas[y0:y1, x0:x1] = resized[:actual_h, :actual_w]
        
        return canvas

def to_gray(img_bgr): 
    """BGR画像をグレースケールに変換"""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def to_bin4(gray): 
    """グレースケール画像を4値化"""
    # return (np.digitize(gray, [10, 30, 120], right=False) * 85).astype(np.uint8)
    return (np.digitize(gray, [10, 30, 120], right=False) * 85).astype(np.uint8)


def save_if_not_exists(img: np.ndarray, path: Path):
    """ファイルが存在しない場合のみ保存"""
    if path.exists():
        return  # スキップ
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

def process_image(path: Path, raw_root: Path, cache_root: Path, resize_mode: str):
    """単一画像の処理"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] 読み込み失敗: {path}")
        return

    split = path.parts[-3]
    if split not in {"train", "valid"}:
        print(f"[SKIP] train/valid 以外: {path}")
        return

    stem = path.stem
    
    # 各チャンネル形式の画像を準備
    gray = to_gray(img)
    bin4 = to_bin4(gray)
    
    # 各サイズとチャンネルの組み合わせで処理
    for sz in SIZES:
        postfix = f"sz{sz}_{resize_mode}"
        
        # RGB
        base_rgb = cache_root / "rgb" / postfix / split / "imgs"
        out_path_rgb = base_rgb / f"{stem}.png"
        resized_rgb = resize_square(img, sz, resize_mode)
        save_if_not_exists(resized_rgb, out_path_rgb)
        
        # Grayscale
        base_gray = cache_root / "gray" / postfix / split / "imgs"
        out_path_gray = base_gray / f"{stem}.png"
        resized_gray = resize_square(gray, sz, resize_mode)
        # グレースケールをBGRに変換してから保存
        resized_gray_bgr = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2BGR)
        save_if_not_exists(resized_gray_bgr, out_path_gray)
        
        # Binary 4-level
        base_bin4 = cache_root / "bin4" / postfix / split / "imgs"
        out_path_bin4 = base_bin4 / f"{stem}.png"
        resized_bin4 = resize_square(bin4, sz, resize_mode)
        # 4値化画像をBGRに変換してから保存
        resized_bin4_bgr = cv2.cvtColor(resized_bin4, cv2.COLOR_GRAY2BGR)
        save_if_not_exists(resized_bin4_bgr, out_path_bin4)

def copy_labels(raw_root: Path, cache_root: Path, resize_mode: str):
    """ラベルファイルをコピー"""
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
    """メイン変換処理"""
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
    ap.add_argument("--resize-mode", type=str, default="area", choices=["nearest", "linear", "area", "cubic", "lanczos", "maxpool"],
                    help="リサイズ手法（default: area）")
    args = ap.parse_args()
    convert(args.dataset_dir.resolve(), args.resize_mode)

if __name__ == "__main__":
    main()