#!/usr/bin/env python3
"""
augment_cache_dataset_v2.py — 21‑variant Data‑Augmentation Generator
===================================================================
* 画像 1 枚 → 最大 21 枚のバリエーションを生成
  ── ① オリジナル
  ── ② クロップ + 貼り付け
  ── ③ ペッパーノイズ
  ── ④ ブライトネス
  ── ⑤ ガウシアンブラー
  ── ⑥ ブライトネス→ペッパーノイズ
  ── ⑦ ブライトネス→ガウシアンブラー
  ── ⑧ 各①–⑦ + 極端隠し (4 分割の 1–3 区画を黒塗り)
  ── ⑨ 各①–⑦ + BBOX 塗りつぶし
       ※ パイプライン ② については「固定 2 箱 → クロップ → ランダム 1–3 箱」の順

CLI:
  python augment_cache_dataset_v2.py --path <dir> [--out <dir>] \
        [--test] [--seed 123] [--pepper_p 0.01] [--bright 0.9 1.1] \
        [--blur_k 3 7] [--rand_boxes 1 3] [--rand_box_wh 20 60] \
        [--rand_box_area 32 32 192 192] [--hide_n 1 3] [--crop_scale 0.5 1.0]

If --test is supplied, *exactly 5* images are sampled (per split) and only
those are augmented (resulting in 105 output images).
"""
from __future__ import annotations
import argparse, random, math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# --------------------------------------------------
# ---------------- ユーティリティ -----------------

# ---------- 基本処理 ----------

def central_crop_resize(img: np.ndarray, scale_rng: Tuple[float, float], canvas_size: Tuple[int, int]) -> np.ndarray:
    """中心クロップ → 拡大 / 縮小 → 黒背景キャンバスに貼付
    * scale_rng : クロップサイズ / 元画像サイズの範囲 (min,max)
    * canvas_size: (W,H) 元画像と同じにすること
    * 貼り付け位置はランダム。見切れを許可 (50%以上は残す)
    """
    h, w = img.shape[:2]
    scale = random.uniform(*scale_rng)
    crop_w = int(w * scale)
    crop_h = int(h * scale)
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    crop = img[y1:y1 + crop_h, x1:x1 + crop_w]

    # ランダムリサイズ (±20% 程度の揺らぎ)
    factor = random.uniform(0.8, 1.2)
    new_w = max(1, int(crop_w * factor))
    new_h = max(1, int(crop_h * factor))
    crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros_like(img)
    cx_max = w - new_w
    cy_max = h - new_h
    # 見切れ可だが 50% 以上写す → 貼り付け範囲を少し広げつつ制約
    cx_min = -new_w // 2
    cy_min = -new_h // 2
    cx = random.randint(cx_min, cx_max)
    cy = random.randint(cy_min, cy_max)
    # 貼付け
    x_from = max(0, cx)
    y_from = max(0, cy)
    x_to = min(w, cx + new_w)
    y_to = min(h, cy + new_h)
    src_x1 = x_from - cx
    src_y1 = y_from - cy
    src_x2 = src_x1 + (x_to - x_from)
    src_y2 = src_y1 + (y_to - y_from)

    canvas[y_from:y_to, x_from:x_to] = crop_resized[src_y1:src_y2, src_x1:src_x2]
    return canvas


def add_pepper_noise(img: np.ndarray, p: float) -> np.ndarray:
    if p <= 0:
        return img
    out = img.copy()
    mask = np.random.rand(*out.shape[:2]) < p
    if out.ndim == 2:
        out[mask] = 0
    else:
        out[mask] = (0, 0, 0)
    return out


def adjust_brightness(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if lo == 1.0 and hi == 1.0:
        return img
    factor = random.uniform(lo, hi)
    out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(img.dtype)
    return out


def gaussian_blur(img: np.ndarray, k_rng: Tuple[int, int]) -> np.ndarray:
    k = random.randrange(k_rng[0] | 1, k_rng[1] | 1, 2)  # 奇数取得
    return cv2.GaussianBlur(img, (k, k), 0)


def hide_quadrants(img: np.ndarray, n_rng: Tuple[int, int]) -> np.ndarray:
    n = random.randint(*n_rng)
    if n == 0:
        return img
    out = img.copy()
    h_mid, w_mid = out.shape[0] // 2, out.shape[1] // 2
    quads = [ (slice(0, h_mid), slice(0, w_mid)),
              (slice(0, h_mid), slice(w_mid, None)),
              (slice(h_mid, None), slice(0, w_mid)),
              (slice(h_mid, None), slice(w_mid, None)) ]
    for idx in random.sample(range(4), n):
        r, c = quads[idx]
        out[r, c] = 0
    return out

# ---------- BBOX 関連 ----------

def rect_vertices(cx: float, cy: float, w: int, h: int):
    tl = (int(cx - w / 2), int(cy - h / 2))
    br = (int(cx + w / 2), int(cy + h / 2))
    return tl, br


def draw_fixed_bboxes(img: np.ndarray, roll_deg: float) -> np.ndarray:
    """規定 2 箱を条件付きで描画 (40×40)"""
    roll_deg %= 360
    roll_rad = math.radians(roll_deg)
    h, w = img.shape[:2]
    boxes = []
    if not (0 <= roll_deg <= 35 or 325 <= roll_deg <= 360):
        cx1, cy1 = w // 2, h // 2 - 50 * math.sin(roll_rad)
        boxes.append(rect_vertices(cx1, cy1, 40, 40))
    if not (135 <= roll_deg <= 225):
        cx2, cy2 = w // 2, h // 2 + 90 * math.sin(roll_rad)
        boxes.append(rect_vertices(cx2, cy2, 40, 40))
    return _fill_boxes(img, boxes)


def gen_random_boxes(num_rng: Tuple[int, int], wh_rng: Tuple[int, int], area: Tuple[int, int, int, int], existing: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    tl_area_x, tl_area_y, br_area_x, br_area_y = area
    boxes = []
    trials = 50
    num_target = random.randint(*num_rng)
    while len(boxes) < num_target and trials:
        trials -= 1
        w = random.randint(*wh_rng)
        h = random.randint(*wh_rng)
        x1 = random.randint(tl_area_x, br_area_x - w)
        y1 = random.randint(tl_area_y, br_area_y - h)
        tl, br = (x1, y1), (x1 + w, y1 + h)
        if _is_overlapping((tl, br), existing + boxes):
            continue
        boxes.append((tl, br))
    return boxes


def _is_overlapping(box: Tuple[Tuple[int, int], Tuple[int, int]], others):
    (ax1, ay1), (ax2, ay2) = box
    for (bx1, by1), (bx2, by2) in others:
        if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
            return True
    return False


def _fill_boxes(img: np.ndarray, boxes):
    out = img.copy()
    for (x1, y1), (x2, y2) in boxes:
        sub = out[y1:y2, x1:x2]
        if sub.size == 0:
            continue
        sub[:] = 0  # 黒塗り
    return out


def draw_random_bboxes(img: np.ndarray, num_rng: Tuple[int, int], wh_rng: Tuple[int, int], area: Tuple[int, int, int, int], existing):
    boxes = gen_random_boxes(num_rng, wh_rng, area, existing)
    return _fill_boxes(img, boxes)

# --------------------------------------------------
# ---------------- パイプライン -------------------

BASE_PIPELINES = {
    "orig": [],
    "crop": ["crop"],
    "pepper": ["pepper"],
    "bright": ["bright"],
    "blur": ["blur"],
    "bright_pepper": ["bright", "pepper"],
    "bright_blur": ["bright", "blur"],
}

HIDE_SUFFIX = "hide"
BBOX_SUFFIX = "bbox"

# --------------------------------------------------


def build_augmented_images(img: np.ndarray, roll: float, args, stem: str, out_dir: Path):
    h, w = img.shape[:2]
    area_full = (0, 0, w, h)
    wh_rng = (args.rand_box_wh[0], args.rand_box_wh[1])
    num_rng = (args.rand_boxes[0], args.rand_boxes[1])
    hide_n_rng = (args.hide_n[0], args.hide_n[1])

    for pid, ops in BASE_PIPELINES.items():
        aug = img.copy()
        # --- 基本 7 パイプライン ---
        aug = apply_ops(aug, roll, ops, pid, args)
        cv2.imwrite(str(out_dir / f"{stem}_{pid}.png"), aug)

        # --- 極端隠し ⑧ ---
        hide_img = hide_quadrants(aug, hide_n_rng)
        cv2.imwrite(str(out_dir / f"{stem}_{pid}_{HIDE_SUFFIX}.png"), hide_img)

        # --- BBOX ⑨ ---
        bbox_img = apply_bboxes_full(aug, roll, pid, num_rng, wh_rng, area_full, args)
        cv2.imwrite(str(out_dir / f"{stem}_{pid}_{BBOX_SUFFIX}.png"), bbox_img)


def apply_ops(img, roll, ops: List[str], pid: str, args):
    for op in ops:
        if op == "crop":
            img = central_crop_resize(img, args.crop_scale, (img.shape[1], img.shape[0]))
        elif op == "pepper":
            img = add_pepper_noise(img, args.pepper_p)
        elif op == "bright":
            img = adjust_brightness(img, args.bright[0], args.bright[1])
        elif op == "blur":
            img = gaussian_blur(img, args.blur_k)
    return img


def apply_bboxes_full(img: np.ndarray, roll: float, pid: str, num_rng, wh_rng, area, args):
    out = img.copy()
    fixed_boxes = []
    if pid == "crop":
        # (a) 規定 2 箱
        out = draw_fixed_bboxes(out, roll)
    # (c) ランダム箱
    out = draw_random_bboxes(out, num_rng, wh_rng, area, fixed_boxes)
    return out

# --------------------------------------------------
# ---------------- メイン --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="画像が入ったフォルダ (imgs/) 直下まで")
    ap.add_argument("--out", default="aug_out", help="出力先ディレクトリ")
    ap.add_argument("--test", action="store_true", help="テストモード: ランダム 5 枚のみ拡張")
    ap.add_argument("--seed", type=int, default=42)

    # 変換パラメータ
    ap.add_argument("--pepper_p", type=float, default=0.01)
    ap.add_argument("--bright", nargs=2, type=float, default=[0.9, 1.1])
    ap.add_argument("--blur_k", nargs=2, type=int, default=[3, 7])
    ap.add_argument("--rand_boxes", nargs=2, type=int, default=[1, 3])
    ap.add_argument("--rand_box_wh", nargs=2, type=int, default=[20, 60])
    ap.add_argument("--rand_box_area", nargs=4, type=int, default=[0, 0, 224, 224])
    ap.add_argument("--hide_n", nargs=2, type=int, default=[1, 3])
    ap.add_argument("--crop_scale", nargs=2, type=float, default=[0.3, 1.0])
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    img_dir = Path(args.path).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if not img_paths:
        print("[ERROR] no images found")
        return

    if args.test and len(img_paths) >= 5:
        img_paths = random.sample(img_paths, 5)
        print(f"[INFO] TEST MODE: {len(img_paths)} images will be augmented → {len(img_paths)*21} files")
    else:
        print(f"[INFO] {len(img_paths)} images ×21 will be generated → {len(img_paths)*21}")

    for p in tqdm(img_paths, desc="Augmenting"):
        stem = p.stem
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        # roll 角度を 0 に固定 (BBOX 条件に使用)。実際は CSV などと同期して渡す
        build_augmented_images(img, roll=0.0, args=args, stem=stem, out_dir=out_dir)

    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
