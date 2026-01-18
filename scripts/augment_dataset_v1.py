#!/usr/bin/env python3
"""
augment_sz_area_v5_batch_parallel.py
---------------------
* 並列処理（Multiprocessing）対応版
* 指定した type-X 配下の cache を一括処理
* 出力フォルダ名を type-X_aug-vN として自動連番
* ベース変換のみの保存可否をフラグで制御可能
* [NEW] stretch（縦横引き伸ばし）変換をベース変換に追加
* [NEW] 実行時のデータ拡張設定を augment_config.yaml として保存
* [NEW] --cache_subdirs により cache/rgb, cache/gray, cache/bin4 など対象サブフォルダを指定可能
"""

from __future__ import annotations
import argparse, random, math, csv, re, os
from pathlib import Path
from typing import List, Tuple, Dict, Callable
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import yaml  # augment_config.yaml 出力用

# ──────────────────────────────────────────────────────────────
# ユーザー設定セクション
# ──────────────────────────────────────────────────────────────
class CONFIG:
    # 1. ベース変換の選択
    # 追加した "stretch" を含めています
    # SELECTED_BASES = ["iso_noise", "blur", "bright", "vstrip", "stretch"]
    # SELECTED_BASES = ["iso_noise", "blur", "vstrip", "stretch"]
    SELECTED_BASES = ["blur"]
    # SELECTED_BASES = ["stretch"]

    # 2. 派生変換の有効化
    ENABLE_RBBOX  = False
    ENABLE_CROP   = False
    ENABLE_HIDE   = False

    # 3. 保存設定
    SAVE_BASE_TRANSFORMS = True

# ──────────────────────────────────────────────────────────────
# 変換ロジック
# ──────────────────────────────────────────────────────────────

def apply_stretch(img: np.ndarray, stretch_range: Tuple[float, float]) -> np.ndarray:
    """画像を縦または横にランダムに引き伸ばし、元のサイズにクロップする"""
    h, w = img.shape[:2]
    scale = random.uniform(*stretch_range)
    
    # 縦か横かをランダムに決定
    mode = random.choice(["vertical", "horizontal"])
    
    if mode == "vertical":
        new_h, new_w = int(h * scale), w
    else:
        new_h, new_w = h, int(w * scale)
    
    # リサイズ実行
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 元のサイズ(h, w)に合わせて中央をクロップ
    start_y = (new_h - h) // 2
    start_x = (new_w - w) // 2
    cropped = resized[start_y:start_y+h, start_x:start_x+w]
    
    # 計算誤差で1pxずれる場合の対応
    if cropped.shape[0] != h or cropped.shape[1] != w:
        cropped = cv2.resize(cropped, (w, h))
        
    return cropped

def apply_vstrip(img: np.ndarray, n_options: List[int], bright_range: Tuple[float, float], blend_ratio: float) -> np.ndarray:
    h, w = img.shape[:2]
    n_strips = random.choice(n_options)
    bright_map = np.zeros((h, w), dtype=np.float32)
    edges = np.linspace(0, w, n_strips + 1).astype(int)
    offsets = []
    for i in range(n_strips):
        found = False
        for _ in range(100):
            val = random.uniform(*bright_range)
            if random.random() < 0.5:
                val = -val
            if i == 0 or abs(val - offsets[i-1]) >= 32:
                offsets.append(val)
                found = True
                break
        if not found:
            offsets.append(val)
    for i in range(n_strips):
        x_start, x_end = edges[i], edges[i+1]
        offset = offsets[i]
        bright_map[:, x_start:x_end] = offset
        if blend_ratio > 0 and i < n_strips - 1:
            strip_w = x_end - x_start
            blend_w = int(strip_w * blend_ratio)
            if blend_w > 0:
                next_offset = offsets[i+1]
                for dx in range(blend_w):
                    alpha = dx / blend_w
                    pos = x_end - (blend_w // 2) + dx
                    if 0 <= pos < w:
                        bright_map[:, pos] = (1 - alpha) * offset + alpha * next_offset
    img_f = img.astype(np.float32)
    res = img_f + (bright_map[:, :, np.newaxis] if img.ndim == 3 else bright_map)
    return np.clip(res, 0, 255).astype(np.uint8)

def add_iso_noise(img, sigma):
    if sigma <= 0:
        return img
    f = img.astype(np.float32) / 255.0
    noisy = np.clip(
        f + (np.random.poisson(f * 255) / 255 - f)
        + np.random.normal(0, sigma / 255, f.shape),
        0,
        1,
    )
    return (noisy * 255).astype(img.dtype)

def adjust_brightness(img, lo, hi):
    return np.clip(img.astype(np.float32) * random.uniform(lo, hi), 0, 255).astype(img.dtype)

def gaussian_blur(img, k_rng):
    k = random.randrange(k_rng[0] | 1, (k_rng[1] + 1) | 1, 2)
    return cv2.GaussianBlur(img, (k, k), 0)

def hide_quadrants(img, n_rng):
    n = random.randint(*n_rng)
    out = img.copy()
    h2, w2 = out.shape[0] // 2, out.shape[1] // 2
    quads = [
        (slice(0, h2), slice(0, w2)),
        (slice(0, h2), slice(w2, None)),
        (slice(h2, None), slice(0, w2)),
        (slice(h2, None), slice(w2, None)),
    ]
    for idx in random.sample(range(4), n):
        out[quads[idx]] = 0
    return out

def specific_crop_and_paste(img, crop, vis_thr=0.6):
    cx, cy, w, h = crop
    H, W = img.shape[:2]
    x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    cropped = img[cy1:cy2, cx1:cx2]
    if cropped.size == 0:
        return np.zeros_like(img)
    ch, cw = cropped.shape[:2]
    min_w, min_h = int(cw * vis_thr), int(ch * vis_thr)
    px, py = random.randint(-(cw - min_w), W - min_w), random.randint(
        -(ch - min_h), H - min_h
    )
    canvas = np.zeros_like(img)
    dx1, dy1 = max(0, px), max(0, py)
    dx2, dy2 = min(W, px + cw), min(H, py + ch)
    sx1, sy1 = dx1 - px, dy1 - py
    sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)
    canvas[dy1:dy2, dx1:dx2] = cropped[sy1:sy2, sx1:sx2]
    return canvas

# ──────────────────────────────────────────────────────────────
# BBOX ユーティリティ
# ──────────────────────────────────────────────────────────────
class BBoxScaler:
    def __init__(self, scale: float):
        self.C = int(round(112 * scale))
        self.O1 = int(round(50 * scale))
        self.O2 = int(round(90 * scale))
        self.BW = self.BH = int(round(40 * scale))

    def vertices(self, cx, cy):
        return (int(cx - self.BW / 2), int(cy - self.BH / 2)), (
            int(cx + self.BW / 2),
            int(cy + self.BH / 2),
        )

def get_fixed_bboxes(roll: float, scaler: BBoxScaler):
    roll %= 360
    rad = math.radians(roll)
    boxes = []
    if not (0 <= roll <= 35 or 325 <= roll <= 360):
        boxes.append(scaler.vertices(scaler.C, scaler.C - scaler.O1 * math.sin(rad)))
    if not (135 <= roll <= 225):
        boxes.append(scaler.vertices(scaler.C, scaler.C + scaler.O2 * math.sin(rad)))
    return boxes

def fill_roi_partial(img, tl, br, color):
    x1, y1 = tl
    x2, y2 = br
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    scale = math.sqrt(random.uniform(0.25, 0.75))
    rw, rh = int(w * scale), int(h * scale)
    rx, ry = random.randint(0, w - rw), random.randint(0, h - rh)
    sub = roi[ry : ry + rh, rx : rx + rw]
    if img.ndim == 2:
        sub[:] = color[0]
    else:
        sub[:] = color if len(color) == 3 else (color[0],) * 3

def apply_fixed_bboxes(img, roll, scaler, color_fn):
    out = img.copy()
    for tl, br in get_fixed_bboxes(roll, scaler):
        fill_roi_partial(out, tl, br, color_fn())
    return out

def apply_random_bboxes(img, roll, scaler, args, color_fn):
    out = img.copy()
    existing = get_fixed_bboxes(roll, scaler)
    tlx, tly, brx, bry = args.rand_box_area
    target = random.randint(*args.rand_boxes)
    boxes = []
    trials = 50
    while len(boxes) < target and trials > 0:
        trials -= 1
        w, h = random.randint(*args.rand_box_wh), random.randint(*args.rand_box_wh)
        x1 = random.randint(tlx, max(tlx, brx - w))
        y1 = random.randint(tly, max(tly, bry - h))
        box = ((x1, y1), (x1 + w, y1 + h))
        (ax1, ay1), (ax2, ay2) = box
        overlap = False
        for (bx1, by1), (bx2, by2) in existing + boxes:
            if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                overlap = True
                break
        if not overlap:
            boxes.append(box)
    for tl, br in boxes:
        fill_roi_partial(out, tl, br, color_fn())
    return out

# ──────────────────────────────────────────────────────────────
# 並列実行用ワーカー
# ──────────────────────────────────────────────────────────────

BASE_TRANSFORMS = {
    "orig": [],
    "iso_noise": ["iso_noise"],
    "blur": ["blur"],
    "bright": ["bright"],
    "bright_iso": ["bright", "iso_noise"],
    "bright_blur": ["bright", "blur"],
    "vstrip": ["vstrip"],
    "stretch": ["stretch"],  # stretchを追加
}
DERIVED_SUFFIXES = {"random_bbox": "rbbox", "crop": "crop", "hide": "hide"}

def apply_seq(img, ops, args):
    out = img
    for op in ops:
        if op == "iso_noise":
            out = add_iso_noise(out, args.iso_sigma)
        elif op == "bright":
            out = adjust_brightness(out, *args.bright)
        elif op == "blur":
            out = gaussian_blur(out, tuple(args.blur_k))
        elif op == "vstrip":
            out = apply_vstrip(
                out,
                args.vstrip_n,
                tuple(args.strip_bright_range),
                args.strip_blend_ratio,
            )
        elif op == "stretch":
            out = apply_stretch(out, tuple(args.stretch_range))  # stretchの実行
    return out

def process_single_image(
    img_path: Path,
    row_in: dict,
    scaler: BBoxScaler,
    args: argparse.Namespace,
    train_out_imgs: Path,
    f_col_key: str,
    r_col_key: str,
):
    """画像1枚を読み込んで全バリエーションを保存する関数（並列ワーカー）"""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return []

    # 色生成関数の再構築
    def get_color(kw):
        if kw == "black":
            return (0,)
        if kw == "white":
            return (255,)
        if kw == "gray":
            return (128,)
        if kw == "random_gray":
            return (random.randint(0, 255),)
        return tuple(random.randint(0, 255) for _ in range(3))

    f_col_fn = lambda: get_color(f_col_key)
    r_col_fn = lambda: get_color(r_col_key)

    roll = float(row_in.get("roll", 0))
    res_rows = []

    # ベース変換
    base_imgs = {}
    for n in CONFIG.SELECTED_BASES:
        if n in BASE_TRANSFORMS:
            base_imgs[n] = apply_seq(img, BASE_TRANSFORMS[n], args)

    # 派生変換と保存
    variants = {}
    variants.update(base_imgs)
    for n, b in base_imgs.items():
        fb = apply_fixed_bboxes(b, roll, scaler, f_col_fn)
        if CONFIG.ENABLE_RBBOX:
            variants[f"{n}_{DERIVED_SUFFIXES['random_bbox']}"] = apply_random_bboxes(
                fb, roll, scaler, args, r_col_fn
            )
        if CONFIG.ENABLE_CROP:
            variants[f"{n}_{DERIVED_SUFFIXES['crop']}"] = specific_crop_and_paste(
                fb, tuple(args.crop_params), args.visibility_threshold
            )
        if CONFIG.ENABLE_HIDE:
            variants[f"{n}_{DERIVED_SUFFIXES['hide']}"] = hide_quadrants(
                b, tuple(args.hide_n)
            )

    for tag, v in variants.items():
        if tag == "orig":
            continue
        if (not CONFIG.SAVE_BASE_TRANSFORMS) and tag in CONFIG.SELECTED_BASES:
            continue
        if np.count_nonzero(v) == 0:
            continue

        fn = f"{img_path.stem}_{tag}.png"
        cv2.imwrite(str(train_out_imgs / fn), v)

        new_row = row_in.copy()
        new_row["filename"] = fn
        res_rows.append(new_row)

    return res_rows

# ──────────────────────────────────────────────────────────────
# augment_config.yaml の作成
# ──────────────────────────────────────────────────────────────

def build_augment_config_dict(
    args: argparse.Namespace,
    in_base: Path,
    final_out_root: Path,
    targets_info: List[dict],
) -> dict:
    """augment_config.yaml 用の辞書を構築する"""
    # meta
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    script_name = Path(__file__).name if "__file__" in globals() else "augment_sz_area_v5_batch_parallel.py"

    meta = {
        "script": script_name,
        "created_at": created_at,
        "in_path": str(in_base),
        "out_root": str(Path(args.out_root).resolve()),
        "output_root": str(final_out_root),
        "seed": args.seed,
        "workers": args.workers,
        "test_mode": bool(args.test),
    }

    # config（CONFIG + コマンドライン引数）
    config = {
        "selected_bases": list(CONFIG.SELECTED_BASES),
        "enable_random_bbox": bool(CONFIG.ENABLE_RBBOX),
        "enable_crop": bool(CONFIG.ENABLE_CROP),
        "enable_hide": bool(CONFIG.ENABLE_HIDE),
        "save_base_transforms": bool(CONFIG.SAVE_BASE_TRANSFORMS),
        "fixed_color": args.fixed_color,
        "rand_color": args.rand_color,
        "cache_subdirs": list(args.cache_subdirs),
        "params": {
            "vstrip_n": list(args.vstrip_n),
            "strip_bright_range": [float(args.strip_bright_range[0]), float(args.strip_bright_range[1])],
            "strip_blend_ratio": float(args.strip_blend_ratio),
            "iso_sigma": float(args.iso_sigma),
            "bright": [float(args.bright[0]), float(args.bright[1])],
            "blur_k": [int(args.blur_k[0]), int(args.blur_k[1])],
            "stretch_range": [float(args.stretch_range[0]), float(args.stretch_range[1])],
            "rand_boxes": [int(args.rand_boxes[0]), int(args.rand_boxes[1])],
            "rand_box_wh_base": [int(args.rand_box_wh[0]), int(args.rand_box_wh[1])],
            "rand_box_area_base": [int(args.rand_box_area[0]), int(args.rand_box_area[1]),
                                   int(args.rand_box_area[2]), int(args.rand_box_area[3])],
            "hide_n": [int(args.hide_n[0]), int(args.hide_n[1])],
            "crop_params_base": [int(args.crop_params[0]), int(args.crop_params[1]),
                                 int(args.crop_params[2]), int(args.crop_params[3])],
            "visibility_threshold": float(args.visibility_threshold),
        },
        "bbox_scaler": {
            "reference_size": 224,
            "base_constants": {
                "C": 112,
                "O1": 50,
                "O2": 90,
                "BW": 40,
                "BH": 40,
            },
        },
    }

    # augmentations（ベース変換・派生変換の詳細）
    augmentations = {
        "base_transforms": {
            "iso_noise": {
                "enabled": "iso_noise" in CONFIG.SELECTED_BASES,
                "description": "Poisson + Gaussian ノイズを加える ISO ノイズ風の変換",
                "operations": [
                    {
                        "op": "iso_noise",
                        "params": {
                            "iso_sigma": float(args.iso_sigma),
                            "formula": "f + (Poisson(f*255)/255 - f) + Normal(0, iso_sigma/255)",
                        },
                    }
                ],
            },
            "blur": {
                "enabled": "blur" in CONFIG.SELECTED_BASES,
                "description": "ガウシアンぼかし",
                "operations": [
                    {
                        "op": "gaussian_blur",
                        "params": {
                            "kernel_size_range": [int(args.blur_k[0]), int(args.blur_k[1])],
                            "note": "奇数に丸めて使用（5,7,9,...）",
                        },
                    }
                ],
            },
            "vstrip": {
                "enabled": "vstrip" in CONFIG.SELECTED_BASES,
                "description": "縦長短冊ごとに輝度を変え、境界を線形ブレンドする",
                "operations": [
                    {
                        "op": "vstrip",
                        "params": {
                            "n_strips_options": list(args.vstrip_n),
                            "strip_bright_range": [float(args.strip_bright_range[0]), float(args.strip_bright_range[1])],
                            "sign_randomization": True,
                            "min_neighbor_diff": 32.0,
                            "blend_ratio": float(args.strip_blend_ratio),
                        },
                    }
                ],
            },
            "stretch": {
                "enabled": "stretch" in CONFIG.SELECTED_BASES,
                "description": "縦または横方向にランダムスケールし、中央でクロップして元サイズに戻す",
                "operations": [
                    {
                        "op": "stretch",
                        "params": {
                            "stretch_range": [float(args.stretch_range[0]), float(args.stretch_range[1])],
                            "mode": ["vertical", "horizontal"],
                            "crop": "center",
                            "resize_if_off_by_one_pixel": True,
                        },
                    }
                ],
            },
        },
        "derived_transforms": {
            "random_bbox": {
                "enabled": bool(CONFIG.ENABLE_RBBOX),
                "suffix": DERIVED_SUFFIXES["random_bbox"],
                "description": "固定BBOXで一部領域を塗った後、重ならないようにランダムBBOXを追加で塗りつぶす",
                "base_transforms_applied_to": list(CONFIG.SELECTED_BASES),
                "params": {
                    "rand_boxes_base": [int(args.rand_boxes[0]), int(args.rand_boxes[1])],
                    "rand_box_wh_base": [int(args.rand_box_wh[0]), int(args.rand_box_wh[1])],
                    "rand_box_area_base": [int(args.rand_box_area[0]), int(args.rand_box_area[1]),
                                           int(args.rand_box_area[2]), int(args.rand_box_area[3])],
                    "fixed_box_color_mode": args.fixed_color,
                    "random_box_color_mode": args.rand_color,
                    "overlap_avoid_trials": 50,
                },
            },
            "crop": {
                "enabled": bool(CONFIG.ENABLE_CROP),
                "suffix": DERIVED_SUFFIXES["crop"],
                "description": "指定した中心と幅高さで切り出して別位置にランダム貼り付け（可視割合の下限を保証）",
                "base_transforms_applied_to": list(CONFIG.SELECTED_BASES),
                "params": {
                    "crop_params_base": [int(args.crop_params[0]), int(args.crop_params[1]),
                                         int(args.crop_params[2]), int(args.crop_params[3])],
                    "visibility_threshold": float(args.visibility_threshold),
                    "fill_outside_region": "black",
                },
            },
            "hide": {
                "enabled": bool(CONFIG.ENABLE_HIDE),
                "suffix": DERIVED_SUFFIXES["hide"],
                "description": "画像を4分割し、そのうちランダムな複数クォドラントを塗りつぶす",
                "base_transforms_applied_to": list(CONFIG.SELECTED_BASES),
                "params": {
                    "hide_n": [int(args.hide_n[0]), int(args.hide_n[1])],
                    "fill_value": 0,
                    "split_mode": "quadrants",
                },
            },
        },
    }

    # file_naming
    file_naming = {
        "pattern": "{orig_stem}_{tag}.png",
        "base_tags": list(CONFIG.SELECTED_BASES),
        "derived_suffixes": dict(DERIVED_SUFFIXES),
        "examples": [
            {
                "original": "0000.png",
                "generated": [
                    "0000_iso_noise.png",
                    "0000_iso_noise_rbbox.png",
                    "0000_iso_noise_crop.png",
                    "0000_iso_noise_hide.png",
                    "0000_blur.png",
                    "0000_blur_rbbox.png",
                    "0000_vstrip.png",
                    "0000_vstrip_crop.png",
                    "0000_stretch_hide.png",
                ],
            }
        ],
    }

    cfg = {
        "meta": meta,
        "config": config,
        "augmentations": augmentations,
        "targets": targets_info,
        "file_naming": file_naming,
    }
    return cfg

# ──────────────────────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(
        description="type-X 配下の cache/szXXX_area/train をまとめてデータ拡張し、*_aug-vN フォルダとして出力するスクリプト"
    )

    # 入出力関連
    pa.add_argument(
        "--in_path",
        required=True,
        help="入力フォルダのパス。例: datasets/type-2 のような type-X ルート（直下に cache/ があるフォルダ）"
    )
    pa.add_argument(
        "--out_root",
        required=True,
        help="出力先ルートディレクトリ。ここに <in_path名>_aug-vN フォルダが自動作成される（例: datasets）"
    )

    # cache サブフォルダの指定
    pa.add_argument(
        "--cache_subdirs",
        nargs="+",
        default=["rgb", "gray", "bin4"],
        help="cache 配下でデータ拡張を行うサブフォルダ名リスト。例: rgb gray bin4。指定しない場合は3つすべてを処理する。"
    )

    # 色・乱数・テスト設定
    pa.add_argument(
        "--fixed_color",
        default="random_rainbow",
        help="固定BBOX（roll 依存の矩形）を塗りつぶすときの色モード。black/white/gray/random_gray など。"
    )
    pa.add_argument(
        "--rand_color",
        default="random_rainbow",
        help="ランダムBBOXを塗りつぶすときの色モード。black/white/gray/random_gray など。"
    )
    pa.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード値。再現性を確保したい場合は同じ値を指定する。"
    )
    pa.add_argument(
        "--test",
        action="store_true",
        help="テストモード。各 szXXX_area について先頭 10 枚のみ処理して動作確認する。"
    )
    pa.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="並列実行するワーカー数（デフォルト: CPU コア数）。大きくしすぎるとメモリを多く消費する。"
    )

    # ── データ拡張パラメータ ─────────────────────────
    # vstrip 関連
    pa.add_argument(
        "--vstrip_n",
        nargs="+",
        type=int,
        default=[2],
        help="vstrip で画像を縦方向に何分割にするかの候補リスト。ここから 1 つがランダムに選ばれる（例: 2,4,5 分割）。"
    )
    pa.add_argument(
        "--strip_bright_range",
        nargs=2,
        type=float,
        default=[30, 100],
        help="vstrip で各ストリップに加える輝度オフセットの絶対値レンジ [min, max]。正負はランダムに決まる。"
    )
    pa.add_argument(
        "--strip_blend_ratio",
        type=float,
        default=0.1,
        help="vstrip のストリップ境界を滑らかにするブレンド幅の割合。ストリップ幅に対する比率（例: 0.1 で幅の 10% をブレンド）。"
    )

    # ノイズ・輝度・ぼかし
    pa.add_argument(
        "--iso_sigma",
        type=float,
        default=8.0,
        help="ISO ノイズ風のガウシアン成分の標準偏差。値を大きくするとノイズが強くなる。"
    )
    pa.add_argument(
        "--bright",
        nargs=2,
        type=float,
        default=[0.7, 1.3],
        help="全体の明るさスケール係数のレンジ [min, max]。この範囲から一様ランダムに係数を選んで乗算する。"
    )
    pa.add_argument(
        "--blur_k",
        nargs=2,
        type=int,
        default=[3, 3],
        help="ガウシアンぼかしのカーネルサイズ範囲 [min, max]。内部では奇数に丸めて使用（例: 5,7,9）。"
    )

    # stretch（縦横引き伸ばし）
    pa.add_argument(
        "--stretch_range",
        nargs=2,
        type=float,
        default=[1.0, 1.5],
        help="stretch で縦または横方向にかけるスケール倍率のレンジ [min, max]。1.0 以上で引き伸ばし。"
    )

    # ランダム BBOX 関連
    pa.add_argument(
        "--rand_boxes",
        nargs=2,
        type=int,
        default=[1, 3],
        help="ランダム BBOX を何個配置するかの個数レンジ [min, max]。この範囲から整数をランダムに選んでその個数だけ生成。"
    )
    pa.add_argument(
        "--rand_box_wh",
        nargs=2,
        type=int,
        default=[20, 60],
        help="ランダム BBOX の一辺の画素数（幅・高さ）のレンジ [min, max]。224px を基準とした値で、sz に応じてスケールされる。"
    )
    pa.add_argument(
        "--rand_box_area",
        nargs=4,
        type=int,
        default=[0, 0, 224, 224],
        help="ランダム BBOX を配置する領域 [x1, y1, x2, y2]。224px ベースの座標で指定し、sz に応じてスケールされる。"
    )

    # hide_quadrants 関連
    pa.add_argument(
        "--hide_n",
        nargs=2,
        type=int,
        default=[1, 3],
        help="hide_quadrants でマスクするクォドラント数のレンジ [min, max]。4 分割のうちランダムにこの個数だけ塗りつぶす。"
    )

    # crop & paste 関連
    pa.add_argument(
        "--crop_params",
        nargs=4,
        type=int,
        default=[112, 112, 150, 224],
        help="crop & paste の基準矩形 [cx, cy, w, h]。224px ベースの中心座標 (cx, cy) と幅 w, 高さ h。sz に応じてスケールされる。"
    )
    pa.add_argument(
        "--visibility_threshold",
        type=float,
        default=0.6,
        help="crop & paste で貼り付ける際に、切り出した領域の少なくともこの割合 (0〜1) が画像内に見えるように配置する閾値。"
    )

    args = pa.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    in_base = Path(args.in_path).resolve()

    # 出力ルート名決定（*_aug-vN）
    v = 1
    while True:
        out_name = f"{in_base.name}_aug-v{v}"
        final_out_root = Path(args.out_root).resolve() / out_name
        if not final_out_root.exists():
            break
        v += 1
    print(f"[*] Target Output: {final_out_root}")

    # 念のため先にルートを作成しておく
    final_out_root.mkdir(parents=True, exist_ok=True)

    # 対象ディレクトリ探索（cache_subdirs を考慮）
    target_dirs: List[Path] = []
    cache_root = in_base / "cache"

    for sub in args.cache_subdirs:
        base = cache_root / sub
        if not base.exists():
            print(f"[WARN] cache subdir not found: {base} → skipping")
            continue
        sub_targets = [
            p
            for p in base.rglob("*")
            if p.is_dir()
            and re.search(r"sz\d+_area$", p.name)
            and (p / "train/imgs").exists()
        ]
        if not sub_targets:
            print(f"[WARN] no sz*_area/train/imgs found under: {base}")
        target_dirs.extend(sub_targets)

    if not target_dirs:
        print("[!] No target dirs found under specified cache_subdirs")
        return

    # 各 szXXX_area ごとのスケール済みパラメータを記録するリスト
    targets_info = []

    for in_dir in target_dirs:
        m = re.search(r"sz(\d+)_area", in_dir.name)
        img_sz = int(m.group(1))
        s = img_sz / 224.0
        scaler = BBoxScaler(s)

        rel_path = in_dir.relative_to(in_base)
        train_out = final_out_root / rel_path / "train"
        img_out_dir = train_out / "imgs"
        img_out_dir.mkdir(parents=True, exist_ok=True)

        train_in = in_dir / "train"
        src = sorted((train_in / "imgs").glob("*.png")) + sorted(
            (train_in / "imgs").glob("*.jpg")
        )
        if args.test:
            src = src[:10]

        meta = {
            r["filename"]: r
            for r in list(csv.DictReader((train_in / "labels.csv").open()))
        }

        # スケーリング済み引数
        temp_args = argparse.Namespace(**vars(args))
        temp_args.rand_box_wh = [int(v * s) for v in args.rand_box_wh]
        temp_args.rand_box_area = [int(v * s) for v in args.rand_box_area]
        temp_args.crop_params = [int(v * s) for v in args.crop_params]

        # この szXXX_area に対する有効パラメータを記録
        targets_info.append(
            {
                "rel_path": str(rel_path),
                "img_size": img_sz,
                "scale": s,
                "effective_params": {
                    "rand_box_wh": list(temp_args.rand_box_wh),
                    "rand_box_area": list(temp_args.rand_box_area),
                    "crop_params": list(temp_args.crop_params),
                },
            }
        )

        all_new_rows = []
        # --- 並列処理の実行 ---
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_single_image,
                    p,
                    meta.get(
                        p.name,
                        {"filename": p.name, "roll": 0, "pitch": 0, "yaw": 0},
                    ),
                    scaler,
                    temp_args,
                    img_out_dir,
                    args.fixed_color,
                    args.rand_color,
                )
                for p in src
            ]
            for f in tqdm(
                as_completed(futures),
                total=len(src),
                desc=f"Processing {rel_path} ({img_sz}px)",
            ):
                all_new_rows.extend(f.result())

        with (train_out / "labels.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, ["filename", "roll", "pitch", "yaw"])
            w.writeheader()
            w.writerows(all_new_rows)

    # ─── augment_config.yaml を保存 ───
    cfg_dict = build_augment_config_dict(args, in_base, final_out_root, targets_info)
    cfg_path = final_out_root / "augment_config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=True)

    print(f"\n[OK] All tasks completed! Saved to: {final_out_root}")
    print(f"[OK] augment_config.yaml saved to: {cfg_path}")

if __name__ == "__main__":

    start_time = time.time()

    main()

    end_time = time.time()
    elapsed = end_time - start_time

    print("\n" + "=" * 40)
    print(f"処理完了時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"合計実行時間: {elapsed:.2f} 秒")
    if elapsed > 60:
        print(f"合計実行時間: {elapsed / 60:.2f} 分")
    print("=" * 40)
