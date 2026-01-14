#!/usr/bin/env python3
"""
augment_sz_area_v5_batch.py
---------------------
* 指定した type-X 配下の cache を一括処理
* 出力フォルダ名を type-X_aug-vN として自動連番
* ベース変換のみの保存可否をフラグで制御可能
"""

from __future__ import annotations
import argparse, random, math, csv, re, os
from pathlib import Path
from typing import List, Tuple, Dict, Callable
import cv2
import numpy as np
from tqdm import tqdm
import time

# ──────────────────────────────────────────────────────────────
# ユーザー設定セクション
# ──────────────────────────────────────────────────────────────
class CONFIG:
    # 1. ベース変換の選択
    # SELECTED_BASES = ["orig", "iso_noise", "blur", "bright", "bright_iso", "bright_blur", "vstrip"]
    SELECTED_BASES = ["iso_noise", "blur", "bright", "vstrip"]

    # 2. 派生変換の有効化
    ENABLE_RBBOX  = True
    ENABLE_CROP   = True
    ENABLE_HIDE   = True

    # 3. 保存設定
    # True: ベース変換のみの画像（例: data_iso_noise.png）を保存する
    # False: 派生変換がある画像（例: data_iso_noise_rbbox.png）のみを保存する
    SAVE_BASE_TRANSFORMS = False

# ──────────────────────────────────────────────────────────────
# 変換ロジック (apply_vstrip, iso_noise 等)
# ──────────────────────────────────────────────────────────────

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
            if random.random() < 0.5: val = -val
            if i == 0 or abs(val - offsets[i-1]) >= 32:
                offsets.append(val); found = True; break
        if not found: offsets.append(val)
    for i in range(n_strips):
        x_start, x_end = edges[i], edges[i+1]
        offset = offsets[i]
        bright_map[:, x_start:x_end] = offset
        if blend_ratio > 0 and i < n_strips - 1:
            strip_w = x_end - x_start; blend_w = int(strip_w * blend_ratio)
            if blend_w > 0:
                next_offset = offsets[i+1]
                for dx in range(blend_w):
                    alpha = dx / blend_w; pos = x_end - (blend_w // 2) + dx
                    if 0 <= pos < w: bright_map[:, pos] = (1 - alpha) * offset + alpha * next_offset
    img_f = img.astype(np.float32)
    res = img_f + (bright_map[:, :, np.newaxis] if img.ndim == 3 else bright_map)
    return np.clip(res, 0, 255).astype(np.uint8)

def add_iso_noise(img, sigma):
    if sigma <= 0: return img
    f = img.astype(np.float32) / 255.0
    noisy = np.clip(f + (np.random.poisson(f*255)/255 - f) + np.random.normal(0, sigma/255, f.shape), 0, 1)
    return (noisy*255).astype(img.dtype)

def adjust_brightness(img, lo, hi):
    return np.clip(img.astype(np.float32)*random.uniform(lo,hi),0,255).astype(img.dtype)

def gaussian_blur(img, k_rng):
    k = random.randrange(k_rng[0]|1, (k_rng[1]+1)|1, 2)
    return cv2.GaussianBlur(img, (k,k), 0)

def hide_quadrants(img, n_rng):
    n=random.randint(*n_rng); out=img.copy(); h2,w2=out.shape[0]//2,out.shape[1]//2
    quads=[(slice(0,h2),slice(0,w2)),(slice(0,h2),slice(w2,None)),(slice(h2,None),slice(0,w2)),(slice(h2,None),slice(w2,None))]
    for idx in random.sample(range(4), n): out[quads[idx]]=0
    return out

def specific_crop_and_paste(img, crop, vis_thr=0.6):
    cx,cy,w,h = crop; H,W = img.shape[:2]
    x1,y1,x2,y2 = cx-w//2, cy-h//2, cx+w//2, cy+h//2
    cx1,cy1,cx2,cy2 = max(0,x1),max(0,y1),min(W,x2),min(H,y2)
    cropped = img[cy1:cy2, cx1:cx2]
    if cropped.size == 0: return np.zeros_like(img)
    ch,cw = cropped.shape[:2]; min_w, min_h = int(cw*vis_thr), int(ch*vis_thr)
    px, py = random.randint(-(cw-min_w), W-min_w), random.randint(-(ch-min_h), H-min_h)
    canvas = np.zeros_like(img); dx1,dy1 = max(0,px), max(0,py); dx2,dy2 = min(W,px+cw), min(H,py+ch)
    sx1,sy1 = dx1-px, dy1-py; sx2,sy2 = sx1+(dx2-dx1), sy1+(dy2-dy1)
    canvas[dy1:dy2, dx1:dx2] = cropped[sy1:sy2, sx1:sx2]
    return canvas

# ──────────────────────────────────────────────────────────────
# BBOX ユーティリティ
# ──────────────────────────────────────────────────────────────
class BBoxScaler:
    def __init__(self, scale: float):
        self.C=int(round(112*scale)); self.O1=int(round(50*scale)); self.O2=int(round(90*scale)); self.BW=self.BH=int(round(40*scale))
    def vertices(self, cx, cy):
        return (int(cx-self.BW/2), int(cy-self.BH/2)), (int(cx+self.BW/2), int(cy+self.BH/2))

def get_fixed_bboxes(roll: float, scaler: BBoxScaler):
    roll %= 360; rad = math.radians(roll); boxes=[]
    if not (0<=roll<=35 or 325<=roll<=360): boxes.append(scaler.vertices(scaler.C, scaler.C - scaler.O1*math.sin(rad)))
    if not (135<=roll<=225): boxes.append(scaler.vertices(scaler.C, scaler.C + scaler.O2*math.sin(rad)))
    return boxes

def fill_roi_partial(img, tl, br, color):
    x1,y1=tl; x2,y2=br; x1,y1=max(0,x1),max(0,y1); x2,y2=min(img.shape[1],x2),min(img.shape[0],y2)
    roi=img[y1:y2, x1:x2]
    if roi.size==0: return
    h,w=roi.shape[:2]; scale=math.sqrt(random.uniform(0.25,0.75)); rw,rh=int(w*scale),int(h*scale)
    rx,ry=random.randint(0,w-rw),random.randint(0,h-rh); sub=roi[ry:ry+rh, rx:rx+rw]
    if img.ndim==2: sub[:]=color[0]
    else: sub[:]=color if len(color)==3 else (color[0],)*3

def apply_fixed_bboxes(img, roll, scaler, color_fn):
    out=img.copy()
    for tl,br in get_fixed_bboxes(roll, scaler): fill_roi_partial(out, tl, br, color_fn())
    return out

def apply_random_bboxes(img, roll, scaler, args, color_fn):
    out=img.copy(); existing=get_fixed_bboxes(roll, scaler)
    tlx,tly,brx,bry=args.rand_box_area; target=random.randint(*args.rand_boxes); boxes=[]; trials=50
    while len(boxes)<target and trials>0:
        trials-=1; w,h=random.randint(*args.rand_box_wh),random.randint(*args.rand_box_wh)
        x1,y1=random.randint(tlx,max(tlx,brx-w)),random.randint(tly,max(tly,bry-h))
        box=((x1,y1),(x1+w,y1+h)); (ax1,ay1),(ax2,ay2)=box; overlap=False
        for (bx1,by1),(bx2,by2) in existing+boxes:
            if ax1<bx2 and ax2>bx1 and ay1<by2 and ay2>by1: overlap=True; break
        if not overlap: boxes.append(box)
    for tl,br in boxes: fill_roi_partial(out, tl, br, color_fn())
    return out

# ──────────────────────────────────────────────────────────────
# パイプライン
# ──────────────────────────────────────────────────────────────
BASE_TRANSFORMS = {
    "orig": [], "iso_noise": ["iso_noise"], "blur": ["blur"], "bright": ["bright"],
    "bright_iso": ["bright", "iso_noise"], "bright_blur": ["bright", "blur"], "vstrip": ["vstrip"]
}
DERIVED_SUFFIXES = {"random_bbox": "rbbox", "crop": "crop", "hide": "hide"}

def apply_seq(img, ops, args):
    out = img
    for op in ops:
        if op == "iso_noise": out = add_iso_noise(out, args.iso_sigma)
        elif op == "bright":  out = adjust_brightness(out, *args.bright)
        elif op == "blur":    out = gaussian_blur(out, tuple(args.blur_k))
        elif op == "vstrip":  out = apply_vstrip(out, args.vstrip_n, tuple(args.strip_bright_range), args.strip_blend_ratio)
    return out

def generate_variants(img, roll, scaler, args, fixed_color_fn, rand_color_fn):
    res = {}; base_imgs = {}
    for n in CONFIG.SELECTED_BASES:
        if n in BASE_TRANSFORMS: base_imgs[n] = apply_seq(img, BASE_TRANSFORMS[n], args)
    res.update(base_imgs)
    for n, b in base_imgs.items():
        fb = apply_fixed_bboxes(b, roll, scaler, fixed_color_fn)
        if CONFIG.ENABLE_RBBOX: res[f"{n}_{DERIVED_SUFFIXES['random_bbox']}"] = apply_random_bboxes(fb, roll, scaler, args, rand_color_fn)
        if CONFIG.ENABLE_CROP:  res[f"{n}_{DERIVED_SUFFIXES['crop']}"] = specific_crop_and_paste(fb, tuple(args.crop_params), args.visibility_threshold)
        if CONFIG.ENABLE_HIDE:  res[f"{n}_{DERIVED_SUFFIXES['hide']}"] = hide_quadrants(b, tuple(args.hide_n))
    return res

def make_color_fn(keyword: str) -> Callable[[], Tuple[int, ...]]:
    kw = keyword.lower()
    if kw == "black": return lambda: (0,)
    if kw == "white": return lambda: (255,)
    if kw == "gray":  return lambda: (128,)
    if kw == "random_gray":  return lambda: (random.randint(0, 255),)
    if kw == "random_color": return lambda: tuple(random.randint(0, 255) for _ in range(3))
    raise ValueError(f"不正な色キーワード: {keyword}")

# ──────────────────────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--in_path",  required=True, help="datasets/type-2 のようなディレクトリを指定")
    pa.add_argument("--out_root", required=True, help="保存先のルート")
    pa.add_argument("--fixed_color", default="random_color")
    pa.add_argument("--rand_color",  default="random_color")
    pa.add_argument("--seed", type=int, default=42); pa.add_argument("--test", action="store_true")
    # 拡張用パラメータ
    pa.add_argument("--vstrip_n", nargs="+", type=int, default=[2, 4, 5])
    pa.add_argument("--strip_bright_range", nargs=2, type=float, default=[30, 100])
    pa.add_argument("--strip_blend_ratio", type=float, default=0.1)
    pa.add_argument("--iso_sigma", type=float, default=8.0); pa.add_argument("--bright", nargs=2, type=float, default=[0.7, 1.3])
    pa.add_argument("--blur_k", nargs=2, type=int, default=[3, 7])
    pa.add_argument("--rand_boxes", nargs=2, type=int, default=[1, 3])
    pa.add_argument("--rand_box_wh", nargs=2, type=int, default=[20, 60])
    pa.add_argument("--rand_box_area", nargs=4, type=int, default=[0, 0, 224, 224])
    pa.add_argument("--hide_n", nargs=2, type=int, default=[1, 3])
    pa.add_argument("--crop_params", nargs=4, type=int, default=[112, 112, 150, 224])
    pa.add_argument("--visibility_threshold", type=float, default=0.6)
    args = pa.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    
    in_base = Path(args.in_path).resolve()
    if not in_base.exists(): raise FileNotFoundError(f"Input path not found: {in_base}")

    # 1. 出力フォルダの自動命名 (type-2_aug-vN)
    v = 1
    while True:
        out_name = f"{in_base.name}_aug-v{v}"
        final_out_root = Path(args.out_root).resolve() / out_name
        if not final_out_root.exists(): break
        v += 1
    print(f"[*] Target Output: {final_out_root}")

    # 2. 処理対象フォルダの探索 (type-2/cache/{gray, rgb...}/szXXX_area)
    target_dirs = []
    cache_path = in_base / "cache"
    if not cache_path.exists():
        raise ValueError(f"'cache' directory not found in {in_base}")
    
    for p in cache_path.rglob("*"):
        if p.is_dir() and re.search(r"sz\d+_area$", p.name):
            if (p / "train/imgs").exists():
                target_dirs.append(p)
    
    if not target_dirs:
        print("[!] No target directories (szXXX_area/train/imgs) found.")
        return

    print(f"[*] Found {len(target_dirs)} directories to process.")

    # 3. 各フォルダを順次処理
    f_col, r_col = make_color_fn(args.fixed_color), make_color_fn(args.rand_color)

    for in_dir in target_dirs:
        m = re.search(r"sz(\d+)_area", in_dir.name)
        img_sz = int(m.group(1)); s = img_sz / 224.0; scaler = BBoxScaler(s)

        rel_path = in_dir.relative_to(in_base)
        train_out = final_out_root / rel_path / "train"
        (train_out / "imgs").mkdir(parents=True, exist_ok=True)
        
        train_in = in_dir / "train"
        src = sorted((train_in/"imgs").glob("*.png")) + sorted((train_in/"imgs").glob("*.jpg"))
        if args.test: src = src[:5]

        meta = {r["filename"]: r for r in list(csv.DictReader((train_in/"labels.csv").open()))}
        
        new_rows = []
        for p in tqdm(src, desc=f"Processing {rel_path} ({img_sz}px)"):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            row_in = meta.get(p.name, {"filename":p.name, "roll":0, "pitch":0, "yaw":0})
            
            temp_args = argparse.Namespace(**vars(args))
            def scale_list(lst): return [int(round(v*s)) for v in lst]
            temp_args.rand_box_wh = scale_list(args.rand_box_wh)
            temp_args.rand_box_area = scale_list(args.rand_box_area)
            temp_args.crop_params = scale_list(args.crop_params)

            variants = generate_variants(img, float(row_in["roll"]), scaler, temp_args, f_col, r_col)
            
            for tag, v in variants.items():
                if tag == "orig": continue 
                
                # --- 追加ロジック: ベース変換保存フラグの判定 ---
                # tag が SELECTED_BASES に含まれている（＝派生サフィックスがついていない）場合、
                # CONFIG.SAVE_BASE_TRANSFORMS が False なら保存をスキップする。
                if not CONFIG.SAVE_BASE_TRANSFORMS and tag in CONFIG.SELECTED_BASES:
                    continue
                # ---------------------------------------------

                if np.count_nonzero(v) == 0: continue
                fn = f"{p.stem}_{tag}.png"
                cv2.imwrite(str(train_out/"imgs"/fn), v)
                new_row = row_in.copy(); new_row["filename"] = fn; new_rows.append(new_row)

        with (train_out/"labels.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, ["filename", "roll", "pitch", "yaw"])
            w.writeheader(); w.writerows(new_rows)

    print(f"\n[OK] All tasks completed! Saved to: {final_out_root}")

if __name__ == "__main__":
    start_time = time.time()

    main()

    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "="*40)
    print(f"処理完了時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"合計実行時間: {elapsed:.2f} 秒")
    # 分単位でも表示すると親切です
    if elapsed > 60:
        print(f"合計実行時間: {elapsed / 60:.2f} 分")
    print("="*40)