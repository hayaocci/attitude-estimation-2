#!/usr/bin/env python3
"""
augment_sz_area_v5.py
---------------------
* 1枚から30パターンを生成 (6 base x 5 derivatives)
* 追加機能: 縦長短冊(vstrip)輝度変換
* 境界の線形ブレンディングによる滑らかな明暗差を再現
"""

from __future__ import annotations
import argparse, random, math, csv, re
from pathlib import Path
from typing import List, Tuple, Dict, Callable
import cv2
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
#  カラー・ユーティリティ
# ──────────────────────────────────────────────────────────────
def make_color_fn(keyword: str) -> Callable[[], Tuple[int, ...]]:
    kw = keyword.lower()
    if kw == "black": return lambda: (0,)
    if kw == "white": return lambda: (255,)
    if kw == "gray":  return lambda: (128,)
    if kw == "random_gray":  return lambda: (random.randint(0, 255),)
    if kw == "random_color": return lambda: tuple(random.randint(0, 255) for _ in range(3))
    raise ValueError(f"不正な色キーワード: {keyword}")

# ──────────────────────────────────────────────────────────────
#  新規：縦長短冊輝度変換 (vstrip)
# ──────────────────────────────────────────────────────────────
# def apply_vstrip(img: np.ndarray, 
#                  n_strips_range: Tuple[int, int], 
#                  bright_range: Tuple[float, float],
#                  blend_ratio: float) -> np.ndarray:
#     """
#     画像を垂直に分割し、各カラムに輝度オフセットを適用する。
#     blend_ratio: 短冊の幅に対する境界ぼかしの割合 (0.0 ~ 0.5)
#     """
#     h, w = img.shape[:2]
#     n_strips = random.randint(*n_strips_range)
    
#     # 輝度マップを作成 (float32)
#     bright_map = np.zeros((h, w), dtype=np.float32)
    
#     # 各短冊の境界位置を計算
#     edges = np.linspace(0, w, n_strips + 1).astype(int)
    
#     # 各短冊のオフセットを決定
#     offsets = []
#     for _ in range(n_strips):
#         val = random.uniform(*bright_range)
#         if random.random() < 0.5: val = -val
#         offsets.append(val)

#     # マップの塗りつぶしと境界ブレンディング
#     for i in range(n_strips):
#         x_start, x_end = edges[i], edges[i+1]
#         offset = offsets[i]
        
#         # 基本の塗りつぶし
#         bright_map[:, x_start:x_end] = offset

#         # 境界のぼかし（線形補間）
#         if blend_ratio > 0 and i < n_strips - 1:
#             strip_w = x_end - x_start
#             blend_w = int(strip_w * blend_ratio)
#             if blend_w > 0:
#                 # 次の短冊との境界
#                 next_offset = offsets[i+1]
#                 for dx in range(blend_w):
#                     alpha = dx / blend_w
#                     # 境界線をまたいで徐々に変化させる
#                     pos = x_end - (blend_w // 2) + dx
#                     if 0 <= pos < w:
#                         bright_map[:, pos] = (1 - alpha) * offset + alpha * next_offset

#     # 画像に適用
#     img_f = img.astype(np.float32)
#     if img.ndim == 3:
#         res = img_f + bright_map[:, :, np.newaxis]
#     else:
#         res = img_f + bright_map
    
#     return np.clip(res, 0, 255).astype(np.uint8)

# def apply_vstrip(img: np.ndarray, 
#                  n_strips_range: Tuple[int, int], 
#                  bright_range: Tuple[float, float],
#                  blend_ratio: float) -> np.ndarray:
#     """
#     画像を垂直に分割し、各カラムに輝度オフセットを適用する。
#     隣り合う短冊の輝度差が32以上になるように制御。
#     """
#     h, w = img.shape[:2]
#     n_strips = random.randint(*n_strips_range)
    
#     bright_map = np.zeros((h, w), dtype=np.float32)
#     edges = np.linspace(0, w, n_strips + 1).astype(int)
    
#     offsets = []
#     for i in range(n_strips):
#         # 隣り合う短冊との差が32以上になるまで再抽選
#         # (無限ループ防止のため最大100回まで試行)
#         found = False
#         for _ in range(100):
#             val = random.uniform(*bright_range)
#             if random.random() < 0.5: val = -val
            
#             if i == 0:
#                 offsets.append(val)
#                 found = True
#                 break
#             else:
#                 prev_val = offsets[i-1]
#                 if abs(val - prev_val) >= 32:
#                     offsets.append(val)
#                     found = True
#                     break
#         if not found:
#             # 万が一見つからなかった場合は、強制的に直前から32離した値にするなどの処理
#             offsets.append(val)

#     # 以降、bright_map への適用とブレンディング処理は同じ
#     for i in range(n_strips):
#         x_start, x_end = edges[i], edges[i+1]
#         offset = offsets[i]
#         bright_map[:, x_start:x_end] = offset

#         if blend_ratio > 0 and i < n_strips - 1:
#             strip_w = x_end - x_start
#             blend_w = int(strip_w * blend_ratio)
#             if blend_w > 0:
#                 next_offset = offsets[i+1]
#                 for dx in range(blend_w):
#                     alpha = dx / blend_w
#                     pos = x_end - (blend_w // 2) + dx
#                     if 0 <= pos < w:
#                         bright_map[:, pos] = (1 - alpha) * offset + alpha * next_offset

#     img_f = img.astype(np.float32)
#     res = img_f + (bright_map[:, :, np.newaxis] if img.ndim == 3 else bright_map)
#     return np.clip(res, 0, 255).astype(np.uint8)

def apply_vstrip(img: np.ndarray, 
                 n_options: List[int], 
                 bright_range: Tuple[float, float],
                 blend_ratio: float) -> np.ndarray:
    """
    画像を垂直に分割し、各カラムに輝度オフセットを適用する。
    n_options: 分割数の選択肢リスト（例: [2, 4, 5]）
    """
    h, w = img.shape[:2]
    n_strips = random.choice(n_options)  # リストから選択
    
    bright_map = np.zeros((h, w), dtype=np.float32)
    edges = np.linspace(0, w, n_strips + 1).astype(int)
    
    offsets = []
    for i in range(n_strips):
        found = False
        for _ in range(100):
            val = random.uniform(*bright_range)
            if random.random() < 0.5: val = -val
            
            if i == 0:
                offsets.append(val)
                found = True
                break
            else:
                if abs(val - offsets[i-1]) >= 32:  # 輝度差32以上を保証
                    offsets.append(val)
                    found = True
                    break
        if not found: offsets.append(val)

    # 描画・ブレンディング処理
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

# ──────────────────────────────────────────────────────────────
#  BBOX ユーティリティ
# ──────────────────────────────────────────────────────────────
class BBoxScaler:
    def __init__(self, scale: float):
        self.C  = int(round(112 * scale))
        self.O1 = int(round(50  * scale))
        self.O2 = int(round(90  * scale))
        self.BW = self.BH = int(round(40 * scale))
    def vertices(self, cx, cy):
        return (int(cx-self.BW/2), int(cy-self.BH/2)), \
               (int(cx+self.BW/2), int(cy+self.BH/2))

def get_fixed_bboxes(roll: float, scaler: BBoxScaler):
    roll %= 360; rad = math.radians(roll); boxes=[]
    if not (0<=roll<=35 or 325<=roll<=360):
        tl,br = scaler.vertices(scaler.C, scaler.C - scaler.O1*math.sin(rad))
        boxes.append((tl,br))
    if not (135<=roll<=225):
        tl,br = scaler.vertices(scaler.C, scaler.C + scaler.O2*math.sin(rad))
        boxes.append((tl,br))
    return boxes

def fill_roi_partial(img, tl, br, color):
    x1,y1=tl; x2,y2=br
    x1,y1=max(0,x1),max(0,y1); x2,y2=min(img.shape[1],x2),min(img.shape[0],y2)
    roi=img[y1:y2, x1:x2]
    if roi.size==0: return
    h,w=roi.shape[:2]; scale=math.sqrt(random.uniform(0.25,0.75))
    rw,rh=int(w*scale),int(h*scale)
    rx,ry=random.randint(0,w-rw),random.randint(0,h-rh)
    sub=roi[ry:ry+rh, rx:rx+rw]
    if img.ndim==2: sub[:]=color[0]
    else: sub[:]=color if len(color)==3 else (color[0],)*3

def apply_fixed_bboxes(img, roll, scaler, color_fn):
    out=img.copy()
    for tl,br in get_fixed_bboxes(roll, scaler):
        fill_roi_partial(out, tl, br, color_fn())
    return out

def gen_random_boxes(n_rng,wh_rng,area,existing):
    tlx,tly,brx,bry=area; boxes=[]
    target=random.randint(*n_rng); trials=50
    while len(boxes)<target and trials>0:
        trials-=1
        w=random.randint(*wh_rng); h=random.randint(*wh_rng)
        x1=random.randint(tlx,max(tlx,brx-w)); y1=random.randint(tly,max(tly,bry-h))
        box=((x1,y1),(x1+w,y1+h))
        (ax1,ay1),(ax2,ay2)=box; overlap=False
        for (bx1,by1),(bx2,by2) in existing+boxes:
            if ax1<bx2 and ax2>bx1 and ay1<by2 and ay2>by1:
                overlap=True; break
        if not overlap: boxes.append(box)
    return boxes

def apply_random_bboxes(img, roll, scaler, args, color_fn):
    out=img.copy()
    existing=get_fixed_bboxes(roll, scaler)
    for tl,br in gen_random_boxes(tuple(args.rand_boxes), tuple(args.rand_box_wh), tuple(args.rand_box_area), existing):
        fill_roi_partial(out, tl, br, color_fn())
    return out

# ──────────────────────────────────────────────────────────────
#  その他変換
# ──────────────────────────────────────────────────────────────
def add_iso_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0: return img
    f = img.astype(np.float32) / 255.0
    noisy = np.clip(f + (np.random.poisson(f*255)/255 - f) + np.random.normal(0, sigma/255, f.shape), 0, 1)
    return (noisy*255).astype(img.dtype)

def adjust_brightness(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return img if (lo,hi)==(1.0,1.0) else np.clip(img.astype(np.float32)*random.uniform(lo,hi),0,255).astype(img.dtype)

def gaussian_blur(img: np.ndarray, k_rng: Tuple[int,int]) -> np.ndarray:
    k = random.randrange(k_rng[0]|1, (k_rng[1]+1)|1, 2)
    return cv2.GaussianBlur(img, (k,k), 0)

def hide_quadrants(img: np.ndarray, n_rng: Tuple[int,int]) -> np.ndarray:
    n=random.randint(*n_rng); out=img.copy()
    if n>0:
        h2,w2=out.shape[0]//2,out.shape[1]//2
        quads=[(slice(0,h2),slice(0,w2)),(slice(0,h2),slice(w2,None)),(slice(h2,None),slice(0,w2)),(slice(h2,None),slice(w2,None))]
        for idx in random.sample(range(4), n):
            r,c=quads[idx]; out[r,c]=0
    return out

def specific_crop_and_paste(img, crop, vis_thr=0.6):
    cx,cy,w,h = crop; H,W = img.shape[:2]
    x1,y1 = cx-w//2, cy-h//2; x2,y2 = x1+w, y1+h
    cx1,cy1=max(0,x1),max(0,y1); cx2,cy2=min(W,x2),min(H,y2)
    cropped = img[cy1:cy2, cx1:cx2]
    if cropped.size == 0: return np.zeros_like(img)
    ch,cw = cropped.shape[:2]; min_w, min_h = int(cw*vis_thr), int(ch*vis_thr)
    px = random.randint(-(cw-min_w), W-min_w); py = random.randint(-(ch-min_h), H-min_h)
    canvas = np.zeros_like(img)
    dx1,dy1 = max(0,px), max(0,py); dx2,dy2 = min(W,px+cw), min(H,py+ch)
    sx1,sy1 = dx1-px, dy1-py; sx2,sy2 = sx1+(dx2-dx1), sy1+(dy2-dy1)
    canvas[dy1:dy2, dx1:dx2] = cropped[sy1:sy2, sx1:sx2]
    return canvas

# ──────────────────────────────────────────────────────────────
#  パイプライン定義（30パターン）
# ──────────────────────────────────────────────────────────────
BASE_TRANSFORMS = {
    "orig":        [],
    "iso_noise":   ["iso_noise"],
    "blur":        ["blur"],
    "bright":      ["bright"],
    "bright_iso":  ["bright", "iso_noise"],
    "bright_blur": ["bright", "blur"],
}
DERIVED_SUFFIXES = {
    "random_bbox": "rbbox",
    "crop":        "crop",
    "hide":        "hide",
    "vstrip":      "vstrip",  # 新規追加
}

def apply_seq(img, ops, args):
    out = img
    for op in ops:
        if op == "iso_noise": out = add_iso_noise(out, args.iso_sigma)
        elif op == "bright":  out = adjust_brightness(out, *args.bright)
        elif op == "blur":    out = gaussian_blur(out, tuple(args.blur_k))
    return out

def generate_variants(img, roll, scaler, args, fixed_color_fn, rand_color_fn):
    res = {}
    base = {n: apply_seq(img, ops, args) for n,ops in BASE_TRANSFORMS.items()}
    res.update(base)   # 基本 6

    for n, b in base.items():
        # A. BBOX系
        fb = apply_fixed_bboxes(b, roll, scaler, fixed_color_fn)
        res[f"{n}_{DERIVED_SUFFIXES['random_bbox']}"] = apply_random_bboxes(fb, roll, scaler, args, rand_color_fn)

        # B. クロップ系
        res[f"{n}_{DERIVED_SUFFIXES['crop']}"] = specific_crop_and_paste(fb, tuple(args.crop_params), args.visibility_threshold)

        # C. 象限隠し
        res[f"{n}_{DERIVED_SUFFIXES['hide']}"] = hide_quadrants(b, tuple(args.hide_n))
        
        # D. 新規：縦長短冊輝度変換
        # res[f"{n}_{DERIVED_SUFFIXES['vstrip']}"] = apply_vstrip(b, tuple(args.vstrip_n), tuple(args.strip_bright_range), args.strip_blend_ratio)
    # args.vstrip_n は [2, 4, 5] というリストとして渡されます
        res[f"{n}_{DERIVED_SUFFIXES['vstrip']}"] = apply_vstrip(
            b, 
            args.vstrip_n,              # 修正点：リストをそのまま渡す
            tuple(args.strip_bright_range), 
            args.strip_blend_ratio
        )


    return res  # 6 + (6x4) = 30枚

# ──────────────────────────────────────────────────────────────
#  Main Logic
# ──────────────────────────────────────────────────────────────
def resolve_paths(in_path: Path, out_root: Path) -> tuple[Path, Path]:
    if not re.search(r"sz\d+_area$", in_path.as_posix()): raise ValueError("--in_path error")
    type_dir = next((p for p in in_path.parents if re.match(r"type-\w+", p.name)), None)
    if type_dir is None: raise ValueError("type-* not found")
    rel = in_path.relative_to(type_dir)
    train_in, train_out = in_path / "train", out_root / rel / "train"
    (train_out / "imgs").mkdir(parents=True, exist_ok=True)
    return train_in, train_out

def main():
    COLORS = ["black","white","gray","random_gray","random_color"]
    pa = argparse.ArgumentParser()
    pa.add_argument("--in_path",  required=True)
    pa.add_argument("--out_root", required=True)
    pa.add_argument("--fixed_color", default="random_color", choices=COLORS)
    pa.add_argument("--rand_color",  default="random_color", choices=COLORS)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--test", action="store_true")
    # vstrip用パラメータ
    pa.add_argument("--vstrip_n", nargs="+", type=int, default=[2, 4, 5], help="分割数の範囲")
    pa.add_argument("--strip_bright_range", nargs=2, type=float, default=[30, 100], help="輝度変化の範囲")
    pa.add_argument("--strip_blend_ratio", type=float, default=0.1, help="境界ぼかし幅（短冊幅に対する比率）")
    # 既存パラメータ
    pa.add_argument("--iso_sigma", type=float, default=8.0)
    pa.add_argument("--bright",    nargs=2, type=float, default=[0.7, 1.3])
    pa.add_argument("--blur_k",    nargs=2, type=int,   default=[3, 7])
    pa.add_argument("--rand_boxes",  nargs=2, type=int, default=[1, 3])
    pa.add_argument("--rand_box_wh", nargs=2, type=int, default=[20, 60])
    pa.add_argument("--rand_box_area", nargs=4, type=int, default=[0, 0, 224, 224])
    pa.add_argument("--hide_n",  nargs=2, type=int, default=[1, 3])
    pa.add_argument("--crop_params", nargs=4, type=int, default=[112, 112, 150, 224])
    pa.add_argument("--visibility_threshold", type=float, default=0.6)
    args = pa.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    # スケール判定
    m = re.search(r"sz(\d+)_area", Path(args.in_path).as_posix())
    img_sz = int(m.group(1)); s = img_sz / 224.0
    scaler = BBoxScaler(s)

    # パラメータスケーリング
    def scale_list(lst): return [int(round(v*s)) for v in lst]
    args.rand_box_wh = scale_list(args.rand_box_wh)
    args.rand_box_area = scale_list(args.rand_box_area)
    args.crop_params = scale_list(args.crop_params)

    train_in, train_out = resolve_paths(Path(args.in_path).resolve(), Path(args.out_root).resolve())
    src = sorted((train_in/"imgs").glob("*.png")) + sorted((train_in/"imgs").glob("*.jpg"))
    if args.test: src = src[:5]

    meta = {r["filename"]: r for r in list(csv.DictReader((train_in/"labels.csv").open()))}
    f_col, r_col = make_color_fn(args.fixed_color), make_color_fn(args.rand_color)

    new_rows = []
    for p in tqdm(src, desc=f"Augment v5 (30x) {img_sz}px"):
        # オリジナルコピー
        dst_orig = train_out/"imgs"/p.name
        if not dst_orig.exists(): cv2.imwrite(str(dst_orig), cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
        
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        row_in = meta.get(p.name, {"filename":p.name, "roll":0, "pitch":0, "yaw":0})
        
        variants = generate_variants(img, float(row_in["roll"]), scaler, args, f_col, r_col)
        for tag, v in variants.items():
            if np.count_nonzero(v) == 0: continue
            fn = f"{p.stem}_{tag}.png"
            cv2.imwrite(str(train_out/"imgs"/fn), v)
            new_row = row_in.copy(); new_row["filename"] = fn
            new_rows.append(new_row)

    # 保存
    with (train_out/"labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["filename", "roll", "pitch", "yaw"])
        w.writeheader(); w.writerows(list(meta.values()) + new_rows)

if __name__ == "__main__":
    main()