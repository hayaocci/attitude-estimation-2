#!/usr/bin/env python3
"""
augment_sz_area_v4.py
---------------------
* szXYZ_area/train/imgs 内の画像を 24 パターンに拡張
  (fbbox パターンは出力しない)
* 56 / 112 / 224 px 画像を自動スケール
* 固定 BBOX・ランダム BBOX の塗り色を CLI で指定
    black | white | gray | random_gray | random_color
"""

from __future__ import annotations
import argparse, random, math, csv, re
from pathlib import Path
from typing import List, Tuple, Dict, Callable
import cv2
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
#  カラー関数
# ──────────────────────────────────────────────────────────────
def make_color_fn(keyword: str) -> Callable[[], Tuple[int, ...]]:
    kw = keyword.lower()
    if kw == "black":
        return lambda: (0,)
    if kw == "white":
        return lambda: (255,)
    if kw == "gray":
        return lambda: (128,)
    if kw == "random_gray":
        return lambda: (random.randint(0, 255),)
    if kw == "random_color":
        return lambda: tuple(random.randint(0, 255) for _ in range(3))
    raise ValueError(f"不正な色キーワード: {keyword}")

# ──────────────────────────────────────────────────────────────
#  パス解決（in_path: …/szXYZ_area）
# ──────────────────────────────────────────────────────────────
def resolve_paths(in_path: Path, out_root: Path) -> tuple[Path, Path]:
    if not re.search(r"sz\d+_area$", in_path.as_posix()):
        raise ValueError("--in_path は sz*_area までを指定してください")
    if not (in_path / "train").is_dir():
        raise ValueError(f"train/ が存在しません: {in_path}")

    type_dir = next((p for p in in_path.parents
                     if re.match(r"type-\w+", p.name)), None)
    if type_dir is None:
        raise ValueError("上位に type-* ディレクトリが見つかりません")

    rel = in_path.relative_to(type_dir)            # cache/rgb/sz224_area
    train_in  = in_path / "train"
    train_out = out_root / rel / "train"
    (train_out / "imgs").mkdir(parents=True, exist_ok=True)
    return train_in, train_out

# ──────────────────────────────────────────────────────────────
#  labels.csv
# ──────────────────────────────────────────────────────────────
def read_labels(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))

def write_labels(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, ["filename", "roll", "pitch", "yaw"])
        w.writeheader(); w.writerows(rows)

# ──────────────────────────────────────────────────────────────
#  スケールファクタ計算
# ──────────────────────────────────────────────────────────────
def get_scale_and_size(in_path: Path) -> tuple[float, int]:
    m = re.search(r"sz(\d+)_area", in_path.as_posix())
    if not m:
        raise ValueError("パス名に sz*_area が見つかりません")
    img_sz = int(m.group(1))
    if img_sz not in (56, 112, 224):
        raise ValueError("サポート外サイズ (56 / 112 / 224)")
    return img_sz / 224.0, img_sz

# ──────────────────────────────────────────────────────────────
#  基本変換ユーティリティ
# ──────────────────────────────────────────────────────────────
def add_iso_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0: return img
    f = img.astype(np.float32) / 255.0
    noisy = np.clip(f + (np.random.poisson(f*255)/255 - f) +
                    np.random.normal(0, sigma/255, f.shape), 0, 1)
    return (noisy*255).astype(img.dtype)

def adjust_brightness(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return img if (lo,hi)==(1.0,1.0) else \
        np.clip(img.astype(np.float32)*random.uniform(lo,hi),0,255).astype(img.dtype)

def gaussian_blur(img: np.ndarray, k_rng: Tuple[int,int]) -> np.ndarray:
    k = random.randrange(k_rng[0]|1, (k_rng[1]+1)|1, 2)
    return cv2.GaussianBlur(img, (k,k), 0)

def hide_quadrants(img: np.ndarray, n_rng: Tuple[int,int]) -> np.ndarray:
    n=random.randint(*n_rng); out=img.copy()
    if n>0:
        h2,w2=out.shape[0]//2,out.shape[1]//2
        quads=[(slice(0,h2),slice(0,w2)),(slice(0,h2),slice(w2,None)),
               (slice(h2,None),slice(0,w2)),(slice(h2,None),slice(w2,None))]
        for idx in random.sample(range(4), n):
            r,c=quads[idx]; out[r,c]=0
    return out

# ──────────────────────────────────────────────────────────────
#  BBOX ユーティリティ（サイズスケール対応）
# ──────────────────────────────────────────────────────────────
class BBoxScaler:
    """画像サイズ 56/112/224 に合わせて定数をスケール"""
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
    # BBOX1
    if not (0<=roll<=35 or 325<=roll<=360):
        tl,br = scaler.vertices(scaler.C,
                                scaler.C - scaler.O1*math.sin(rad))
        boxes.append((tl,br))
    # BBOX2
    if not (135<=roll<=225):
        tl,br = scaler.vertices(scaler.C,
                                scaler.C + scaler.O2*math.sin(rad))
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
    else:           sub[:]=color if len(color)==3 else (color[0],)*3

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
        # overlap check
        (ax1,ay1),(ax2,ay2)=box; overlap=False
        for (bx1,by1),(bx2,by2) in existing+boxes:
            if ax1<bx2 and ax2>bx1 and ay1<by2 and ay2>by1:
                overlap=True; break
        if not overlap: boxes.append(box)
    return boxes

def apply_random_bboxes(img, roll, scaler, args, color_fn):
    out=img.copy()
    existing=get_fixed_bboxes(roll, scaler)
    for tl,br in gen_random_boxes(tuple(args.rand_boxes),
                                  tuple(args.rand_box_wh),
                                  tuple(args.rand_box_area),
                                  existing):
        fill_roi_partial(out, tl, br, color_fn())
    return out

# ──────────────────────────────────────────────────────────────
#  クロップ & ペースト
# ──────────────────────────────────────────────────────────────
def specific_crop_and_paste(img: np.ndarray, crop, vis_thr=0.6):
    cx,cy,w,h = crop; H,W = img.shape[:2]
    x1,y1 = cx-w//2, cy-h//2; x2,y2 = x1+w, y1+h
    cx1,cy1=max(0,x1),max(0,y1); cx2,cy2=min(W,x2),min(H,y2)
    cropped = img[cy1:cy2, cx1:cx2]
    if cropped.size == 0:
        return np.zeros_like(img)

    ch,cw = cropped.shape[:2]
    min_w, min_h = int(cw*vis_thr), int(ch*vis_thr)
    px = random.randint(-(cw-min_w), W-min_w)
    py = random.randint(-(ch-min_h), H-min_h)

    canvas = np.zeros_like(img)
    dx1,dy1 = max(0,px), max(0,py)
    dx2,dy2 = min(W,px+cw), min(H,py+ch)
    sx1,sy1 = dx1-px, dy1-py
    sx2,sy2 = sx1+(dx2-dx1), sy1+(dy2-dy1)
    canvas[dy1:dy2, dx1:dx2] = cropped[sy1:sy2, sx1:sx2]
    return canvas

# ──────────────────────────────────────────────────────────────
#  パイプライン定義（fbbox なし）
# ──────────────────────────────────────────────────────────────
BASE_TRANSFORMS = {
    "orig":        [],
    "iso_noise":   ["iso_noise"],
    "blur":        ["blur"],
    "bright":      ["bright"],
    "bright_iso":  ["bright", "iso_noise"],
    "bright_blur": ["bright", "blur"],
}
DERIVED_SUFFIXES = {       # fbbox は生成しない
    "random_bbox": "rbbox",
    "crop":        "crop",
    "hide":        "hide",
}

def apply_seq(img, ops, args):
    out = img
    for op in ops:
        if op == "iso_noise":
            out = add_iso_noise(out, args.iso_sigma)
        elif op == "bright":
            out = adjust_brightness(out, *args.bright)
        elif op == "blur":
            out = gaussian_blur(out, tuple(args.blur_k))
    return out

def generate_variants(img, roll, scaler, args,
                      fixed_color_fn, rand_color_fn):
    res = {}
    base = {n: apply_seq(img, ops, args) for n,ops in BASE_TRANSFORMS.items()}
    res.update(base)   # 基本 6

    for n, b in base.items():
        fb = apply_fixed_bboxes(b, roll, scaler, fixed_color_fn)

        rb = apply_random_bboxes(fb, roll, scaler, args, rand_color_fn)
        res[f"{n}_{DERIVED_SUFFIXES['random_bbox']}"] = rb

        cp = specific_crop_and_paste(
                 fb, tuple(args.crop_params), args.visibility_threshold)
        res[f"{n}_{DERIVED_SUFFIXES['crop']}"] = cp

        hd = hide_quadrants(b, tuple(args.hide_n))
        res[f"{n}_{DERIVED_SUFFIXES['hide']}"] = hd

    return res  # 24 枚

# ──────────────────────────────────────────────────────────────
#  画像 1 枚処理
# ──────────────────────────────────────────────────────────────
def process_single_image(p: Path, meta, scaler, args,
                         fixed_color_fn, rand_color_fn, out_imgs: Path):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        print("[WARN] 読込失敗:", p); return []
    vars = generate_variants(img, float(meta["roll"]), scaler, args,
                             fixed_color_fn, rand_color_fn)
    stem = p.stem; new = []
    for tag, v in vars.items():
        if np.count_nonzero(v)==0: continue
        fn = f"{stem}_{tag}.png"
        cv2.imwrite(str(out_imgs/fn), v)
        row = meta.copy(); row["filename"] = fn
        new.append(row)
    return new

# ──────────────────────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────────────────────
def main():
    COLORS = ["black","white","gray","random_gray","random_color"]
    pa = argparse.ArgumentParser()
    pa.add_argument("--in_path",  required=True)
    pa.add_argument("--out_root", required=True)
    pa.add_argument("--fixed_color", default="random_color", choices=COLORS)
    pa.add_argument("--rand_color",  default="random_color", choices=COLORS)
    pa.add_argument("--test", action="store_true")
    pa.add_argument("--seed", type=int, default=42)
    # 変換パラメータ (224 基準値)
    pa.add_argument("--iso_sigma", type=float, default=8.0)
    pa.add_argument("--bright",   nargs=2, type=float, default=[0.7,1.3])
    pa.add_argument("--blur_k",   nargs=2, type=int,   default=[3,7])
    pa.add_argument("--rand_boxes",  nargs=2, type=int, default=[1,3])
    pa.add_argument("--rand_box_wh", nargs=2, type=int, default=[20,60])
    pa.add_argument("--rand_box_area", nargs=4, type=int, default=[0,0,224,224])
    pa.add_argument("--hide_n",  nargs=2, type=int, default=[1,3])
    pa.add_argument("--crop_params", nargs=4, type=int, default=[112,112,150,224])
    pa.add_argument("--visibility_threshold", type=float, default=0.6)
    args = pa.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    # 画像サイズスケール
    s, img_sz = get_scale_and_size(Path(args.in_path))
    scaler = BBoxScaler(s)

    # パラメータをスケール（ユーザが override しても相対的に OK）
    def scale_list(lst): return [int(round(v*s)) for v in lst]
    args.rand_box_wh   = scale_list(args.rand_box_wh)
    args.rand_box_area = scale_list(args.rand_box_area)
    args.crop_params   = scale_list(args.crop_params)

    train_in, train_out  = resolve_paths(Path(args.in_path).resolve(),
                                         Path(args.out_root).resolve())
    in_imgs, out_imgs = train_in/"imgs", train_out/"imgs"

    src = sorted(in_imgs.glob("*.png"))+sorted(in_imgs.glob("*.jpg"))
    if not src:
        print("[ERROR] 画像無し"); return
    if args.test and len(src)>=5:
        src = random.sample(src,5); print("[INFO] テスト5枚のみ")

    base_rows = read_labels(train_in/"labels.csv")
    meta = {r["filename"]: r for r in base_rows}

    fixed_color_fn = make_color_fn(args.fixed_color)
    rand_color_fn  = make_color_fn(args.rand_color)

    # オリジナルをコピー
    for p in src:
        dst = out_imgs/p.name
        if not dst.exists():
            cv2.imwrite(str(dst), cv2.imread(str(p), cv2.IMREAD_UNCHANGED))

    new_rows=[]
    bar = tqdm(src, total=len(src), unit="img", dynamic_ncols=True,
               desc=f"augment {img_sz}px (24×)")
    for p in bar:
        meta_row = meta.get(p.name, {"filename":p.name,"roll":0,"pitch":0,"yaw":0})
        new_rows += process_single_image(p, meta_row, scaler, args,
                                         fixed_color_fn, rand_color_fn, out_imgs)

    write_labels(train_out/"labels.csv", base_rows+new_rows)
    print(f"[DONE] 追加 {len(new_rows)} 行 / 出力 {train_out}")

if __name__ == "__main__":
    main()
