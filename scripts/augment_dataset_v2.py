#!/usr/bin/env python3
"""
augment_cache_dataset.py  (revised)
-----------------------------------
ロール角に応じて 0～2 個の BBOX を決定し、各 BBOX 内の 32–75 % を
単色 / ノイズで部分的に塗りつぶしてデータ拡張するスクリプト。

● BBOX 仕様（画像サイズは 224 想定、必要なら数値を調整してください）
    BBOX-1 : center (112 , 112 − 50·sin(roll)) , size 40×40
             * roll が 0–35° または 325–360° なら **描画しない**
    BBOX-2 : center (112 , 112 + 90·sin(roll)) , size 40×40
             * roll が 135–225° なら **描画しない**

● 入出力のフォルダ構成 / CLI オプションは旧版と同じ
    - --path         : datasets ～ szXYZ_area まで柔軟指定
    - --num_aug      : 1 枚あたり何枚生成するか（元画像 + num_aug）
    - --box_frac     : （未使用になりました）→ 残しても無害
    - --color        : "gray" "black" "white" "noise" か数値 1–3 個
    - --seed         : 乱数シード
"""
from __future__ import annotations
import argparse, random, math, re
from pathlib import Path
import cv2, numpy as np, pandas as pd
from tqdm import tqdm


# ────── 色パース ───────────────────────────────────────────
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


# ────── 新しい塗りつぶしユーティリティ ──────────────────
def get_bboxes(roll_deg: float) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    roll (deg) から描画対象の BBOX を返す。
    戻り値: [(tl, br), ...]  tl=(x1,y1) br=(x2,y2)
    """
    roll_deg %= 360
    roll_rad = math.radians(roll_deg)
    boxes = []

    # BBOX-1
    if not (0 <= roll_deg <= 35 or 325 <= roll_deg <= 360):
        cx1, cy1 = 112, 112 - 50 * math.sin(roll_rad)
        boxes.append(rect_vertices(cx1, cy1, 40, 40))

    # BBOX-2
    if not (135 <= roll_deg <= 225):
        cx2, cy2 = 112, 112 + 90 * math.sin(roll_rad)
        boxes.append(rect_vertices(cx2, cy2, 40, 40))

    return boxes


def rect_vertices(cx: float, cy: float, w: int, h: int):
    """中心 (cx,cy)・幅 w・高さ h → 左上 / 右下"""
    tl = (int(cx - w / 2), int(cy - h / 2))
    br = (int(cx + w / 2), int(cy + h / 2))
    return tl, br


def fill_roi_partial(img: np.ndarray, tl: tuple[int, int], br: tuple[int, int],
                     mode: str, col):
    """
    ROI (tl,br) の 32–75 % をランダム矩形で塗りつぶし
    mode='solid' or 'noise'
    """
    x1, y1 = tl
    x2, y2 = br
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return

    h, w = roi.shape[:2]
    cov   = random.uniform(0.25, 0.75)
    scale = math.sqrt(cov)
    rw    = int(max(1, w * scale * random.uniform(0.9, 1.1)))
    rh    = int(max(1, h * scale * random.uniform(0.9, 1.1)))
    rw, rh = min(rw, w), min(rh, h)

    rx = random.randint(0, w - rw)
    ry = random.randint(0, h - rh)
    sub = roi[ry:ry + rh, rx:rx + rw]

    if mode == "noise":
        sub[:] = np.random.randint(0, 256, sub.shape, dtype=img.dtype)
    else:  # solid
        if img.ndim == 2:          # Gray
            gray_val = col[0]
            sub[:] = gray_val
        else:                      # BGR
            color = col if len(col) == 3 else (col[0],) * 3
            sub[:] = color


# ────── split(train/valid) 拡張 ───────────────────────────
def augment_split(src_split: Path, dst_split: Path,
                  num_aug: int, mode: str, col):
    """
    元画像をコピーしつつ num_aug 個の部分隠し画像 (_occX) を生成。
    """
    img_dir, csv_path = src_split / "imgs", src_split / "labels.csv"
    if not img_dir.is_dir() or not csv_path.is_file():
        return False

    dst_img = dst_split / "imgs"
    dst_img.mkdir(parents=True, exist_ok=True)

    # ---------- オリジナル画像コピー ----------
    for p in img_dir.glob("*.png"):
        cv2.imwrite(str(dst_img / p.name),
                    cv2.imread(str(p), cv2.IMREAD_UNCHANGED))

    df = pd.read_csv(csv_path)
    aug_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=str(src_split.relative_to(src_split.parents[3])),
                       leave=False):
        im = cv2.imread(str(img_dir / row["filename"]), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue

        # ----- ロール角から BBOX を取得 -----
        bboxes = get_bboxes(float(row["roll"]))
        if not bboxes:
            continue  # 描画対象なし → 拡張スキップ

        stem = Path(row["filename"]).stem
        for i in range(num_aug):
            aug = im.copy()
            for tl, br in bboxes:
                fill_roi_partial(aug, tl, br, mode, col)

            new_name = f"{stem}_occ{i}.png"
            cv2.imwrite(str(dst_img / new_name), aug)
            aug_rows.append({"filename": new_name,
                             "roll": row["roll"],
                             "pitch": row["pitch"],
                             "yaw": row["yaw"]})

    pd.concat([df, pd.DataFrame(aug_rows)], ignore_index=True).to_csv(
        dst_split / "labels.csv", index=False)
    return True


# ────── szXYZ_area 列挙（旧版そのまま） ─────────────────
def gather_sz_dirs(top: Path):
    if (top / "train").is_dir() or (top / "valid").is_dir():
        return [top]
    if any(p.is_dir() and p.name.startswith("sz") for p in top.iterdir()):
        return [p for p in top.iterdir() if p.is_dir()]
    if (top / "cache").is_dir():
        return [sz for ch in (top / "cache").iterdir()
                for sz in ch.iterdir() if sz.is_dir()]
    return [sz for t in top.glob("type-*")
            for ch in (t / "cache").iterdir()
            for sz in ch.iterdir() if sz.is_dir()]


def split_type_rel(sz_path: Path) -> tuple[Path, Path]:
    cache_dir = next((p for p in sz_path.parents if p.name == "cache"), None)
    if cache_dir is None:
        raise ValueError(f"'cache' が見つかりません: {sz_path}")

    type_dir = cache_dir.parent
    if not re.match(r"type-\d+", type_dir.name):
        raise ValueError(f"type-* フォルダが不正: {type_dir}")

    rel_path = sz_path.relative_to(cache_dir)
    return type_dir, rel_path


# ────── main ────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True,
                    help="datasets 〜 szXYZ_area まで柔軟に指定")
    ap.add_argument("--num_aug", type=int, default=5,
                    help="各画像から生成する拡張枚数")
    ap.add_argument("--box_frac", type=float, default=0.15,  # ← 未使用になったが互換のため残す
                    help="(legacy) 旧スクリプト互換。無視されます")
    ap.add_argument("--color", nargs="+", default=["black"],
                    help='"gray" "black" "white" "noise" か数値 1〜3 個')
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    mode, col, suffix = parse_color(args.color)
    sz_dirs = gather_sz_dirs(Path(args.path).resolve())

    for sz in sz_dirs:
        type_dir, rel_path = split_type_rel(sz)          # type-x
        ds_root   = type_dir.parent                      # datasets/
        out_cache = ds_root / f"{type_dir.name}_occ_{suffix}" / "cache"
        out_sz    = out_cache / rel_path

        for split in ("train", "valid"):
            augment_split(sz / split, out_sz / split,
                          num_aug=args.num_aug, mode=mode, col=col)


if __name__ == "__main__":
    main()
