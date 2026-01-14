#!/usr/bin/env python3
# make_gif.py -------------------------------------------------
# images in SRC_DIR  → GIF_FILE
# -------------------------------------------------------------

from pathlib import Path
from PIL import Image
import re

# ===== 固定設定 ==============================================================
SRC_DIR   = Path("../raw/0721_ss10e5_crop")   # クロップ画像フォルダ
GIF_FILE  = Path("../raw/demo_5.gif")      # 出力 GIF ファイル
FPS       = 5                    # フレーム/秒
LOOP      = 0                     # 0: 無限ループ
# ============================================================================

def natural_key(path: Path):
    """
    'img12.png' → ['img', 12, '.png']  のように分割し、数字部分を int 化。
    これでソートすると 1,2,10... が正しい数値順になる。
    """
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', path.stem + path.suffix)]

def make_gif(src_dir: Path, gif_path: Path, fps: int, loop: int):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    frames = sorted((p for p in src_dir.iterdir()
                     if p.suffix.lower() in exts),
                    key=natural_key)

    if not frames:
        raise RuntimeError("対象画像が見つかりません。")

    imgs = [Image.open(f) for f in frames]

    # サイズをそろえる（1枚目基準）
    w0, h0 = imgs[0].size
    imgs = [im if im.size == (w0, h0)
            else im.resize((w0, h0), Image.Resampling.LANCZOS)
            for im in imgs]

    duration = int(1000 / fps)   # フレーム表示時間 [ms]

    imgs[0].save(
        gif_path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )
    print(f"[DONE] GIF 作成 → {gif_path.resolve()}")

if __name__ == "__main__":
    make_gif(SRC_DIR, GIF_FILE, FPS, LOOP)
