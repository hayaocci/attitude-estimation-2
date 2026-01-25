import argparse
from pathlib import Path
import shutil
from typing import Optional, Set, Tuple
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm


SIZES    = [56, 112, 224]
CHANNELS = ["rgb", "gray", "bin4"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

INTERPOLATION_MAP = {
    "nearest": cv2.INTER_NEAREST,
    "linear":  cv2.INTER_LINEAR,
    "area":    cv2.INTER_AREA,
    "cubic":   cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def max_pool_resize(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Max pooling によるリサイズ（縮小専用）。
    target_size: (th, tw)
    """
    h, w = img.shape[:2]
    th, tw = target_size

    # スケールファクタ
    scale_h, scale_w = th / h, tw / w

    # 拡大の場合は通常の補間（最近傍）を使用
    if scale_h >= 1.0 and scale_w >= 1.0:
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_NEAREST)

    # 縮小の場合のみ max pooling
    pool_h = max(1, int(h / th))
    pool_w = max(1, int(w / tw))

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
    """アスペクト比を保持して正方形にリサイズし、中央に配置する。

    - size: 出力の一辺の長さ (56, 112, 224 など)
    - method: "nearest", "linear", "area", "cubic", "lanczos", "maxpool"
    """
    h, w = img.shape[:2]
    is_grayscale = img.ndim == 2

    # 既に正方形で目標サイズと同じ場合はそのまま返す
    if h == w == size:
        return img

    # 正方形
    if h == w:
        if method == "maxpool":
            resized = max_pool_resize(img, (size, size))
        else:
            interpolation = INTERPOLATION_MAP[method]
            resized = cv2.resize(img, (size, size), interpolation=interpolation)
        return resized

    # 長方形: 長辺を size に合わせてスケーリング
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    if method == "maxpool":
        resized = max_pool_resize(img, (nh, nw))
    else:
        interpolation = INTERPOLATION_MAP[method]
        resized = cv2.resize(img, (nw, nh), interpolation=interpolation)

    actual_h, actual_w = resized.shape[:2]

    # 正方形キャンバスを作成（元画像の次元に合わせる）
    if is_grayscale:
        canvas = np.zeros((size, size), dtype=img.dtype)
    else:
        canvas = np.zeros((size, size, img.shape[2]), dtype=img.dtype)

    # 中央配置の座標計算
    y0 = (size - actual_h) // 2
    x0 = (size - actual_w) // 2

    y1 = min(y0 + actual_h, size)
    x1 = min(x0 + actual_w, size)
    actual_h = y1 - y0
    actual_w = x1 - x0

    canvas[y0:y1, x0:x1] = resized[:actual_h, :actual_w]

    return canvas


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """BGR画像をLab変換し、Lチャンネル（輝度）を返す。"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    return lab[:, :, 0]


def to_bin4(gray: np.ndarray) -> np.ndarray:
    """Lチャンネル画像（0-255）を4値化して 0, 85, 170, 255 にする。"""
    return (np.digitize(gray, [10, 30, 120], right=False) * 85).astype(np.uint8)


def save_if_not_exists(img: np.ndarray, path: Path) -> None:
    """ファイルが存在しない場合のみ保存する。"""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def apply_gaussian_blur_if_needed(img_bgr: np.ndarray, blur: bool, kernel_size: int) -> np.ndarray:
    """blur フラグとカーネルサイズに応じて GaussianBlur を適用。"""
    if not blur:
        return img_bgr

    k = int(kernel_size)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1  # 偶数の場合は次の奇数に丸める

    return cv2.GaussianBlur(img_bgr, (k, k), sigmaX=0)


def process_image(
    path: Path,
    dst_cache_root: Path,
    resize_mode: str,
    blur: bool,
    blur_kernel: int,
    outputs_filter: Optional[Set[Tuple[str, int]]],
) -> None:
    """単一画像の処理。

    - path: src_dataset_dir/raw/... にある画像
    - dst_cache_root: dst_dataset_dir/cache
    - outputs_filter: None の場合は全組み合わせ、
                      それ以外は (channel, size) のセットでフィルタ
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] 読み込み失敗: {path}")
        return

    # パスから split(train/valid) を推定 (…/raw/train/imgs/xxx.png を想定)
    parts = path.parts
    if len(parts) < 3:
        print(f"[SKIP] パス構造が想定外: {path}")
        return

    split = parts[-3]
    if split not in {"train", "valid"}:
        print(f"[SKIP] train/valid 以外: {path}")
        return

    stem = path.stem

    # --- ブラーを適用（必要なら） ---
    img_bgr = apply_gaussian_blur_if_needed(img, blur=blur, kernel_size=blur_kernel)

    # --- Lab-L グレイスケール ＆ 4値化 ---
    gray = to_gray(img_bgr)
    bin4 = to_bin4(gray)

    # --- 各サイズ × 各チャンネルで保存 ---
    for sz in SIZES:
        postfix = f"sz{sz}_{resize_mode}"

        # RGB
        if outputs_filter is None or ("rgb", sz) in outputs_filter:
            base_rgb = dst_cache_root / "rgb" / postfix / split / "imgs"
            out_path_rgb = base_rgb / f"{stem}.png"
            resized_rgb = resize_square(img_bgr, sz, resize_mode)
            save_if_not_exists(resized_rgb, out_path_rgb)

        # Grayscale (Lab-L)
        if outputs_filter is None or ("gray", sz) in outputs_filter:
            base_gray = dst_cache_root / "gray" / postfix / split / "imgs"
            out_path_gray = base_gray / f"{stem}.png"
            resized_gray = resize_square(gray, sz, resize_mode)
            resized_gray_bgr = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2BGR)
            save_if_not_exists(resized_gray_bgr, out_path_gray)

        # Binary 4-level
        if outputs_filter is None or ("bin4", sz) in outputs_filter:
            base_bin4 = dst_cache_root / "bin4" / postfix / split / "imgs"
            out_path_bin4 = base_bin4 / f"{stem}.png"
            resized_bin4 = resize_square(bin4, sz, resize_mode)
            resized_bin4_bgr = cv2.cvtColor(resized_bin4, cv2.COLOR_GRAY2BGR)
            save_if_not_exists(resized_bin4_bgr, out_path_bin4)


def copy_labels(
    src_raw_root: Path,
    dst_cache_root: Path,
    resize_mode: str,
    outputs_filter: Optional[Set[Tuple[str, int]]],
) -> None:
    """ラベルファイル (labels.csv) を src の raw から dst の cache 側にコピーする。

    outputs_filter が指定されている場合は、その (channel, size) の組み合わせに
    対応するディレクトリにのみコピーする。
    """
    for split in ("train", "valid"):
        src = src_raw_root / split / "labels.csv"
        if not src.exists():
            print(f"[WARN] labels.csv が見つかりません: {src}")
            continue

        for sz in SIZES:
            postfix = f"sz{sz}_{resize_mode}"
            for ch in CHANNELS:
                if outputs_filter is not None and (ch, sz) not in outputs_filter:
                    continue
                dst_dir = dst_cache_root / ch / postfix / split
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst_dir / "labels.csv")


def convert(
    src_dataset_dir: Path,
    dst_dataset_dir: Path,
    resize_mode: str,
    blur: bool,
    blur_kernel: int,
    outputs_filter: Optional[Set[Tuple[str, int]]],
    num_workers: int,
) -> None:
    """メイン変換処理。

    - src_dataset_dir: 元データセット (例: datasets/type-08)
    - dst_dataset_dir: 出力データセット (例: datasets/type-08_lab_blur_3)
    - num_workers: 並列処理に使用するプロセス数
    """
    src_raw_root = src_dataset_dir / "raw"
    if not src_raw_root.exists():
        raise FileNotFoundError(f"raw フォルダが見つかりません: {src_raw_root}")

    dst_cache_root = dst_dataset_dir / "cache"

    imgs = [p for p in src_raw_root.rglob("*") if p.suffix.lower() in IMG_EXTS]

    print(f"[INFO] 入力データセット: {src_dataset_dir}")
    print(f"[INFO] 出力データセット: {dst_dataset_dir}")
    print(f"[INFO] 対象画像枚数: {len(imgs)}")
    print(f"[INFO] resize_mode = {resize_mode}, blur = {blur}, blur_kernel = {blur_kernel}")
    print(f"[INFO] num_workers = {num_workers}")

    if outputs_filter is None:
        print("[INFO] 出力組み合わせ: 全チャンネル・全サイズ (rgb/gray/bin4 × 56/112/224)")
    else:
        combos_str = ", ".join(
            f"{ch}{sz}" for ch, sz in sorted(outputs_filter, key=lambda x: (CHANNELS.index(x[0]), x[1]))
        )
        print(f"[INFO] 出力組み合わせ (フィルタ): {combos_str}")

    if len(imgs) == 0:
        print("[WARN] 対象画像が見つかりませんでした。何も処理せず終了します。")
        return

    # 並列実行のための部分適用
    worker = partial(
        process_image,
        dst_cache_root=dst_cache_root,
        resize_mode=resize_mode,
        blur=blur,
        blur_kernel=blur_kernel,
        outputs_filter=outputs_filter,
    )

    if num_workers <= 1:
        # シングルプロセスで逐次処理
        for p in tqdm(imgs, desc=f"Converting ({resize_mode})"):
            worker(p)
    else:
        # マルチプロセスで並列処理
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # executor.map を tqdm でラップして進捗表示
            list(
                tqdm(
                    executor.map(worker, imgs, chunksize=10),
                    total=len(imgs),
                    desc=f"Converting ({resize_mode})",
                )
            )

    # labels.csv のコピーは最後に1回だけメインプロセスで実施
    copy_labels(src_raw_root, dst_cache_root, resize_mode, outputs_filter)
    print("✅ 完了")


def parse_outputs(outputs_list: Optional[list[str]]) -> Optional[Set[Tuple[str, int]]]:
    """--outputs で指定された文字列から (channel, size) のセットを作る。

    例:
        ["rgb224", "gray112"] -> {("rgb", 224), ("gray", 112)}

    未指定 (None や空リスト) の場合は None を返し、
    その場合は全組み合わせを出力する。
    """
    if not outputs_list:
        return None

    allowed: Set[Tuple[str, int]] = set()

    for token in outputs_list:
        token = token.strip()
        if not token:
            continue

        if token.startswith("rgb"):
            ch = "rgb"
            size_str = token[3:]
        elif token.startswith("gray"):
            ch = "gray"
            size_str = token[4:]
        elif token.startswith("bin4"):
            ch = "bin4"
            size_str = token[4:]
        else:
            raise ValueError(
                f"--outputs の指定が不正です: '{token}' "
                f"(先頭は rgb / gray / bin4 のいずれかにしてください)"
            )

        if not size_str.isdigit():
            raise ValueError(f"--outputs のサイズ指定が不正です: '{token}'")

        size = int(size_str)
        if size not in SIZES:
            raise ValueError(
                f"--outputs で指定されたサイズがサポート外です: '{token}' "
                f"(許可サイズ: {SIZES})"
            )

        allowed.add((ch, size))

    if not allowed:
        return None

    return allowed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "raw 画像 (Lab-L + bin4 対応) を rgb/gray/bin4 caches に変換するスクリプト。\n"
            "入力データセット (例: datasets/type-08) の raw/ を読み込み、"
            "出力データセット (例: datasets/type-08_lab, datasets/type-08_lab_blur_3) に cache/ を生成する。"
        )
    )
    ap.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="元データセットのディレクトリ (例: datasets/type-08)",
    )
    ap.add_argument(
        "--out_dataset_dir",
        type=Path,
        default=None,
        help=(
            "出力データセットのディレクトリ (任意)。"
            "指定しない場合は、元の名前に '_lab' および '_blur_<k>' を付けたものを自動生成する。"
        ),
    )
    ap.add_argument(
        "--resize-mode",
        type=str,
        default="area",
        choices=["nearest", "linear", "area", "cubic", "lanczos", "maxpool"],
        help="リサイズ手法（default: area / INTER_AREA）",
    )
    ap.add_argument(
        "--blur",
        action="store_true",
        help="Gaussian ブラーを適用する場合に指定。",
    )
    ap.add_argument(
        "--blur-kernel",
        type=int,
        default=3,
        help="Gaussian ブラーのカーネルサイズ (奇数推奨, default: 3)。",
    )
    ap.add_argument(
        "--outputs",
        nargs="*",
        default=None,
        metavar="CHSZ",
        help=(
            "出力する (channel, size) の組み合わせを指定します。"
            "例: --outputs rgb224 gray112 bin456\n"
            "未指定の場合は rgb/gray/bin4 × 56/112/224 の全組み合わせを出力します。"
        ),
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "並列処理に使用するプロセス数 (default: CPUコア数)。"
            "1 を指定するとシングルプロセスで実行します。"
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    src_dataset_dir = args.dataset_dir.resolve()

    if args.out_dataset_dir is not None:
        dst_dataset_dir = args.out_dataset_dir.resolve()
    else:
        # 自動で type-XX_lab / type-XX_lab_blur_k を決める
        base_name = src_dataset_dir.name
        suffix = "_lab"
        if args.blur:
            suffix += f"_blur_{args.blur_kernel}"
        dst_dataset_dir = src_dataset_dir.parent / f"{base_name}{suffix}"

    outputs_filter = parse_outputs(args.outputs)

    # num_workers が指定されていなければ CPU コア数を使用
    if args.num_workers is None or args.num_workers <= 0:
        num_workers = os.cpu_count() or 1
    else:
        num_workers = args.num_workers

    convert(
        src_dataset_dir=src_dataset_dir,
        dst_dataset_dir=dst_dataset_dir,
        resize_mode=args.resize_mode,
        blur=args.blur,
        blur_kernel=args.blur_kernel,
        outputs_filter=outputs_filter,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()
