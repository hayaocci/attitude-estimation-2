import cv2
import numpy as np
from pathlib import Path

def quantize_to_4_levels(gray_img):
    # 入力画像はグレースケール前提
    bins = [5, 30, 120]
    levels = [0, 160, 220, 255]
    quantized = np.zeros_like(gray_img)

    # 各しきい値に従ってマッピング
    quantized[gray_img < bins[0]] = levels[0]
    quantized[(gray_img >= bins[0]) & (gray_img < bins[1])] = levels[1]
    quantized[(gray_img >= bins[1]) & (gray_img < bins[2])] = levels[2]
    quantized[gray_img >= bins[2]] = levels[3]

    return quantized

def process_images(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    for img_path in image_files:
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"読み込み失敗: {img_path}")
            continue

        quantized = quantize_to_4_levels(image)
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), quantized)
        print(f"保存しました: {output_path}")

input_folder = "../raw/0721_ss10e5_crop"  # 入力フォルダ名
output_folder = "../raw/0721_ss10e5_crop_shodo"  # 出力フォルダ名
process_images(input_folder, output_folder)

