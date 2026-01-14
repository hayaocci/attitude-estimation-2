import cv2
import numpy as np
import os
from glob import glob

def stitch_images_horizontally(image_dir, output_path):
    # 指定ディレクトリ内の画像ファイルを取得（拡張子 png に限定）
    image_files = sorted(glob(os.path.join(image_dir, '*.jpg')))

    if not image_files:
        print("No images found.")
        return

    # 画像をすべて読み込む
    images = [cv2.imread(img) for img in image_files]

    # 全ての画像サイズを基準に揃える（最初の画像のサイズに）
    height, width = images[0].shape[:2]
    resized_images = [cv2.resize(img, (width, height)) for img in images]

    # 横に連結
    stitched_image = cv2.hconcat(resized_images)

    # 保存
    cv2.imwrite(output_path, stitched_image)
    print(f"Saved stitched image to {output_path}")

def stitch_images_grid(image_dir, output_path, columns=6):
    image_files = sorted(glob(os.path.join(image_dir, '*.jpg')))
    if not image_files:
        print("No images found.")
        return

    images = [cv2.imread(img) for img in image_files]
    height, width = images[0].shape[:2]
    resized_images = [cv2.resize(img, (width, height)) for img in images]

    # 行数を計算
    rows = (len(resized_images) + columns - 1) // columns

    # 空白画像でパディングして矩形に
    blank = np.zeros_like(resized_images[0])
    while len(resized_images) < rows * columns:
        resized_images.append(blank)

    # 行ごとに hconcat → 最後に vconcat
    row_images = [
        cv2.hconcat(resized_images[i*columns:(i+1)*columns]) for i in range(rows)
    ]
    stitched_image = cv2.vconcat(row_images)

    cv2.imwrite(output_path, stitched_image)
    print(f"Saved grid image to {output_path}")


if __name__ == "__main__":
    # image_dir = 'crd2_adrasj_fly-around-observation-wide_0715'
    image_dir = '../raw/0721_ss10e5_crop'
    output_path = '../raw/0721_ss10e5_stitched.png'
    # stitch_images_horizontally(image_dir, output_path)
    stitch_images_grid(image_dir, output_path, columns=9)