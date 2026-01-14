### 環境の起動  
1. powershellからUbuntuの起動  
`wsl -d Ubuntu-24.04`

2. conda環境のactivate  
`conda activate py312-onnx`


* eval-with-dataset.py  
指定したデータセットと学習済みのモデルを用いて検証を行う

### 暗室画像を新しくデータセット化する手順
1. 画像の位置合わせとクロップ  
./scripts/raw_image_editor.pyを使用する  
使用する際はディレクトリを指定する必要がある  
`raw_image_editor.py {画像フォルダ名}`  
{画像フォルダ名_crop}が出力される。これを正しい方向に回転させる。

2. データセット名とlabels.csvを登録する  
`datasets/type-x/raw/valid/imgs`というディレクトをつくる  
`valid/labels.csv`も忘れずに作る  

3. rawからcacheを作成する  
`scripts/raw2edit.py`を使用する

### augment_dataset_v4.pyの使い方

| Option                        |     Default     | Summary                                      |
| ----------------------------- | :-------------: | -------------------------------------------- |
| `--path DIR`                  |        —        | **Input directory** containing PNG / JPG     |
| `--out DIR`                   |    `aug_out`    | Output directory (auto-create)               |
| `--test`                      |       off       | Augment *5 random images* only (→ 105 files) |
| `--seed N`                    |       `42`      | RNG seed for reproducibility                 |
| **ISO / Photometric**         |                 |                                              |
| `--iso_sigma F`               |     **8.0**     | Gaussian σ for high-ISO noise (≈ ISO 1600)   |
| `--bright LO HI`              |   **0.9 1.1**   | Random brightness scale in `[LO,HI]`         |
| `--blur_k MIN MAX`            |     **3 7**     | Gaussian-blur kernel size (odd, px)          |
| **BBox Masks**                |                 |                                              |
| `--rand_boxes MIN MAX`        |     **1 3**     | # of *random* black boxes                    |
| `--rand_box_wh MIN MAX`       |    **20 60**    | Width / height of those boxes (px)           |
| `--rand_box_area x1 y1 x2 y2` | **0 0 224 224** | Area where random boxes may appear           |
| **Quadrant Hide**             |                 |                                              |
| `--hide_n MIN MAX`            |     **1 3**     | Number of quadrants (out of 4) to black out  |
| **Crop / Zoom**               |                 |                                              |
| `--crop_scale MIN MAX`        |   **0.3 1.0**   | Center-crop size ratio before random resize  |

`python scripts/augment_dataset_v5.py --in_path datasets/type-2/cache/rgb/sz224_area --out_root datasets/type-2_aug-rainbow --test`


### augment_dataset_v6.pyについて
このaugmentationは、サイド光の画像に対してのロバスト性を確保するために必要である。  
修正点についていかにまとめる。  

### augmentation_dataset_v10.pyについて
このaugmentationには、画像の縦方向および横法への拡大（引き伸ばし）をするものである。  
パラメータは `strech range` である。  
