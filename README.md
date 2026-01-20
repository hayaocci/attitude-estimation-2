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

| 項目          | 内容                                                                                |
| ----------- | --------------------------------------------------------------------------------- |
| 入力ルート       | `--in_path` で指定した `datasets/type-X` 直下の `cache/` フォルダ                             |
| 対象サブフォルダ    | `cache/rgb`, `cache/gray`, `cache/bin4`（`--cache_subdirs` で任意選択可）                 |
| 対象 split    | `szXXX_area/<split_name>/imgs`（`--split_name` で `train` / `valid` / `test` などを指定） |
| 対象画像サイズフォルダ | `sz56_area`, `sz112_area`, `sz224_area` など、`sz\d+_area` にマッチするフォルダ                |
| 対象画像ファイル    | `imgs/` 配下の `.png` および `.jpg`                                                     |
| メタデータ       | 各 `szXXX_area/<split>/labels.csv`（`filename, roll, pitch, yaw`）を読み込み・更新           |
| 出力ルート       | `--out_root` の直下に `<in_path の名前>_aug-vN/` を自動採番で作成（例: `type-2_aug-v2`）            |
| 出力ディレクトリ構造  | 元と同じ構造を再現：`cache/<subdir>/szXXX_area/<split>/imgs`, `labels.csv`                  |
| ベース変換       | `iso_noise`, `blur`, `vstrip`, `stretch`（`CONFIG.SELECTED_BASES` で選択）             |
| 派生変換        | 固定 + ランダム BBOX (`*_rbbox`), crop & paste (`*_crop`), クォドラント隠し (`*_hide`)          |
| 出力画像ファイル名   | 元画像名 + `_tag` 形式（例: `0000_iso_noise_rbbox.png`）                                   |
| 並列処理        | `ProcessPoolExecutor` によるマルチプロセス処理（`--workers` で数を指定）                             |
| テストモード      | `--test` で各 `szXXX_area` の先頭 10 枚のみ処理                                             |
| 設定ログ        | 実行時の全パラメータ・有効な拡張内容を `augment_config.yaml` として出力ルート直下に保存                           |


`python scripts/augment_dataset_v5.py --in_path datasets/type-2/cache/rgb/sz224_area --out_root datasets/type-2_aug-rainbow --test`


### augment_dataset_v6.pyについて
このaugmentationは、サイド光の画像に対してのロバスト性を確保するために必要である。  
修正点についていかにまとめる。  

### augmentation_dataset_v10.pyについて
このaugmentationには、画像の縦方向および横法への拡大（引き伸ばし）をするものである。  
パラメータは `strech range` である。  

### 学習&評価の流れ
1. train_lab.py - 学習
2. eval_lab_all.py - 評価用データセットを指定して、評価を行いデータセットを保存する
3. compile_results.py - すべてのexpのデータを1つのcsvにまとめる
4. 目的にあったコードで、評価データをまとめたcsvから抽出＆分析





# lab_logs Evaluation Tools

`lab_logs/expXX` に保存された学習済みモデル (`best.pth`) を  
指定した `datasets/XXX` に対して一括評価し、その結果を集約・解析しやすくするための  
2つのスクリプトを提供します。

- `code_B_eval_all.py` … 各 `expXX` の `best.pth` を指定データセットで評価して結果＆図を保存
- `code_C_compile_results.py` … すべての `expXX` の評価結果＋config情報を1つのCSVに集約

---

## 📁 ディレクトリ前提構成

スクリプトと同階層に、最低限以下がある想定です：

```
./
├── code_B_eval_all.py
├── code_C_compile_results.py
├── lab_logs/
│   ├── exp01/
│   │   ├── config_used.yaml
│   │   └── checkpoints/
│   │       └── best.pth
│   ├── exp02/
│   └── ...
└── datasets/
    ├── type-8/
    │   └── cache/
    │       └── rgb/
    │           └── sz224_area/
    │               ├── valid/
    │               │   ├── imgs/
    │               │   └── labels.csv
    │               └── ...
    └── ...
```

---

# ✅ **1. code_B_eval_all.py — 評価実行 & 図作成**

### 🎯 概要

指定したデータセット（例：`type-8`）を使い  
`lab_logs/expYY` ～ `expZZ` の `best.pth` を一括評価し、

- `eval_results_kf.csv`
- 評価可視化画像（PNG）

を `expXX` の内部に自動生成します。

---

### 🧾 **使い方**

基本形：

```bash
python code_B_eval_all.py --dataset <DATASET_NAME>
```

例：`datasets/type-8` を使用して評価：

```bash
python code_B_eval_all.py --dataset type-8 --split valid
```

指定範囲のみ（例：exp05〜exp12）：

```bash
python code_B_eval_all.py --dataset type-8 --split valid --exp_from 5 --exp_to 12
```

または：

```bash
python code_B_eval_all.py --dataset type-8 --split valid --exp_from exp05 --exp_to exp12
```

---

### 📌 **主な引数**

| 引数 | 必須 | デフォルト | 説明 |
|---|:---:|---|---|
| `--dataset` | ✔️ | - | `datasets/<DATASET>` フォルダ名 |
| `--split` |  | `valid` | 使用する split (`train`,`valid`,`test`) |
| `--exp_from` |  | - | 評価開始 (`5` または `exp05`) |
| `--exp_to` |  | - | 評価終了 (`12` or `exp12`) |
| `--batch_size` |  | `64` | DataLoader batch |
| `--num_workers` |  | `4` | DataLoader workers |
| `--log_root` |  | `lab_logs` | exp があるルート |
| `--dataset_root` |  | `datasets` | dataset ルート |

---

### 📂 **出力構成**

例：`--dataset type-8 --split valid` の場合：

```
lab_logs/
└── exp03/
    └── eval_type-8_valid_kf/
        ├── eval_results_kf.csv
        └── figs/
            ├── extremes.png
            ├── ranks_grid.png
            ├── error_vs_true_compare.png
            └── error_hist.png
```

#### `eval_results_kf.csv` の列：

| 列 | 説明 |
|---|---|
| `filename` | 画像名 |
| `true_roll` | ラベル角度 |
| `pred_roll` | 予測角度 |
| `err_roll` | 絶対誤差 |
| `kf_roll` | EKF 後角度 |
| `kf_err` | EKF 後誤差 |

---

# 📊 **2. code_C_compile_results.py — 結果集約CSV作成**

### 🎯 概要

以下を統合して **全 exp 分の1つのCSV** を作ります：

- `eval_results_kf.csv`
- `config_used.yaml`

出力は：

```
lab_logs/compilation/comp_<DATASET>_<SPLIT>_kf.csv
```

---

### 🧾 **使い方**

```bash
python code_C_compile_results.py --dataset type-8 --split valid
```

---

### 🗂 **集約内容**

`comp_*.csv` は画像1枚ごとに以下を統合：

#### （A）評価結果列：

| 列 | 説明 |
|---|---|
| `filename` | 画像名 |
| `true_roll` | ラベル角度 |
| `pred_roll` | 推定角度 |
| `err_roll` | 誤差 |
| `kf_roll` | EKF後角度 |
| `kf_err` | EKF後誤差 |

#### （B）config情報（抽出項目）

| config key | CSV列名 | 値例 |
|---|---|---|
| `id` | `exp_id` | `exp03` |
| `BATCH_SIZE` | `BATCH_SIZE` | `128` |
| `DROPOUT_P` | `DROPOUT_P` | `0.3` |
| `IMG_SIZE` | `IMG_SIZE` | `224` |
| `INPUT_MODE` | `INPUT_MODE` | `rgb` |
| `MAX_LR` | `MAX_LR` | `0.001` |
| `WEIGHT_DECAY` | `WEIGHT_DECAY` | `0.05` |
| `TRAIN_DATASET_ROOT` | `TRAIN_DATASET_ROOT` | `type-2_aug-v2` |

※ `TRAIN_DATASET_ROOT` は `datasets/XXX` の `XXX` だけ抽出されます。

---

# 🔍 `--split valid` とは？

`datasets/<DATASET>/cache/.../<split>/imgs` の `<split>` を指定します。

3種類に対応：

| split | 説明 |
|---|---|
| `train` | 学習用 |
| `valid` | 学習時の検証用（early stopping 等） |
| `test` | 最終評価用 |

例：

```bash
--split valid
```

の場合、参照するフォルダは：

```
datasets/<DATASET>/cache/<INPUT_MODE>/sz<IMG>_<RESIZE>/valid/imgs/
```

同階層の `labels.csv` がラベルになります。

---

## 👌 最後に

この2つのスクリプトにより：

- **モデル横断比較**
- **ハイパラ vs 誤差分析**
- **学習データ差分の影響評価**

が容易になります。

