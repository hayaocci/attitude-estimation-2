#!/usr/bin/env python3
# =============================================================
# eval_hardcoded.py – 指定 run を「新しく作る eval_ フォルダ」に保存
# =============================================================
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, List

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from math import ceil


# ---------- ★ ここだけ書き換えて使う --------------------------
LOG_DIR       = Path("logs") / "type-2" / "ResNet18Scaled" \
                / "rgb_sz56_maxpool" / "20250730_1733_exp53"

DATASET_TYPE  = "type-7"      # テストに使うデータセット種別
SPLIT_NAME    = "valid"       # "valid" / "test" など

TOP_K_SPEC    = "1-5"       # "30" | "5,10,20" | "10-30"
                              # 例: "10-30" → Rank10〜30 を 1 枚に
BATCH_SZ      = 128
NUM_WORKERS   = 4
# --------------------------------------------------------------

# ------------------------------------------------------------------
# 角度ヘルパ
# ------------------------------------------------------------------
def sincos2deg(v: torch.Tensor):
    return (torch.atan2(v[..., 0], v[..., 1]) * 180.0 / math.pi) % 360.0

def circular_error(pred_deg: torch.Tensor, true_deg: torch.Tensor):
    return ((pred_deg - true_deg + 180.0) % 360.0 - 180.0).abs()

def circular_distance(a: float, b: pd.Series | np.ndarray) -> np.ndarray:
    diff = (b - a + 180.0) % 360.0 - 180.0
    return np.abs(diff.values)

# ------------------------------------------------------------------
# ResNet18Scaled (学習時と同一)
# ------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet18Scaled(nn.Module):
    def __init__(self, in_ch=3, out_dim=2, width_mult=1.0,
                 hidden_dim=256, dropout_p=0.3):
        super().__init__()
        base_ch = np.array([64,128,256,512])
        chs = np.maximum(1,(base_ch*width_mult).astype(int)).tolist()
        self.inplanes = chs[0]
        self.conv1 = nn.Conv2d(in_ch, chs[0], 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(chs[0])
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3,2,1)
        self.layer1 = self._make_layer(chs[0],2,1)
        self.layer2 = self._make_layer(chs[1],2,2)
        self.layer3 = self._make_layer(chs[2],2,2)
        self.layer4 = self._make_layer(chs[3],2,2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chs[3], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, out_dim),
        )
    def _make_layer(self, planes, blocks, stride):
        down=None
        if stride!=1 or self.inplanes!=planes:
            down=nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
        layers=[BasicBlock(self.inplanes,planes,stride,down)]
        self.inplanes=planes
        layers.extend(BasicBlock(self.inplanes,planes) for _ in range(1,blocks))
        return nn.Sequential(*layers)
    def forward(self,x):
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x=self.layer1(x);x=self.layer2(x);x=self.layer3(x);x=self.layer4(x)
        x=self.avgpool(x)
        v=self.fc(x).view(x.size(0),-1,2)
        v=nn.functional.normalize(v,dim=2)
        return v.view(v.size(0),-1)

def parse_width_from_name(name, default=1.0):
    try: return float(name.split("_")[-1].replace("p","."))
    except: return default

def get_model(name,out_dim,h=256,p=0.3):
    return ResNet18Scaled(3,out_dim,parse_width_from_name(name),h,p)

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class ImageRegressionDataset(Dataset):
    def __init__(self,img_dir:Path,img_size:int):
        df=pd.read_csv(img_dir.parent/"labels.csv")
        self.paths=[img_dir/f for f in df["filename"]]
        self.targets=df["roll"].astype(np.float32).values
        mean,std=[0.5]*3,[0.5]*3
        self.tfm=transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)])
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        img=Image.open(self.paths[i]).convert("RGB")
        return self.tfm(img), self.targets[i], self.paths[i].name

# ------------------------------------------------------------------
# パス生成ヘルパ
# ------------------------------------------------------------------
def build_test_dir(cfg:Dict, dataset_type:str, split:str)->Path:
    size=cfg["IMG_SIZE"][0]
    mode=cfg["INPUT_MODE"]
    resize=cfg.get("RESIZE_MODE","area")
    return Path("datasets")/dataset_type/"cache"/mode/f"sz{size}_{resize}"/split/"imgs"

def get_test_tag(test_dir:Path)->str:
    parts=list(test_dir.resolve().parts)
    if "datasets" in parts:
        idx=parts.index("datasets")
        if idx+1<len(parts): return parts[idx+1]
    return test_dir.parent.name

# ------------------------------------------------------------------
# Rank 指定文字列をリストに変換
# ------------------------------------------------------------------
def parse_k_spec(spec:str)->List[int]:
    spec=spec.strip()
    if "-" in spec:
        lo,hi=map(int,spec.split("-"))
        return list(range(lo,hi+1))
    if "," in spec:
        return [int(s) for s in spec.split(",") if s.strip()]
    return [int(spec)]

# ------------------------------------------------------------------
# Rank リストを 2 行×n 列グリッドで保存
# ------------------------------------------------------------------
def save_ranklist_confused_grid(df:pd.DataFrame,test_dir:Path,
                                out_png:Path, rank_list:List[int],
                                img_size:int=224):
    df_sorted=df.sort_values("err_roll",ascending=False).reset_index()
    pairs=[]
    for r in rank_list:
        if r<1 or r>len(df_sorted): continue
        row_err=df_sorted.iloc[r-1]
        pred_ref=row_err["pred_roll"]
        dist=((df["true_roll"]-pred_ref+180)%360-180).abs()
        dist[row_err["index"]]=np.inf
        idx_conf=dist.idxmin()
        row_conf=df.loc[idx_conf]
        pairs.append((row_err,row_conf))
    n=len(pairs)
    if n==0: return
    dpi=100
    fig_w,fig_h=n*img_size/dpi,2*img_size/dpi
    fig,axes=plt.subplots(2,n,figsize=(fig_w,fig_h),dpi=dpi)
    if n==1: axes=np.array([[axes[0]],[axes[1]]])
    for c,(row_err,row_conf) in enumerate(pairs):
        rank_no=rank_list[c]
        ax_t=axes[0,c]; ax_b=axes[1,c]
        ax_t.imshow(mpimg.imread(test_dir/row_err["filename"])); ax_t.axis("off")
        ax_t.set_title(f"Rank {rank_no}\nerr {row_err['err_roll']:.2f}°\n"
                       f"true {row_err['true_roll']:.1f}°\npred {row_err['pred_roll']:.1f}°",
                       fontsize=8,loc="left",pad=6)
        ax_b.imshow(mpimg.imread(test_dir/row_conf["filename"])); ax_b.axis("off")
        ax_b.set_title(f"Most Likely Misrecognized\ntrue {row_conf['true_roll']:.1f}°",
                       fontsize=8,loc="left",pad=6)
    plt.subplots_adjust(wspace=0.03,hspace=0.20,
                        left=0.02,right=0.98,top=0.95,bottom=0.02)
    fig.savefig(out_png,bbox_inches="tight"); plt.close(fig)
    print(f"Saved ranks {rank_list} grid → {out_png}")

# ------------------------------------------------------------------
# 極端サンプル 2 枚図
# ------------------------------------------------------------------
# =============================================================
# 最大／最小誤差サンプルを 1 枚にまとめて保存（matplotlib版）
# =============================================================
# =============================================================
# 最大／最小誤差サンプルを 1 枚に保存（画像セル : 余白セル = 98 : 2）
# =============================================================
# =============================================================
# 最大／最小誤差サンプルを 1 枚に保存（画像セル : 余白セル = 98 : 2）
# =============================================================
# ------------------------------------------------------------------
# 最大／最小誤差サンプル 2 枚を 1 枚の PNG に出力
#   ・セル幅・高さを物理インチで固定（cell_inch）
#   ・画像間ギャップはセル幅の gap_ratio （＝2 %）で確保
#   ・フォントサイズもセル幅に連動
# ------------------------------------------------------------------
from matplotlib.gridspec import GridSpec

def save_extremes_fig(df: pd.DataFrame, test_dir: Path,
                      out_png: Path,
                      cell_inch: float = 2.2,      # 画像 1 枚の幅・高さ
                      gap_ratio: float = 0.02,     # 余白 = 画像幅×2 %
                      dpi: int = 300):
    """
    cell_inch : 画像 1 枚の見た目サイズ（インチ）
    gap_ratio : セル幅に対する余白比 (0.02 → 2 %)
    dpi       : 出力解像度
    ※ img_size（元画像ピクセル寸法）には依存しない
    """
    # ---------- サンプル抽出 -------------------------------------
    idx_max = df["err_roll"].idxmax()
    idx_min = df["err_roll"].idxmin()
    rows = [("MAX ERROR", df.loc[idx_max]),
            ("MIN  ERROR", df.loc[idx_min])]

    # ---------- 図サイズ計算 -------------------------------------
    gap_in  = cell_inch * gap_ratio
    fig_w   = 2 * cell_inch + gap_in
    fig_h   = cell_inch

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs  = GridSpec(1, 3, figure=fig,
                   width_ratios=[1, gap_ratio, 1])

    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 2])]

    # フォントサイズはセル幅に対して相対
    fsize = 8 * cell_inch / 2.2      # 基準 2.2inch → 8pt

    # ---------- 描画ループ ---------------------------------------
    for ax, (label, rec) in zip(axes, rows):
        ax.imshow(mpimg.imread(test_dir / rec["filename"]))
        ax.axis("off")
        ax.set_title(
            f"{label}\n"
            f"err  {rec['err_roll']:.2f}°\n"
            f"true {rec['true_roll']:.1f}°\n"
            f"pred {rec['pred_roll']:.1f}°",
            fontsize=fsize,
            loc="left", pad=7
        )

    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved extremes figure → {out_png}  "
          f"({fig_w:.2f}×{fig_h:.2f} inch @ {dpi}dpi)")




# =============================================================
# 最大誤差サンプルと「pred_roll に最も近い true_roll」をもつ別サンプル
# =============================================================
def circular_distance(a: float, b: pd.Series | np.ndarray) -> np.ndarray:
    """角度 a と b(列ベクトル) の円周距離 |Δθ| を返す (0–180]"""
    diff = (b - a + 180.0) % 360.0 - 180.0
    return np.abs(diff.values)

def save_max_and_confused_fig(df: pd.DataFrame, test_dir: Path,
                              out_png: Path, img_size: int = 224):
    """
    左 : err_roll が最大のサンプル（err, true, pred を表示）
    右 : その pred_roll に最も近い true_roll をもつ別サンプル
         （true のみ表示）
    """
    # ---------- ① 最大誤差サンプル ----------
    idx_max = df["err_roll"].idxmax()
    row_max = df.loc[idx_max]
    pred_ref = row_max["pred_roll"]

    # ---------- ② pred_roll に最も近い true_roll ----------
    df_other = df.drop(idx_max)
    dist = circular_distance(pred_ref, df_other["true_roll"])
    idx_conf = df_other.index[dist.argmin()]
    row_conf = df.loc[idx_conf]

    rows = [("MAX ERROR", row_max), ("CLOSE TRUE", row_conf)]

    # ---------- Figure ----------
    dpi = 100
    fig_w, fig_h = 2 * img_size / dpi, img_size / dpi
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi)

    for ax, (label, r) in zip(axes, rows):
        ax.imshow(mpimg.imread(test_dir / r["filename"]))
        ax.axis("off")

        if label == "MAX ERROR":
            title = (f"{label}\n"
                     f"err  = {r['err_roll']:.2f}°\n"
                     f"true = {r['true_roll']:.1f}°\n"
                     f"pred = {r['pred_roll']:.1f}°")
        else:  # CLOSE TRUE
            title = (f"{label}\n"
                     f"true = {r['true_roll']:.1f}°")

        ax.set_title(title, fontsize=10, loc="left", pad=10)

    plt.subplots_adjust(wspace=0.03, left=0.02, right=0.98,
                        top=0.88, bottom=0.02)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved max-and-confused figure → {out_png}")

# =============================================================
# 誤差 Top-k と「pred_roll に最も近い true_roll」ペアを
# 2 行 × k 列 (縦ペア) に並べて保存
# =============================================================
def save_topk_confused_grid(df: pd.DataFrame,
                            test_dir: Path,
                            out_png: Path,
                            k: int = 5,
                            img_size: int = 224):
    """
    上段 (row 0) : err_roll 上位 k サンプル
    下段 (row 1) : それぞれの pred_roll と true_roll が最も近い
                   “別” サンプル
    ・上段には err/true/pred を表示
    ・下段には true だけ表示
    ・k は任意 (1 ≤ k ≤ len(df))
    """

    # ---------------- Top-k 誤差行の index を取得 ----------------
    k = max(1, min(k, len(df)))              # 安全にクリップ
    top_idx = df["err_roll"].nlargest(k).index.tolist()

    # ---------------- ペア (row_err, row_conf) を作成 ------------
    pairs = []
    for idx_max in top_idx:
        row_err  = df.loc[idx_max]
        pred_ref = row_err["pred_roll"]      # 基準角 (deg)

        # ❶ 全行で円周距離 |Δθ| を計算
        dist = ((df["true_roll"] - pred_ref + 180) % 360 - 180).abs()

        # ❷ 誤差最大行は候補から除外
        dist[idx_max] = np.inf

        # ❸ 残りで距離最小の行を取得
        idx_conf = dist.idxmin()
        row_conf = df.loc[idx_conf]

        pairs.append((row_err, row_conf))

    # ---------------- Figure を作成 ----------------
    dpi   = 100
    fig_w = k * img_size / dpi
    fig_h = 2 * img_size / dpi
    fig, axes = plt.subplots(2, k, figsize=(fig_w, fig_h), dpi=dpi)

    if k == 1:                               # k=1 でも 2×k 配列化
        axes = np.array([[axes[0]], [axes[1]]])

    for c, (row_err, row_conf) in enumerate(pairs):
        # ---- 上段：誤差最大サンプル ---------------------------------
        ax_top = axes[0, c]
        ax_top.imshow(mpimg.imread(test_dir / row_err["filename"]))
        ax_top.axis("off")
        ax_top.set_title(
            f"Rank {c+1}\n"
            f"err  {row_err['err_roll']:.2f}°\n"
            f"true {row_err['true_roll']:.1f}°\n"
            f"pred {row_err['pred_roll']:.1f}°",
            fontsize=8, loc="left", pad=6
        )

        # ---- 下段：pred_roll に最も近い true_roll サンプル -----------
        ax_bot = axes[1, c]
        ax_bot.imshow(mpimg.imread(test_dir / row_conf["filename"]))
        ax_bot.axis("off")
        ax_bot.set_title(
            f"Most Likely Misrecognized\ntrue {row_conf['true_roll']:.1f}°",
            fontsize=8, loc="left", pad=6
        )

    # ---------------- 余白調整 & 保存 ----------------
    # plt.subplots_adjust(wspace=0.03,   # 列間
    #                     hspace=0.20,   # 行間 ← ここを大きくすれば縦余白が広がる
    #                     left=0.02, right=0.98,
    #                     top=0.95, bottom=0.02)

    # 新: 画像サイズに対して相対 2 % のギャップをとる
    gap_ratio = 0.02                 # 2 % ギャップ
    n_cols    = len(pairs)           # グリッド列数 (画像ペア数)

    wspace = gap_ratio / n_cols      # 列間 = 画像幅の 2 %
    hspace = gap_ratio / 2           # 行は 2 段なので高さで割る

    plt.subplots_adjust(wspace=wspace,
                        hspace=hspace,
                        left=0.02, right=0.98,
                        top=0.95, bottom=0.02)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Top-{k} confused grid → {out_png}")

def build_test_dir(cfg: Dict, dataset_type: str, split: str) -> Path:
    """
    cfg … config_used.yaml をロードした dict
    dataset_type … 例: "type-3"
    split … "valid", "test" など
    return … /datasets/<dataset_type>/cache/<mode>/sz<size>_<resize>/split/imgs
    """
    size         = cfg["IMG_SIZE"][0]
    mode         = cfg["INPUT_MODE"]            # rgb / gray / bin4
    resize_mode  = cfg.get("RESIZE_MODE", "area")
    return (Path("datasets") / dataset_type / "cache" / mode
            / f"sz{size}_{resize_mode}" / split / "imgs")

# ------------------------------------------------------------------
# Rank リストを 2 行×n 列で保存（画像セル : 余白セル = 98 : 2）
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Rank リストを 2 行×n 列で保存（画像セル : 余白セル = 98 : 2）
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Rank リストを 2 行×n 列で保存（画像セル : 余白セル = 98 : 2）
# ------------------------------------------------------------------
from math import ceil
from matplotlib.gridspec import GridSpec

def save_ranklist_confused_grid(df: pd.DataFrame, test_dir: Path,
                                out_png: Path, rank_list: List[int],
                                cell_inch: float = 2.2, gap_ratio: float = 0.02):
    """
    rank_list   : 1 始まり Rank のリスト（例 [1,2,3,4,5]）
    cell_inch   : 画像 1 枚の幅・高さ（物理インチ）
    gap_ratio   : セル幅に対する余白割合（= 2 % 推奨）
    """
    # ---------- ペア抽出 -----------------------------------------
    df_sorted = df.sort_values("err_roll", ascending=False).reset_index()
    pairs = []
    for r in rank_list:
        if not (1 <= r <= len(df_sorted)):
            continue
        row_err = df_sorted.iloc[r - 1]
        pred_ref = row_err["pred_roll"]
        dist = ((df["true_roll"] - pred_ref + 180) % 360 - 180).abs()
        dist[row_err["index"]] = np.inf
        idx_conf = dist.idxmin()
        pairs.append((row_err, df.loc[idx_conf]))

    n = len(pairs)
    if n == 0:
        print("No valid ranks to display."); return

    # ---------- キャンバス物理サイズ計算 --------------------------
    gap_inch = cell_inch * gap_ratio
    fig_w = n * cell_inch + (n - 1) * gap_inch
    fig_h = 2 * cell_inch + gap_inch         # 2 行

    dpi = 300
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # GridSpec : 画像セルと余白セルを比率で分離
    width_ratios  = [1, gap_ratio] * n
    width_ratios  = width_ratios[: 2 * n - 1]        # 末尾の余白を削除
    height_ratios = [1, gap_ratio, 1]

    gs = GridSpec(3, 2 * n - 1, figure=fig,
                  width_ratios=width_ratios,
                  height_ratios=height_ratios)

    # ---------- 描画 ---------------------------------------------
    fsize = 8 * cell_inch / 2.2                # 基準セル(2.2in)換算
    for c, (row_err, row_conf) in enumerate(pairs):
        # 上段
        ax_t = fig.add_subplot(gs[0, 2 * c])
        ax_t.imshow(mpimg.imread(test_dir / row_err["filename"]))
        ax_t.axis("off")
        ax_t.set_title(
            f"Rank {rank_list[c]}\n"
            f"err {row_err['err_roll']:.2f}°\n"
            f"true {row_err['true_roll']:.1f}°\n"
            f"pred {row_err['pred_roll']:.1f}°",
            fontsize=fsize, loc="left", pad=7)

        # 下段
        ax_b = fig.add_subplot(gs[2, 2 * c])
        ax_b.imshow(mpimg.imread(test_dir / row_conf["filename"]))
        ax_b.axis("off")
        ax_b.set_title(
            f"Most Likely Misrecognized\ntrue {row_conf['true_roll']:.1f}°",
            fontsize=fsize, loc="left", pad=7)

    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved ranks {rank_list} grid → {out_png}")



# ------------------------------------------------------------------
# メイン
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# メイン
# ------------------------------------------------------------------
def main():
    # ── ① Run ディレクトリと設定を取得 ─────────────────────────
    run_dir = LOG_DIR.resolve()
    cfg = yaml.safe_load((run_dir / "config_used.yaml").read_text())

    # ── ② テスト用キャッシュパスを構築 ────────────────────────
    TEST_DIR = build_test_dir(cfg, DATASET_TYPE, SPLIT_NAME)
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"TEST_DIR not found: {TEST_DIR}")

    # ── ③ モデルをロード ──────────────────────────────────────
    ckpt = run_dir / "checkpoints" / "best.pth"
    img_sz = cfg["IMG_SIZE"][0]
    single = cfg["OUTPUT_AXES"] == ["roll"]
    out_dim = 2 if single else len(cfg["OUTPUT_AXES"])

    model = get_model(cfg["MODEL_NAME"], out_dim,
                      cfg.get("HIDDEN_DIM", 256),
                      cfg.get("DROPOUT_P", 0.3))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # ── ④ データローダ ───────────────────────────────────────
    ds = ImageRegressionDataset(TEST_DIR, img_sz)
    ld = DataLoader(ds, BATCH_SZ, False, num_workers=NUM_WORKERS)

    # ── ⑤ 推論ループ ─────────────────────────────────────────
    records, errs = [], []
    with torch.no_grad():
        for imgs, tgt_deg, names in tqdm(ld, desc="Testing"):
            imgs = imgs.to(device)
            out = model(imgs)

            if single:
                pred = sincos2deg(out.cpu())
                err = circular_error(pred, torch.tensor(tgt_deg))
            else:
                pred = out.squeeze().cpu()
                err = (pred - torch.tensor(tgt_deg)).abs()

            errs.extend(err.numpy())
            for n, t, p, e in zip(names, tgt_deg, pred.numpy(), err.numpy()):
                records.append({"filename": n,
                                "true_roll": float(t),
                                "pred_roll": float(p),
                                "err_roll":  float(e)})

    # ── ⑥ 結果保存用フォルダ ──────────────────────────────────
    test_tag = get_test_tag(TEST_DIR)
    eval_dir = run_dir / f"eval_{test_tag}"
    figs_dir = eval_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(eval_dir / "eval_results.csv", index=False)

    # ── ⑦ 可視化：極端例・Rank グリッド ───────────────────────
    save_extremes_fig(
        df, TEST_DIR,
        figs_dir / "extremes.png",
        cell_inch=2.5   # ← 画像 1 枚を 2.5 inch 四方で出力
    )

    rank_list = parse_k_spec(TOP_K_SPEC)   # 例 "1-5"
    save_ranklist_confused_grid(
        df, TEST_DIR,
        figs_dir / f"ranks_{TOP_K_SPEC.replace(',','-')}_grid.png",
        rank_list=parse_k_spec(TOP_K_SPEC),
        cell_inch=2.2       # ← 画像 1 枚の物理サイズをここで指定
    )


    # ── ⑧ メトリクスと散布図 ───────────────────────────────────
    metrics = {"eval_MAE": float(np.mean(errs)),
               "eval_count": len(errs)}
    json.dump(metrics, (eval_dir / "metrics.json").open("w"), indent=2)

    mean_err, max_err = df["err_roll"].mean(), df["err_roll"].max()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(df["true_roll"], df["err_roll"], s=8, alpha=0.6)
    ax.set_xlabel("True theta (deg)")
    ax.set_ylabel("Error theta (deg)")
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(0.12, 0.93,
             f"Average Error: {mean_err:.2f}°    Max Error: {max_err:.2f}°",
             ha="left", va="top", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="white", alpha=0.8, edgecolor="black"))
    fig.savefig(figs_dir / "error_vs_true.png")
    plt.close(fig)

    # ── ⑨ 誤差ヒストグラム ──────────────────────────────────────
    # bins は 0–180° を 5°刻み。刻み幅を変えたい場合は np.arange(0, 181, 好きな刻み幅)
    bins = np.arange(0, 181, 10)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # 細めの帯状ヒストグラム  …… rwidth でバー幅を 60 % に
    ax.hist(df["err_roll"],
            bins=bins,            # ← ここはそのまま (5°刻みなど)
            rwidth=1,           # 0–1 の相対幅。0.6 なら 6 割の細さ
            color="skyblue",
            edgecolor="black",
            alpha=0.85)

    # 平均誤差の破線を追加
    ax.axvline(mean_err, color="red", linestyle="--",
               linewidth=1.5, label=f"Mean = {mean_err:.2f}°")

    ax.set_xlabel("Error theta (deg)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, bins[-1])
    ax.set_title("Distribution of Absolute Roll Error")
    ax.legend()

    plt.tight_layout()
    fig.savefig(figs_dir / "error_hist.png")
    plt.close(fig)

    print(f"Finished. Results saved under: {eval_dir}")

if __name__=="__main__":
    main()
