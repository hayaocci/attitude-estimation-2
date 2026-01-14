#!/usr/bin/env python3
# =============================================================
# eval_hardcoded.py – 指定 run を「新しく作る eval_ フォルダ」に保存
# （EKF で (sin,cos) を時系列平滑化する処理を追加）
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
LOG_DIR       = Path("logs") / "type-2_aug-random_gray_vs_type-7" / "ResNet18Scaled" \
                / "rgb_sz224_area" / "20250821_0354_exp106"

DATASET_TYPE  = "type-7_painted"      # テストに使うデータセット種別
SPLIT_NAME    = "valid"       # "valid" / "test" など

TOP_K_SPEC    = "1-5"         # "30" | "5,10,20" | "10-30"
BATCH_SZ      = 128
NUM_WORKERS   = 4

# ====== ★ ADD: Kalman Filter params (必要ならここだけ触ればOK) ======
KF_FPS         = 15.0
KF_SIGMA_Z_DEG = 1.0
KF_SIGMA_A_DEG = 2.0
KF_INIT_STD_TH_DEG  = 20.0      # 初期角の不確かさ [deg]
KF_INIT_STD_OM_DEG  = 10.0      # 初期角速度の不確かさ [deg/s]
# ===============================================================

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
# Rank リストを 2 行×n 列グリッドで保存（簡易版）
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
# 極端サンプル 2 枚図（高解像・余白比指定）
# ------------------------------------------------------------------
def save_extremes_fig(df: pd.DataFrame, test_dir: Path,
                      out_png: Path,
                      cell_inch: float = 2.2,      # 画像 1 枚の幅・高さ
                      gap_ratio: float = 0.02,     # 余白 = 画像幅×2 %
                      dpi: int = 300):
    idx_max = df["err_roll"].idxmax()
    idx_min = df["err_roll"].idxmin()
    rows = [("MAX ERROR", df.loc[idx_max]),
            ("MIN  ERROR", df.loc[idx_min])]

    gap_in  = cell_inch * gap_ratio
    fig_w   = 2 * cell_inch + gap_in
    fig_h   = cell_inch

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs  = GridSpec(1, 3, figure=fig,
                   width_ratios=[1, gap_ratio, 1])

    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 2])]

    fsize = 8 * cell_inch / 2.2

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

# ------------------------------------------------------------------
# Top-k 可視化（高解像・余白比指定）
# ------------------------------------------------------------------
def save_topk_confused_grid(df: pd.DataFrame,
                            test_dir: Path,
                            out_png: Path,
                            k: int = 5,
                            img_size: int = 224):
    k = max(1, min(k, len(df)))
    top_idx = df["err_roll"].nlargest(k).index.tolist()

    pairs = []
    for idx_max in top_idx:
        row_err  = df.loc[idx_max]
        pred_ref = row_err["pred_roll"]
        dist = ((df["true_roll"] - pred_ref + 180) % 360 - 180).abs()
        dist[idx_max] = np.inf
        idx_conf = dist.idxmin()
        row_conf = df.loc[idx_conf]
        pairs.append((row_err, row_conf))

    dpi   = 100
    fig_w = k * img_size / dpi
    fig_h = 2 * img_size / dpi
    fig, axes = plt.subplots(2, k, figsize=(fig_w, fig_h), dpi=dpi)

    if k == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for c, (row_err, row_conf) in enumerate(pairs):
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

        ax_bot = axes[1, c]
        ax_bot.imshow(mpimg.imread(test_dir / row_conf["filename"]))
        ax_bot.axis("off")
        ax_bot.set_title(
            f"Most Likely Misrecognized\ntrue {row_conf['true_roll']:.1f}°",
            fontsize=8, loc="left", pad=6
        )

    gap_ratio = 0.02
    n_cols    = len(pairs)
    wspace = gap_ratio / n_cols
    hspace = gap_ratio / 2

    plt.subplots_adjust(wspace=wspace,
                        hspace=hspace,
                        left=0.02, right=0.98,
                        top=0.95, bottom=0.02)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Top-{k} confused grid → {out_png}")

# ------------------------------------------------------------------
# 高解像 Rank グリッド（画像セル : 余白セル = 98 : 2）
# ------------------------------------------------------------------
def save_ranklist_confused_grid(df: pd.DataFrame, test_dir: Path,
                                out_png: Path, rank_list: List[int],
                                cell_inch: float = 2.2, gap_ratio: float = 0.02):
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

    gap_inch = cell_inch * gap_ratio
    fig_w = n * cell_inch + (n - 1) * gap_inch
    fig_h = 2 * cell_inch + gap_inch

    dpi = 300
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    width_ratios  = [1, gap_ratio] * n
    width_ratios  = width_ratios[: 2 * n - 1]
    height_ratios = [1, gap_ratio, 1]

    gs = GridSpec(3, 2 * n - 1, figure=fig,
                  width_ratios=width_ratios,
                  height_ratios=height_ratios)

    fsize = 8 * cell_inch / 2.2
    for c, (row_err, row_conf) in enumerate(pairs):
        ax_t = fig.add_subplot(gs[0, 2 * c])
        ax_t.imshow(mpimg.imread(test_dir / row_err["filename"]))
        ax_t.axis("off")
        ax_t.set_title(
            f"Rank {rank_list[c]}\n"
            f"err {row_err['err_roll']:.2f}°\n"
            f"true {row_err['true_roll']:.1f}°\n"
            f"pred {row_err['pred_roll']:.1f}°",
            fontsize=fsize, loc="left", pad=7)

        ax_b = fig.add_subplot(gs[2, 2 * c])
        ax_b.imshow(mpimg.imread(test_dir / row_conf["filename"]))
        ax_b.axis("off")
        ax_b.set_title(
            f"Most Likely Misrecognized\ntrue {row_conf['true_roll']:.1f}°",
            fontsize=fsize, loc="left", pad=7)

    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved ranks {rank_list} grid → {out_png}")

# =============================================================
# ★ ADD: EKF for roll with (sin,cos) observation
# =============================================================
def _wrap_pi(x: float) -> float:
    return (x + math.pi) % (2*math.pi) - math.pi

class EKFRoll:
    """状態 x=[theta, omega], 観測 z=[sin(theta), cos(theta)] のEKF"""
    def __init__(self, theta0, omega0=0.0,
                 sigma_theta0=math.radians(KF_INIT_STD_TH_DEG),
                 sigma_omega0=math.radians(KF_INIT_STD_OM_DEG),
                 sigma_a=math.radians(KF_SIGMA_A_DEG),
                 sigma_z=math.radians(KF_SIGMA_Z_DEG)):
        self.x = np.array([float(theta0), float(omega0)], dtype=float)
        self.P = np.diag([sigma_theta0**2, sigma_omega0**2])
        self.sigma_a = float(sigma_a)
        self.R = np.eye(2) * (sigma_z**2)

    def predict(self, dt: float):
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=float)
        G = np.array([[0.5*dt**2],
                      [dt]], dtype=float)
        Q = (self.sigma_a**2) * (G @ G.T)
        self.x = F @ self.x
        self.x[0] = _wrap_pi(self.x[0])
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray):  # z = [sin, cos]
        th = self.x[0]
        h  = np.array([math.sin(th), math.cos(th)], dtype=float)
        H  = np.array([[ math.cos(th), 0.0],
                       [-math.sin(th), 0.0]], dtype=float)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - h
        self.x = self.x + K @ y
        self.x[0] = _wrap_pi(self.x[0])
        self.P = (np.eye(2) - K @ H) @ self.P

    def step(self, z: np.ndarray, dt: float):
        self.predict(dt)
        self.update(z)
        return self.x.copy()

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
    sincos_bank: List[List[float]] = []   # ★ ADD: EKF用 (sin,cos) を時系列格納
    with torch.no_grad():
        for imgs, tgt_deg, names in tqdm(ld, desc="Testing"):
            imgs = imgs.to(device)
            out = model(imgs)  # L2正規化済み

            if single:
                # (sin,cos) を保存（順序＝DataLoader順）
                sc = out.cpu().numpy()      # shape: (B,2), 各行 [sin, cos]
                sincos_bank.extend(sc.tolist())

                pred = sincos2deg(out.cpu())
                err  = circular_error(pred, torch.tensor(tgt_deg))
            else:
                pred = out.squeeze().cpu()
                err  = (pred - torch.tensor(tgt_deg)).abs()

            errs.extend(err.numpy())
            for n, t, p, e in zip(names, tgt_deg, pred.numpy(), err.numpy()):
                records.append({"filename": n,
                                "true_roll": float(t),
                                "pred_roll": float(p),
                                "err_roll":  float(e)})

    # ── ⑥ 結果保存用フォルダ ──────────────────────────────────
    test_tag = get_test_tag(TEST_DIR)
    eval_dir = run_dir / f"eval_{test_tag}-kf"
    figs_dir = eval_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(eval_dir / "eval_results.csv", index=False)

    # ── ⑦ EKF を回して列追加・保存 ─────────────────────────────
    if single and len(df) > 0:
        # 1) 時系列順に整列（ここでは filename の辞書順を時系列と仮定）
        df_kf = df.sort_values("filename").reset_index(drop=True)

        # 2) EKF 初期化
        s0, c0 = sincos_bank[0]
        th0 = math.atan2(s0, c0)
        kf = EKFRoll(
            theta0=th0,
            omega0=0.0,
            sigma_theta0=math.radians(KF_INIT_STD_TH_DEG),
            sigma_omega0=math.radians(KF_INIT_STD_OM_DEG),
            sigma_a=math.radians(KF_SIGMA_A_DEG),
            sigma_z=math.radians(KF_SIGMA_Z_DEG),
        )

        # 3) 逐次更新（Δt は固定 1/FPS）
        dt = 1.0 / max(1e-6, KF_FPS)
        kf_list = []
        for (s, c), t_true in zip(sincos_bank, df_kf["true_roll"].values):
            th, om = kf.step(np.array([s, c], dtype=float), dt)
            th_deg = (math.degrees(th)) % 360.0
            err = abs(((th_deg - t_true + 180.0) % 360.0) - 180.0)
            kf_list.append((th_deg, err))

        kf_roll, kf_err = map(list, zip(*kf_list))
        df_kf["kf_roll"] = kf_roll
        df_kf["kf_err"]  = kf_err

        # 4) 元の df へ filename キーでマージして列を付与
        df = df.merge(df_kf[["filename","kf_roll","kf_err"]],
                      on="filename", how="left")

        # 5) 保存
        df.to_csv(eval_dir / "eval_results_kf.csv", index=False)
        print("Saved EKF results →", eval_dir / "eval_results_kf.csv")

    # ── ⑧ 可視化：極端例・Rank グリッド ───────────────────────
    save_extremes_fig(
        df, TEST_DIR,
        figs_dir / "extremes.png",
        cell_inch=2.5
    )

    rank_list = parse_k_spec(TOP_K_SPEC)   # 例 "1-5"
    save_ranklist_confused_grid(
        df, TEST_DIR,
        figs_dir / f"ranks_{TOP_K_SPEC.replace(',','-')}_grid.png",
        rank_list=rank_list,
        cell_inch=2.2
    )

    # ── ⑨ メトリクスと散布図 ───────────────────────────────────
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

    # ── ⑩ 誤差ヒストグラム ─────────────────────────────────────
    bins = np.arange(0, 181, 10)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(df["err_roll"], bins=bins, rwidth=1, color="skyblue",
            edgecolor="black", alpha=0.85)
    ax.axvline(mean_err, color="red", linestyle="--",
               linewidth=1.5, label=f"Mean = {mean_err:.2f}°")
    ax.set_xlabel("|Error| (deg)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, bins[-1])
    ax.set_title("Distribution of Absolute Roll Error")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figs_dir / "error_hist.png")
    plt.close(fig)

    # ── ⑪ （任意）KF後の散布図 ────────────────────────────────
    if "kf_err" in df.columns:
        # 2種類の誤差の平均値を計算
        mean_err_raw = df["err_roll"].mean()
        mean_err_kf  = df["kf_err"].mean()

        # プロットを作成
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        
        # 散布図を描画
        ax.scatter(df["true_roll"], df["err_roll"], s=8, alpha=0.35, label="Raw Error")
        ax.scatter(df["true_roll"], df["kf_err"],  s=8, alpha=0.8,  label="Kalman Error")

        # --- ▼▼▼ ここから変更 ▼▼▼ ---
        # 平均誤差を水平線 (axhline) で表示
        # ax.axhline(mean_err_raw, color="red", linestyle="--", 
        #             linewidth=1.5, label=f"Mean Raw: {mean_err_raw:.2f}°")
                    
        # ax.axhline(mean_err_kf, color="green", linestyle="--", 
        #             linewidth=1.5, label=f"Mean Kalman: {mean_err_kf:.2f}°")
        # --- ▲▲▲ ここまで変更 ▲▲▲ ---

        ax.set_xlabel("True theta (deg)")
        ax.set_ylabel("Error theta (deg)")
        
        # 凡例を表示
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(figs_dir / "error_vs_true_kf_compare.png")
        plt.close(fig)

    print(f"Finished. Results saved under: {eval_dir}")

if __name__=="__main__":
    main()
