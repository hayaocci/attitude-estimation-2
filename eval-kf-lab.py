#!/usr/bin/env python3
from __future__ import annotations
import json, math, cv2, yaml
from pathlib import Path
from typing import Dict, List

import matplotlib; matplotlib.use("Agg") # GUIなし環境用
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

# ============================================================
# ★ 設定項目：ご自身の環境に合わせて変更してください
# ============================================================
LOG_DIR       = Path("lab_logs") / "exp14"  # 学習時の ID
# DATASET_TYPE  = "type-7_painted"  
DATASET_TYPE  = "type-8"                                # テスト用データセット種別
SPLIT_NAME    = "valid"                                 # "valid" または "test"
BATCH_SZ      = 64
NUM_WORKERS   = 4

# Kalman Filter 設定
KF_FPS         = 15.0
KF_SIGMA_Z_DEG = 1.0
KF_SIGMA_A_DEG = 2.0
KF_INIT_STD_TH_DEG  = 20.0
KF_INIT_STD_OM_DEG  = 10.0

# ============================================================
# 1. 角度計算・数学ヘルパー (修正済み)
# ============================================================
def sincos2deg(v: torch.Tensor):
    rad = torch.atan2(v[..., 0], v[..., 1])
    deg = rad * 180.0 / math.pi
    return deg.remainder(360.0)

def circular_error(p, t):
    """NumPy型でも動作するように np.abs を使用"""
    diff = (p - t + 180.0) % 360.0 - 180.0
    return np.abs(diff)

def _wrap_pi(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi

# ============================================================
# 2. モデル定義 (Code 1 と完全に一致)
# ============================================================
class DilatedBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet18Dilated(nn.Module):
    def __init__(self, in_ch=3, out_dim=2, width_mult=1.0, hidden_dim=256, dropout_p=0.3):
        super().__init__()
        base_channels = np.array([64, 128, 256, 512])
        chs = np.maximum(1, (base_channels * width_mult).astype(int)).tolist()
        self.inplanes = chs[0]
        self.conv1 = nn.Conv2d(in_ch, chs[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(chs[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(chs[0], 2, 1, 1)
        self.layer2 = self._make_layer(chs[1], 2, 2, 2)
        self.layer3 = self._make_layer(chs[2], 2, 2, 4)
        self.layer4 = self._make_layer(chs[3], 2, 2, 8)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(chs[3], hidden_dim), nn.ReLU(True), nn.Dropout(dropout_p), nn.Linear(hidden_dim, out_dim))

    def _make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
        layers = [DilatedBasicBlock(self.inplanes, planes, stride, downsample, dilation)]
        self.inplanes = planes
        for _ in range(1, blocks): layers.append(DilatedBasicBlock(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        v = nn.functional.normalize(self.fc(x).view(x.size(0), -1, 2), dim=2)
        return v.view(v.size(0), -1)

# ============================================================
# 3. データセット (Lab 変換対応)
# ============================================================
class ImageRegressionDataset(Dataset):
    def __init__(self, img_dir: Path, img_size: int, color_mode: str = "lab"):
        self.img_dir, self.img_size, self.color_mode = img_dir, img_size, color_mode.lower()
        df = pd.read_csv(self.img_dir.parent / "labels.csv")
        self.paths = [self.img_dir / f for f in df["filename"]]
        self.targets = df["roll"].astype(np.float32).values

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img_pil = Image.open(self.paths[idx]).convert("RGB")
        img_np = np.array(img_pil.resize((self.img_size, self.img_size), Image.LANCZOS))
        if self.color_mode == "lab":
            img_lab = (cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0 - 0.5) / 0.5
            img_tensor = torch.from_numpy(img_lab.transpose(2, 0, 1))
        else:
            img_tensor = transforms.ToTensor()(Image.fromarray(img_np))
            img_tensor = transforms.Normalize([0.5]*3, [0.5]*3)(img_tensor)
        return img_tensor, self.targets[idx], self.paths[idx].name

# ============================================================
# 4. EKF クラス
# ============================================================
class EKFRoll:
    def __init__(self, theta0, omega0=0.0):
        self.x = np.array([float(theta0), float(omega0)])
        self.P = np.diag([math.radians(KF_INIT_STD_TH_DEG)**2, math.radians(KF_INIT_STD_OM_DEG)**2])
        self.R = np.eye(2) * (math.radians(KF_SIGMA_Z_DEG)**2)
        self.sigma_a = math.radians(KF_SIGMA_A_DEG)

    def step(self, z, dt):
        F = np.array([[1.0, dt], [0.0, 1.0]])
        G = np.array([[0.5*dt**2], [dt]])
        self.x = F @ self.x
        self.x[0] = _wrap_pi(self.x[0])
        self.P = F @ self.P @ F.T + (self.sigma_a**2) * (G @ G.T)
        th = self.x[0]
        h = np.array([math.sin(th), math.cos(th)])
        H = np.array([[math.cos(th), 0.0], [-math.sin(th), 0.0]])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ (z - h)
        self.x[0] = _wrap_pi(self.x[0])
        self.P = (np.eye(2) - K @ H) @ self.P
        return self.x.copy()

# ============================================================
# 5. 可視化関数
# ============================================================
def save_extremes_fig(df, test_dir, out_png):
    idx_max, idx_min = df["err_roll"].idxmax(), df["err_roll"].idxmin()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    for ax, idx, label in zip(axes, [idx_max, idx_min], ["MAX ERROR", "MIN ERROR"]):
        row = df.loc[idx]
        ax.imshow(mpimg.imread(test_dir / row["filename"]))
        ax.set_title(f"{label}\nTrue: {row['true_roll']:.1f}° Pred: {row['pred_roll']:.1f}°\nErr: {row['err_roll']:.2f}°")
        ax.axis("off")
    plt.tight_layout(); fig.savefig(out_png); plt.close()

def save_ranklist_grid(df, test_dir, out_png, rank_list):
    df_s = df.sort_values("err_roll", ascending=False).reset_index()
    n = len(rank_list)
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6), dpi=150)
    if n == 1: axes = axes.reshape(2, 1)
    for i, r in enumerate(rank_list):
        row = df_s.iloc[r-1]
        axes[0, i].imshow(mpimg.imread(test_dir / row["filename"]))
        axes[0, i].set_title(f"Rank {r}\nErr: {row['err_roll']:.1f}°", fontsize=9)
        axes[0, i].axis("off")
        # 混同されやすい画像を探索（簡易版）
        dist = ((df["true_roll"] - row["pred_roll"] + 180) % 360 - 180).abs()
        dist[row["index"]] = np.inf
        conf_row = df.loc[dist.idxmin()]
        axes[1, i].imshow(mpimg.imread(test_dir / conf_row["filename"]))
        axes[1, i].set_title(f"Confused with\nTrue: {conf_row['true_roll']:.1f}°", fontsize=8)
        axes[1, i].axis("off")
    plt.tight_layout(); fig.savefig(out_png); plt.close()

# ============================================================
# 6. メイン
# ============================================================
def main():
    run_dir = LOG_DIR.resolve()
    with open(run_dir / "config_used.yaml") as f: cfg = yaml.safe_load(f)
    eval_dir = run_dir / f"eval_{DATASET_TYPE}_{SPLIT_NAME}_kf"
    figs_dir = eval_dir / "figs"; figs_dir.mkdir(parents=True, exist_ok=True)

    # モデル読み込み
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    width = float(cfg["MODEL_NAME"].split("_")[-1].replace("p", ".")) if "_" in cfg["MODEL_NAME"] else 1.0
    model = ResNet18Dilated(width_mult=width, hidden_dim=cfg["HIDDEN_DIM"]).to(device)
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pth", map_location=device))
    model.eval()

    # データセット
    test_path = Path("datasets") / DATASET_TYPE / "cache" / cfg["INPUT_MODE"] / f"sz{cfg['IMG_SIZE'][0]}_{cfg.get('RESIZE_MODE','area')}" / SPLIT_NAME / "imgs"
    ds = ImageRegressionDataset(test_path, cfg["IMG_SIZE"][0], cfg.get("COLOR_MODE", "lab"))
    ld = DataLoader(ds, BATCH_SZ, False, num_workers=NUM_WORKERS)

    # 推論
    results, sincos_list = [], []
    with torch.no_grad():
        for imgs, tgts, names in tqdm(ld, desc="Evaluating"):
            outs = model(imgs.to(device)).cpu()
            sincos_list.extend(outs.numpy().tolist())
            preds = sincos2deg(outs).numpy()
            for n, t, p in zip(names, tgts.numpy(), preds):
                results.append({"filename": n, "true_roll": float(t), "pred_roll": float(p), "err_roll": float(circular_error(p, t))})

    df = pd.DataFrame(results)

    # EKF 適用
    df_kf = df.sort_values("filename").reset_index(drop=True)
    dt = 1.0 / KF_FPS
    kf = EKFRoll(theta0=math.atan2(sincos_list[0][0], sincos_list[0][1]))
    kf_res = []
    for i, row in df_kf.iterrows():
        idx_orig = df[df["filename"] == row["filename"]].index[0]
        state = kf.step(np.array(sincos_list[idx_orig]), dt)
        th_deg = math.degrees(state[0]) % 360.0
        kf_res.append({"filename": row["filename"], "kf_roll": th_deg, "kf_err": float(circular_error(th_deg, row["true_roll"]))})
    
    df = df.merge(pd.DataFrame(kf_res), on="filename")
    df.to_csv(eval_dir / "eval_results_kf.csv", index=False)

    # グラフ作成
    print("📈 Creating graphs...")
    save_extremes_fig(df, test_path, figs_dir / "extremes.png")
    save_ranklist_grid(df, test_path, figs_dir / "ranks_grid.png", [1, 2, 3, 4, 5])
    
    # 散布図
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["true_roll"], df["err_roll"], s=10, alpha=0.5, label="Raw")
    ax.scatter(df["true_roll"], df["kf_err"], s=10, alpha=0.5, label="Kalman")
    ax.set_xlabel("True Roll [deg]"); ax.set_ylabel("Error [deg]"); ax.legend()
    fig.savefig(figs_dir / "error_vs_true_compare.png"); plt.close()

    # ヒストグラム
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["err_roll"], bins=30, alpha=0.5, label=f"Raw (Mean:{df['err_roll'].mean():.2f})")
    ax.hist(df["kf_err"], bins=30, alpha=0.5, label=f"Kalman (Mean:{df['kf_err'].mean():.2f})")
    ax.set_xlabel("Absolute Error [deg]"); ax.set_ylabel("Count"); ax.legend()
    fig.savefig(figs_dir / "error_hist.png"); plt.close()

    print(f"✨ Finished! Results saved in: {eval_dir}")

if __name__ == "__main__":
    main()