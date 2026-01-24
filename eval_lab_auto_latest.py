#!/usr/bin/env python3
"""
eval_lab_auto.py – used_config.yaml に基づき自動評価
=========================================================
- 各 lab_logs/expXX 内の used_config.yaml (or config_used.yaml) を読み込み
- 学習時と同じモデル構成 (ResNet18Normal / ResNet18Dilated, width_mult 等) を再構築
- VALID_DIR を必ず評価
- --use_valid_dirs 指定時のみ VALID_DIRS も追加で評価
- 出力: expXX/eval_<DATASET>_<split>_kf/ に CSV & 図を保存
"""

from __future__ import annotations
import argparse, math, re, cv2, yaml
from pathlib import Path
from typing import Dict, List

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ============================================================
# Math helpers (train_dilated_lab.py と整合)
# ============================================================
def deg2rad(x: torch.Tensor | float):
    return x * math.pi / 180.0

def sincos2deg(v: torch.Tensor):
    rad = torch.atan2(v[..., 0], v[..., 1])
    deg = rad * 180.0 / math.pi
    return deg.remainder(360.0)

def circular_error_np(pred_deg: np.ndarray, true_deg: np.ndarray):
    """
    numpy版の環状誤差（評価用）
    pred_deg, true_deg: shape (N,)
    """
    diff = (pred_deg - true_deg + 180.0) % 360.0 - 180.0
    return np.abs(diff)

def _wrap_pi(x: float):
    return (x + math.pi) % (2 * math.pi) - math.pi

# ============================================================
# Model (train_dilated_lab.py と同じ構造)
# ============================================================
class DilatedBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, 3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet18Base(nn.Module):
    """
    train_dilated_lab.py の ResNet18Base と同じ。
    派生クラス側で layer1〜4 を定義。
    """
    def __init__(self, in_ch: int = 3, out_dim: int = 2, width_mult: float = 1.0,
                 hidden_dim: int = 256, dropout_p: float = 0.3):
        super().__init__()
        base_channels = np.array([64, 128, 256, 512])
        chs = np.maximum(1, (base_channels * width_mult).astype(int)).tolist()
        self.chs = chs
        self.inplanes = chs[0]

        self.conv1 = nn.Conv2d(in_ch, chs[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(chs[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1: nn.Module | None = None
        self.layer2: nn.Module | None = None
        self.layer3: nn.Module | None = None
        self.layer4: nn.Module | None = None

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chs[3], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, out_dim),
        )

    def _make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [DilatedBasicBlock(self.inplanes, planes, stride, downsample, dilation)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(DilatedBasicBlock(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        v = self.fc(x)
        v = v.view(v.size(0), -1, 2)
        v = nn.functional.normalize(v, dim=2)
        return v.view(v.size(0), -1)

class ResNet18Normal(ResNet18Base):
    """
    ノーマル版 ResNet-18
    - すべて dilation=1
    - stride は [1, 2, 2, 2]
    """
    def __init__(self, in_ch: int = 3, out_dim: int = 2, width_mult: float = 1.0,
                 hidden_dim: int = 256, dropout_p: float = 0.3):
        super().__init__(in_ch, out_dim, width_mult, hidden_dim, dropout_p)
        chs = self.chs
        self.inplanes = chs[0]
        self.layer1 = self._make_layer(chs[0], 2, stride=1, dilation=1)
        self.layer2 = self._make_layer(chs[1], 2, stride=2, dilation=1)
        self.layer3 = self._make_layer(chs[2], 2, stride=2, dilation=1)
        self.layer4 = self._make_layer(chs[3], 2, stride=2, dilation=1)

class ResNet18Dilated(ResNet18Base):
    """
    dilation 版 ResNet-18
    - layer2〜4 で dilation を 2,4,8 と段階的に拡大
    """
    def __init__(self, in_ch: int = 3, out_dim: int = 2, width_mult: float = 1.0,
                 hidden_dim: int = 256, dropout_p: float = 0.3):
        super().__init__(in_ch, out_dim, width_mult, hidden_dim, dropout_p)
        chs = self.chs
        self.inplanes = chs[0]
        self.layer1 = self._make_layer(chs[0], 2, stride=1, dilation=1)
        self.layer2 = self._make_layer(chs[1], 2, stride=2, dilation=2)
        self.layer3 = self._make_layer(chs[2], 2, stride=2, dilation=4)
        self.layer4 = self._make_layer(chs[3], 2, stride=2, dilation=8)

# ============================================================
# Dataset (train_dilated_lab.py と完全一致)
# ============================================================
class ImageRegressionDataset(Dataset):
    def __init__(self, img_dir: Path | str, img_size: int,
                 input_mode: str,
                 color_mode: str = "lab"):
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.input_mode = input_mode.lower()
        self.color_mode = color_mode.lower()
        
        label_csv = self.img_dir.parent / "labels.csv"
        if not label_csv.exists():
            raise FileNotFoundError(f"Label CSV not found: {label_csv}")
        df = pd.read_csv(label_csv)
        self.paths = df["filename"].apply(lambda x: self.img_dir / x).tolist()
        self.targets_deg = df["roll"].astype(np.float32).values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_pil = Image.open(self.paths[idx]).convert("RGB")
        img_pil = img_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
        img_np = np.array(img_pil)

        if self.color_mode == "lab":
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)

            if self.input_mode in ("gray", "bin4"):
                L = img_lab[:, :, 0:1]
                L = (L / 255.0 - 0.5) / 0.5
                img_tensor = torch.from_numpy(L.transpose(2, 0, 1))  # (1,H,W)
            else:
                lab = (img_lab / 255.0 - 0.5) / 0.5
                img_tensor = torch.from_numpy(lab.transpose(2, 0, 1))  # (3,H,W)
        else:
            img_tensor = transforms.ToTensor()(img_pil)
            img_tensor = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(img_tensor)

        target_deg = float(self.targets_deg[idx])
        return img_tensor, target_deg, self.paths[idx].name

# ============================================================
# EKF (roll のみ, sin/cos 観測)
# ============================================================
KF_FPS = 15.0
KF_SIGMA_Z_DEG = 1.0
KF_SIGMA_A_DEG = 2.0
KF_INIT_STD_TH_DEG = 20.0
KF_INIT_STD_OM_DEG = 10.0

class EKFRoll:
    def __init__(self, theta0, omega0=0.0):
        self.x = np.array([float(theta0), float(omega0)])
        self.P = np.diag([
            math.radians(KF_INIT_STD_TH_DEG) ** 2,
            math.radians(KF_INIT_STD_OM_DEG) ** 2
        ])
        self.R = np.eye(2) * (math.radians(KF_SIGMA_Z_DEG) ** 2)
        self.sigma_a = math.radians(KF_SIGMA_A_DEG)

    def step(self, z, dt):
        F = np.array([[1, dt], [0, 1]])
        G = np.array([[0.5 * dt ** 2], [dt]])

        # 予測
        self.x = F @ self.x
        self.x[0] = _wrap_pi(self.x[0])
        self.P = F @ self.P @ F.T + (self.sigma_a ** 2) * (G @ G.T)

        th = self.x[0]
        h = np.array([math.sin(th), math.cos(th)])
        H = np.array([[math.cos(th), 0], [-math.sin(th), 0]])

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ (z - h)
        self.x[0] = _wrap_pi(self.x[0])
        self.P = (np.eye(2) - K @ H) @ self.P
        return self.x.copy()

# ============================================================
# Utility
# ============================================================
def parse_width_from_name(name: str, default: float = 1.0):
    if "_" not in name:
        return default
    try:
        token = name.split("_")[-1]
        return float(token.replace("p", "."))
    except ValueError:
        return default

def _extract_dataset_tag_and_split(valid_dir: Path) -> tuple[str, str]:
    """
    datasets/<DATASET>/cache/.../<split>/imgs というパスから
    DATASET名と split 名を推定。
    """
    parts = valid_dir.parts
    dataset_tag = "dataset"
    split_name = valid_dir.parent.name  # .../<split>/imgs → split

    if "datasets" in parts:
        idx = parts.index("datasets")
        if idx + 1 < len(parts):
            dataset_tag = parts[idx + 1]
    return dataset_tag, split_name

# ============================================================
# 1 データセット (1つの VALID_DIR) を評価
# ============================================================
def eval_one_dataset_for_exp(
    exp_dir: Path,
    cfg: dict,
    model: nn.Module,
    valid_dir: str,
    batch_size: int,
    num_workers: int,
):
    input_mode = cfg.get("INPUT_MODE", "rgb")
    color_mode = cfg.get("COLOR_MODE", "lab")
    img_size = cfg.get("IMG_SIZE", [224, 224])[0]

    test_path = Path(valid_dir)
    if not test_path.exists():
        print(f"  [SKIP] VALID_DIR not found: {test_path}")
        return

    ds = ImageRegressionDataset(
        test_path, img_size,
        input_mode=input_mode, color_mode=color_mode
    )
    ld = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = next(model.parameters()).device
    results: List[dict] = []
    sincos_list: List[np.ndarray] = []

    dataset_tag, split_name = _extract_dataset_tag_and_split(test_path)
    print(f"  Eval [{dataset_tag}] ({split_name}) with {len(ds)} samples")

    # ========= CNN 推論 =========
    with torch.no_grad():
        for imgs, tgt_deg, names in tqdm(ld, desc=f"{exp_dir.name} [{dataset_tag}:{split_name}]"):
            imgs = imgs.to(device)
            outs = model(imgs).cpu()  # (N,2)
            preds = sincos2deg(outs).numpy()  # (N,)
            tgts_np = np.array(tgt_deg, dtype=np.float32)
            sincos_list.extend(outs.numpy())

            for n, t, p in zip(names, tgts_np, preds):
                err = circular_error_np(np.array([p]), np.array([t]))[0]
                results.append({
                    "filename": n,
                    "true_roll": float(t),
                    "pred_roll": float(p),
                    "err_roll": float(err),  # CNN単体の誤差
                })

    if not results:
        print("  [SKIP] No results (empty dataset?)")
        return

    df = pd.DataFrame(results)

    # ========= EKF =========
    df_kf = df.sort_values("filename").reset_index(drop=True)
    theta0 = math.atan2(sincos_list[0][0], sincos_list[0][1])
    kf = EKFRoll(theta0)
    kf_res = []
    for i, row in df_kf.iterrows():
        idx = df[df["filename"] == row["filename"]].index[0]
        st = kf.step(np.array(sincos_list[idx]), 1.0 / KF_FPS)
        th = math.degrees(st[0]) % 360
        err_kf = circular_error_np(
            np.array([th]),
            np.array([row["true_roll"]])
        )[0]
        kf_res.append({
            "filename": row["filename"],
            "kf_roll": th,
            "kf_err": float(err_kf),
        })

    df = df.merge(pd.DataFrame(kf_res), on="filename")

    # ========= 保存先 =========
    out_dir = exp_dir / f"eval_{dataset_tag}_{split_name}_kf_latest"
    figs = out_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "eval_results_kf.csv"
    df.to_csv(csv_path, index=False)
    print("  Saved:", csv_path)

    # ========= 散布図 =========

    # 1) CNNのみの散布図
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["true_roll"], df["err_roll"], s=8, alpha=0.5)
    ax.set_title("Error vs True (CNN)")
    ax.set_xlabel("True Deg")
    ax.set_ylabel("Error Deg")
    fig.savefig(figs / "error_vs_true_cnn.png")
    plt.close(fig)

    # 2) CNN vs EKF の比較散布図
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["true_roll"], df["err_roll"],
               s=8, alpha=0.5, label="CNN")
    ax.scatter(df["true_roll"], df["kf_err"],
               s=8, alpha=0.5, label="EKF")
    ax.set_title("Error vs True (CNN vs EKF)")
    ax.set_xlabel("True Deg")
    ax.set_ylabel("Error Deg")
    ax.legend()
    fig.savefig(figs / "error_vs_true.png")
    plt.close(fig)

    # ========= ヒストグラム =========
    # エラーは circular_error なので理論上 0〜180deg
    # bin 幅 10deg → 0,10,20,...,180 が棒の左右の境界になる
    bins = np.arange(0, 181, 10)  # [0,10,20,...,180]

    # 3) CNNのみのヒストグラム
    fig, ax = plt.subplots(figsize=(7, 5))
    Y_cnn, bin_edges_cnn, _ = ax.hist(
        df["err_roll"],
        bins=bins,
        alpha=0.7,
        range=(0, 180),
        label="CNN",
        edgecolor="black",
        linewidth=1.0
    )

    ax.set_title("Error Histogram (CNN)")
    ax.set_xlabel("Error Deg [deg]")
    ax.set_ylabel("Count")
    ax.legend()

    # ★ 20度刻みの目盛りに変更
    ax.set_xticks(np.arange(0, 181, 20))

    # Y軸：整数 & 見やすい刻み幅
    y_max = int(max(Y_cnn)) + 1
    if y_max <= 10:
        step = 1
    elif y_max <= 40:
        step = 2
    else:
        step = 5
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, step))

    fig.savefig(figs / "error_hist_cnn.png")
    plt.close(fig)

    # 4) CNN vs EKF のヒストグラム
    fig, ax = plt.subplots(figsize=(7, 5))
    Y1, bin_edges, _ = ax.hist(
        df["err_roll"],
        bins=bins,
        alpha=0.5,
        range=(0, 180),
        label="CNN",
        edgecolor="black",
        linewidth=1.0
    )
    Y2, _, _ = ax.hist(
        df["kf_err"],
        bins=bins,
        alpha=0.5,
        range=(0, 180),
        label="EKF",
        edgecolor="black",
        linewidth=1.0
    )

    ax.set_title("Error Histogram (CNN vs EKF)")
    ax.set_xlabel("Error Deg [deg]")
    ax.set_ylabel("Count")
    ax.legend()

    # ★ 20度刻みの目盛りに変更
    ax.set_xticks(np.arange(0, 181, 20))

    # 2つの系列の最大値から共通のYスケール & 整数目盛
    y_max = int(max(max(Y1), max(Y2))) + 1
    if y_max <= 10:
        step = 1
    elif y_max <= 40:
        step = 2
    else:
        step = 5
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, step))

    fig.savefig(figs / "error_hist.png")
    plt.close(fig)

    # ========= 画像ペアの可視化 (誤差大 Top3 / 小 Top3) =========
    # valid 側: df の err_roll からランキング
    if len(df) >= 1:
        # Top3 (大きい誤差)
        df_large = df.sort_values("err_roll", ascending=False).head(3)
        # Top3 (小さい誤差)
        df_small = df.sort_values("err_roll", ascending=True).head(3)

        # ---- train データセットの場所は config の TRAIN_DIR を使う ----
        train_dir_str = cfg.get("TRAIN_DIR", None)
        if train_dir_str is None:
            print("  [WARN] TRAIN_DIR not found in config. Skip pair visualization.")
        else:
            train_imgs_dir = Path(train_dir_str)
            if not train_imgs_dir.exists():
                print(f"  [WARN] TRAIN_DIR does not exist: {train_imgs_dir}")
            else:
                # labels.csv の場所: パターンA想定 → train/imgs の 1つ上に labels.csv
                labels_candidates = [
                    train_imgs_dir.parent / "labels.csv",  # datasets/type-2_lab/cache/rgb/sz224_area/train/labels.csv
                    train_imgs_dir.parent.parent / "labels.csv",  # 念のため: sz224_area/labels.csv
                ]
                train_labels_csv = None
                for cand in labels_candidates:
                    if cand.exists():
                        train_labels_csv = cand
                        break

                if train_labels_csv is None:
                    print("  [WARN] train labels.csv not found. Tried:")
                    for cand in labels_candidates:
                        print(f"         - {cand}")
                else:
                    print(f"  Use train imgs:   {train_imgs_dir}")
                    print(f"  Use train labels: {train_labels_csv}")

                    df_train = pd.read_csv(train_labels_csv)
                    # 必要な列だけチェック
                    if "filename" not in df_train.columns or "roll" not in df_train.columns:
                        print("  [WARN] train labels.csv does not contain 'filename' or 'roll'. Skip pair visualization.")
                    else:
                        train_fnames = df_train["filename"].values
                        train_rolls = df_train["roll"].astype(np.float32).values

                        def _find_closest_train_index(target_deg: float) -> int:
                            """target_deg (pred_roll) に最も近い真値 roll を持つ train index を返す"""
                            target_arr = np.full_like(train_rolls, target_deg, dtype=np.float32)
                            diffs = circular_error_np(train_rolls, target_arr)
                            return int(np.argmin(diffs))

                        def _make_pair_figure(row, rank_idx: int, kind: str):
                            """
                            kind: 'large' or 'small'
                            rank_idx: 1,2,3
                            """
                            v_fname = row["filename"]
                            v_true = float(row["true_roll"])
                            v_pred = float(row["pred_roll"])
                            v_err = float(row["err_roll"])

                            v_img_path = test_path / v_fname
                            if not v_img_path.exists():
                                print(f"   [WARN] valid image not found: {v_img_path}")
                                return

                            # pred_roll を基準に、train の真値 roll が最も近いものを選ぶ
                            idx_best = _find_closest_train_index(v_pred)
                            t_fname = train_fnames[idx_best]
                            t_roll = float(train_rolls[idx_best])
                            t_img_path = train_imgs_dir / t_fname
                            if not t_img_path.exists():
                                print(f"   [WARN] train image not found: {t_img_path}")
                                return

                            # 画像読み込み（BGR→RGB）
                            v_img_bgr = cv2.imread(str(v_img_path))
                            t_img_bgr = cv2.imread(str(t_img_path))
                            if v_img_bgr is None or t_img_bgr is None:
                                print(f"   [WARN] failed to read images for pair ({v_img_path}, {t_img_path})")
                                return
                            v_img_rgb = cv2.cvtColor(v_img_bgr, cv2.COLOR_BGR2RGB)
                            t_img_rgb = cv2.cvtColor(t_img_bgr, cv2.COLOR_BGR2RGB)

                            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                            ax1, ax2 = axes

                            ax1.imshow(v_img_rgb)
                            ax1.axis("off")
                            ax1.set_title(
                                f"VALID\n{v_fname}\ntrue={v_true:.1f}°, pred={v_pred:.1f}°\nerr={v_err:.1f}°",
                                fontsize=8
                            )

                            ax2.imshow(t_img_rgb)
                            ax2.axis("off")
                            ax2.set_title(
                                f"TRAIN\n{t_fname}\nroll(true)={t_roll:.1f}°",
                                fontsize=8
                            )

                            fig.suptitle(
                                f"{kind.upper()} #{rank_idx}: err={v_err:.2f} deg (pred={v_pred:.1f}°)",
                                fontsize=10
                            )
                            fig.tight_layout()

                            out_name = f"pair_{kind}_{rank_idx}_{v_fname}.png"
                            fig.savefig(figs / out_name, dpi=150)
                            plt.close(fig)
                            print(f"  Saved pair image: {figs / out_name}")

                        # 誤差大 Top3
                        for i, (_, row) in enumerate(df_large.iterrows(), start=1):
                            _make_pair_figure(row, i, kind="large")

                        # 誤差小 Top3
                        for i, (_, row) in enumerate(df_small.iterrows(), start=1):
                            _make_pair_figure(row, i, kind="small")

    print(f"  Done [{dataset_tag}:{split_name}]")

# ============================================================
# exp 単位の処理
# ============================================================
def eval_one_exp(exp_dir: Path, args):
    # used_config.yaml 優先, なければ config_used.yaml
    cfg_path_used = exp_dir / "used_config.yaml"
    cfg_path_alt = exp_dir / "config_used.yaml"

    if cfg_path_used.exists():
        cfg_path = cfg_path_used
    elif cfg_path_alt.exists():
        cfg_path = cfg_path_alt
    else:
        print("[SKIP] No used_config.yaml or config_used.yaml in", exp_dir)
        return

    ckpt = exp_dir / "checkpoints" / "latest.pth"
    if not ckpt.exists():
        print("[SKIP] checkpoints/latest.pth not found in", exp_dir)
        return

    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    # VALID_DIR は必ず評価
    main_valid = cfg.get("VALID_DIR", None)
    if main_valid is None:
        print("[SKIP] VALID_DIR not found in", exp_dir)
        return

    # VALID_DIR リストの構築
    valid_list: List[str] = [main_valid]

    # --use_valid_dirs 指定時のみ VALID_DIRS も追加
    if args.use_valid_dirs:
        extra_dirs = cfg.get("VALID_DIRS", [])
        for v in extra_dirs:
            if v not in valid_list:
                valid_list.append(v)

    # モデル構成の復元（train_dilated_lab.py と同じロジック）
    input_mode = cfg.get("INPUT_MODE", "rgb")
    color_mode = cfg.get("COLOR_MODE", "lab")
    img_size = cfg.get("IMG_SIZE", [224, 224])[0]
    model_name = cfg.get("MODEL_NAME", "ResNet18Dilated_1p0")
    width = parse_width_from_name(model_name, 1.0)

    imode = input_mode.lower()
    in_ch = 1 if imode in ("gray", "bin4") else 3

    name_lower = model_name.lower()
    if "dilated" in name_lower or "dil" in name_lower:
        ModelClass = ResNet18Dilated
    else:
        ModelClass = ResNet18Normal

    batch_size = args.batch_size or cfg.get("BATCH_SIZE", 64)
    num_workers = args.num_workers or cfg.get("NUM_WORKERS", 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelClass(
        in_ch=in_ch,
        out_dim=2,
        width_mult=width,
        hidden_dim=cfg.get("HIDDEN_DIM", 256),
        dropout_p=cfg.get("DROPOUT_P", 0.3),
    ).to(device)

    print("  Load ckpt:", ckpt)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    print(f"  INPUT_MODE={input_mode}, COLOR_MODE={color_mode}, IMG_SIZE={img_size}")
    print(f"  MODEL={model_name}, width_mult={width}, in_ch={in_ch}")
    print(f"  VALID_DIR: {main_valid}")
    if args.use_valid_dirs:
        print(f"  VALID_DIRS enabled, total {len(valid_list)} dirs")

    for vdir in valid_list:
        print(f"  --- VALID_DIR: {vdir} ---")
        try:
            eval_one_dataset_for_exp(
                exp_dir,
                cfg,
                model,
                vdir,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        except Exception as e:
            print("   [ERROR in VALID_DIR]", vdir, ":", e)

# ============================================================
# CLI / MAIN
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="lab_logs/expXX を used_config.yaml に基づいて自動評価"
    )
    parser.add_argument("--batch_size", type=int, default=None,
                        help="未指定なら used_config.yaml の BATCH_SIZE を使用")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="未指定なら used_config.yaml の NUM_WORKERS を使用")
    parser.add_argument("--log_root", type=str, default="lab_logs",
                        help="expXX が入っているルート")
    parser.add_argument("--exp_from", type=str, default=None,
                        help="評価対象 exp の下限 (例: exp100)")
    parser.add_argument("--exp_to", type=str, default=None,
                        help="評価対象 exp の上限 (例: exp200)")
    parser.add_argument("--use_valid_dirs", action="store_true",
                        help="VALID_DIR に加え VALID_DIRS も評価する")
    return parser.parse_args()

def parse_exp_number(v: str | None):
    if v is None:
        return None
    m = re.search(r"(\d+)$", v.strip())
    return int(m.group(1)) if m else None

def main():
    args = parse_args()
    root = Path(args.log_root)
    if not root.exists():
        print("Not found:", root)
        return

    f = parse_exp_number(args.exp_from)
    t = parse_exp_number(args.exp_to)

    exp_dirs: List[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("exp"):
            m = re.match(r"exp(\d+)", p.name)
            if not m:
                continue
            n = int(m.group(1))
            if f is not None and n < f:
                continue
            if t is not None and n > t:
                continue
            exp_dirs.append(p)

    if not exp_dirs:
        print("No matching exp dirs")
        return

    print("Found", len(exp_dirs), "experiments")

    for p in exp_dirs:
        print("\n=== ", p.name, " ===")
        try:
            eval_one_exp(p, args)
        except Exception as e:
            print(" [ERROR in exp]", p.name, ":", e)

if __name__ == "__main__":
    main()
