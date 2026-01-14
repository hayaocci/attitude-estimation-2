#!/usr/bin/env python3
"""
train_dilated.py ― Research project trainer (Dilated ResNet version)
===================================================================
* Dilated Convolutionを採用し、パラメータ数を増やさずに受容野を拡大。
* Gridding Effect対策として、各Stageで異なるDilation Rateを適用。
* 幅縮小 (WIDTH_MULT) および sin・cos 2 出力学習に対応。
"""
from __future__ import annotations

import argparse, csv, json, math, random, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ============================================================
# Reproducibility helper
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# Angle helpers (roll only)
# ============================================================

def deg2rad(x: torch.Tensor | float):
    return x * math.pi / 180.0

def deg2sincos(x: torch.Tensor):
    x_rad = deg2rad(x)
    return torch.stack((torch.sin(x_rad), torch.cos(x_rad)), dim=-1)

def sincos2deg(v: torch.Tensor):
    rad = torch.atan2(v[..., 0], v[..., 1])
    deg = rad * 180.0 / math.pi
    return deg.remainder(360.0)

def circular_error(pred_deg: torch.Tensor, true_deg: torch.Tensor):
    diff = (pred_deg - true_deg + 180.0) % 360.0 - 180.0
    return diff.abs()

# ============================================================
# Dilated ResNet Implementation
# ============================================================

class DilatedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, dilation=1):
        super().__init__()
        # k=3 の場合、padding=dilation とすることで解像度を維持 (stride=1時)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, 
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        # 2層目も同様の dilation を適用
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, 
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18Dilated(nn.Module):
    """
    Gridding Effect対策を施した拡張ResNet18。
    各Stageで Dilation Rate を 1, 2, 4, 8 と変化させ、広域な特徴を密に抽出します。
    """
    def __init__(self, in_ch: int = 3, out_dim: int = 2, width_mult: float = 1.0, 
                 hidden_dim: int = 256, dropout_p: float = 0.3):
        super().__init__()
        base_channels = np.array([64, 128, 256, 512])
        chs = np.maximum(1, (base_channels * width_mult).astype(int)).tolist()
        self.inplanes = chs[0]

        # Stem (通常通りの 3x3)
        self.conv1 = nn.Conv2d(in_ch, chs[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(chs[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # 各Stageで dilation を変更 (Hybrid Dilated Convolution)
        # padding は _make_layer 内の DilatedBasicBlock で自動適用
        self.layer1 = self._make_layer(chs[0], 2, stride=1, dilation=1)
        self.layer2 = self._make_layer(chs[1], 2, stride=2, dilation=2)
        self.layer3 = self._make_layer(chs[2], 2, stride=2, dilation=4)
        self.layer4 = self._make_layer(chs[3], 2, stride=2, dilation=8)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chs[3], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, out_dim),
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"🏗️ ResNet18Dilated (width={width_mult}): {total/1e6:.2f}M params")

    def _make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        
        layers = []
        layers.append(DilatedBasicBlock(self.inplanes, planes, stride, downsample, dilation))
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
        # 角度用ベクトル L2 正規化
        v = v.view(v.size(0), -1, 2)
        v = nn.functional.normalize(v, dim=2)
        return v.view(v.size(0), -1)

# ============================================================
# Dataset
# ============================================================

class ImageRegressionDataset(Dataset):
    def __init__(self, img_dir: Path | str, img_size: int, normalize: bool = True):
        self.img_dir = Path(img_dir)
        label_csv = self.img_dir.parent / "labels.csv"
        if not label_csv.exists():
            raise FileNotFoundError(f"Label CSV not found: {label_csv}")
        df = pd.read_csv(label_csv)
        self.paths = df["filename"].apply(lambda x: self.img_dir / x).tolist()
        self.targets_deg = df["roll"].astype(np.float32).values

        mean, std = ([0.5], [0.5]) if normalize else ([0.0], [1.0])
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean * 3, std=std * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        target_deg = torch.tensor(self.targets_deg[idx])
        return img, target_deg, self.paths[idx].name

# ============================================================
# Factory & Config Helpers
# ============================================================

def parse_width_from_name(name: str, default: float = 1.0):
    if "_" not in name: return default
    try:
        token = name.split("_")[-1]
        return float(token.replace("p", "."))
    except ValueError: return default

def get_model(name: str, out_dim: int, hidden_dim: int = 256, dropout_p: float = 0.3):
    width = parse_width_from_name(name, 1.0)
    # 常に Dilated 版を返すように設定
    return ResNet18Dilated(3, out_dim, width_mult=width, hidden_dim=hidden_dim, dropout_p=dropout_p)

def load_configs(cfg_path: str | Path) -> List[Dict]:
    with open(cfg_path, encoding="utf-8") as f:
        cfgs = yaml.safe_load(f)
    processed = []
    for cfg in cfgs:
        train_root = Path(cfg.get("TRAIN_DATASET_ROOT", cfg["DATASET_ROOT"]))
        valid_root = Path(cfg.get("VALID_DATASET_ROOT", cfg["DATASET_ROOT"]))
        size, mode, resize_mode = int(cfg["IMG_SIZE"][0]), cfg["INPUT_MODE"], cfg.get("RESIZE_MODE", "area")
        
        def build(root: Path, split: str):
            return str(root / "cache" / mode / f"sz{size}_{resize_mode}" / split / "imgs")

        cfg["TRAIN_DIR"], cfg["VALID_DIR"] = build(train_root, "train"), build(valid_root, "valid")
        cfg["EVAL_DIR"], cfg["RESIZE_MODE"] = cfg["VALID_DIR"], resize_mode
        cfg["_TRAIN_ROOT"], cfg["_VALID_ROOT"] = str(train_root), str(valid_root)
        processed.append(cfg)
    return processed

def make_log_dir(cfg: Dict) -> Path:
    train_tag, valid_tag = Path(cfg["_TRAIN_ROOT"]).name, Path(cfg["_VALID_ROOT"]).name
    dataset_tag = f"{train_tag}_vs_{valid_tag}" if train_tag != valid_tag else train_tag
    mode_size = f"{cfg['INPUT_MODE']}_sz{cfg['IMG_SIZE'][0]}_{cfg['RESIZE_MODE']}"
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{cfg['id']}"
    run_dir = Path("logs") / dataset_tag / cfg["MODEL_NAME"] / mode_size / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    for sub in ["checkpoints", "figs"]: (run_dir / sub).mkdir(exist_ok=True)
    return run_dir

# ============================================================
# Training Logic
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, sincos_mode):
    model.train()
    total = 0.0
    for imgs, tgt_deg, _ in loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        if sincos_mode:
            tgt_vec = deg2sincos(tgt_deg).to(device)
            loss = criterion(model(imgs), tgt_vec)
        else:
            loss = criterion(model(imgs).squeeze(), tgt_deg.to(device))
        loss.backward(); optimizer.step()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)

def validate(model, loader, criterion, device, sincos_mode):
    model.eval()
    total, mae = 0.0, []
    with torch.no_grad():
        for imgs, tgt_deg, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            if sincos_mode:
                total += criterion(outputs, deg2sincos(tgt_deg).to(device)).item() * imgs.size(0)
                err = circular_error(sincos2deg(outputs.cpu()), tgt_deg)
            else:
                total += criterion(outputs.squeeze(), tgt_deg.to(device)).item() * imgs.size(0)
                err = (outputs.squeeze().cpu() - tgt_deg).abs()
            mae.extend(err.numpy())
    return total / len(loader.dataset), float(np.mean(mae))

def plot_training_curves(train_losses, val_losses, val_maes, out_dir: Path):
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, train_losses, label="train_loss")
    ax[0].plot(epochs, val_losses, label="val_loss")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(epochs, val_maes, label="val_MAE", color="orange")
    ax[1].set_title("MAE"); ax[1].legend()
    plt.savefig(out_dir / "figs" / "curves.png"); plt.close()

# ============================================================
# Main Experiment Driver
# ============================================================

def run_experiment(cfg):
    set_seed(cfg["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_log_dir(cfg)

    img_sz = cfg["IMG_SIZE"][0]
    train_ld = DataLoader(ImageRegressionDataset(cfg["TRAIN_DIR"], img_sz), cfg["BATCH_SIZE"], True, num_workers=cfg["NUM_WORKERS"])
    valid_ld = DataLoader(ImageRegressionDataset(cfg["VALID_DIR"], img_sz), cfg["BATCH_SIZE"], False, num_workers=cfg["NUM_WORKERS"])

    single_roll = cfg["OUTPUT_AXES"] == ["roll"]
    model = get_model(cfg["MODEL_NAME"], 2 if single_roll else len(cfg["OUTPUT_AXES"]), cfg["HIDDEN_DIM"], cfg["DROPOUT_P"]).to(device)

    criterion = nn.SmoothL1Loss() if single_roll else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["MAX_LR"], weight_decay=cfg["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["EPOCHS"])

    train_losses, val_losses, val_maes, best_mae, no_improve = [], [], [], float("inf"), 0
    
    for ep in range(1, cfg["EPOCHS"] + 1):
        tl = train_epoch(model, train_ld, optimizer, criterion, device, single_roll)
        vl, mae = validate(model, valid_ld, criterion, device, single_roll)
        scheduler.step()
        
        train_losses.append(tl); val_losses.append(vl); val_maes.append(mae)
        print(f"Epoch {ep:03d}: train={tl:.4f} valid={vl:.4f} MAE={mae:.3f}")
        
        if mae < best_mae:
            best_mae, no_improve = mae, 0
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best.pth")
        else:
            no_improve += 1
        
        if cfg.get("EARLY_STOP") and no_improve >= int(cfg.get("PATIENCE", 10)):
            print(f"Early stop at {ep}"); break

    plot_training_curves(train_losses, val_losses, val_maes, run_dir)
    print(f"Finished exp {cfg['id']}: Best MAE={best_mae:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    for cfg in load_configs(args.config):
        try: run_experiment(cfg)
        except Exception as e: print(f"[ERROR] Exp {cfg['id']} failed: {e}", file=sys.stderr)

        