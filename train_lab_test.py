#!/usr/bin/env python3
"""
train_dilated_lab.py ― Space Debris Attitude Estimation Trainer
===================================================================
Features:
- Dilated ResNet-18 for expanded receptive field.
- CIELAB (Lab) color space support for illumination robustness.
- Save 'latest.pth' and 'best.pth'.
- Pretrained model loading toggle.
- Simplified log directory: lab_logs/{id}
- Real-time progress tracking with tqdm.
"""
from __future__ import annotations

import argparse, math, random, sys
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
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ============================================================
# Reproducibility & Math Helpers
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    """角度の環状性(0=360)を考慮した絶対誤差を計算"""
    diff = (pred_deg - true_deg + 180.0) % 360.0 - 180.0
    return diff.abs()

# ============================================================
# Model Architecture (Dilated ResNet)
# ============================================================



class DilatedBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, 
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
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
    def __init__(self, in_ch: int = 3, out_dim: int = 2, width_mult: float = 1.0, 
                 hidden_dim: int = 256, dropout_p: float = 0.3):
        super().__init__()
        base_channels = np.array([64, 128, 256, 512])
        chs = np.maximum(1, (base_channels * width_mult).astype(int)).tolist()
        self.inplanes = chs[0]
        
        self.conv1 = nn.Conv2d(in_ch, chs[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(chs[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # 各レイヤーで dilation を段階的に拡大
        self.layer1 = self._make_layer(chs[0], 2, stride=1, dilation=1)
        self.layer2 = self._make_layer(chs[1], 2, stride=2, dilation=2)
        self.layer3 = self._make_layer(chs[2], 2, stride=2, dilation=4)
        self.layer4 = self._make_layer(chs[3], 2, stride=2, dilation=8)

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
        # 出力を単位ベクトルに正規化 (L2 Normalize)
        v = v.view(v.size(0), -1, 2)
        v = nn.functional.normalize(v, dim=2)
        return v.view(v.size(0), -1)

# ============================================================
# Dataset Logic
# ============================================================

class ImageRegressionDataset(Dataset):
    def __init__(self, img_dir: Path | str, img_size: int, color_mode: str = "lab"):
        self.img_dir = Path(img_dir)
        self.img_size = img_size
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
        # 1. 画像読み込み & リサイズ
        img_pil = Image.open(self.paths[idx]).convert("RGB")
        img_pil = img_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
        img_np = np.array(img_pil)

        if self.color_mode == "lab":
            # 2. RGB -> Lab 変換
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            # 3. 正規化 (uint8 0-255 を [-1, 1] 程度にマッピング)
            img_lab = (img_lab / 255.0 - 0.5) / 0.5
            img_tensor = torch.from_numpy(img_lab.transpose(2, 0, 1))
        else:
            # RGBモード
            img_tensor = transforms.ToTensor()(img_pil)
            img_tensor = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(img_tensor)

        target_deg = torch.tensor(self.targets_deg[idx])
        return img_tensor, target_deg, self.paths[idx].name

# ============================================================
# Utility Functions
# ============================================================

def parse_width_from_name(name: str, default: float = 1.0):
    if "_" not in name: return default
    try:
        token = name.split("_")[-1]
        return float(token.replace("p", "."))
    except ValueError: return default

def load_configs(cfg_path: str | Path) -> List[Dict]:
    with open(cfg_path, encoding="utf-8") as f:
        cfgs = yaml.safe_load(f)
    processed = []
    for cfg in cfgs:
        train_root = Path(cfg.get("TRAIN_DATASET_ROOT", cfg["DATASET_ROOT"]))
        valid_root = Path(cfg.get("VALID_DATASET_ROOT", cfg["DATASET_ROOT"]))
        size, mode = int(cfg["IMG_SIZE"][0]), cfg["INPUT_MODE"]
        resize_mode = cfg.get("RESIZE_MODE", "area")
        
        def build(root: Path, split: str):
            return str(root / "cache" / mode / f"sz{size}_{resize_mode}" / split / "imgs")

        cfg["TRAIN_DIR"], cfg["VALID_DIR"] = build(train_root, "train"), build(valid_root, "valid")
        cfg["_TRAIN_ROOT"], cfg["_VALID_ROOT"] = str(train_root), str(valid_root)
        processed.append(cfg)
    return processed

def make_log_dir(cfg: Dict) -> Path:
    # ③ 保存先を lab_logs/{id} に設定
    run_dir = Path("lab_logs") / cfg["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    
    for sub in ["checkpoints", "figs"]: (run_dir / sub).mkdir(exist_ok=True)
    return run_dir

def plot_training_curves(train_losses, val_losses, val_maes, out_dir: Path):
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, train_losses, label="train_loss")
    ax[0].plot(epochs, val_losses, label="val_loss")
    ax[0].set_title("Loss (SmoothL1)"); ax[0].legend()
    ax[1].plot(epochs, val_maes, label="val_MAE", color="orange")
    ax[1].set_title("MAE (Degrees)"); ax[1].legend()
    plt.savefig(out_dir / "figs" / "curves.png"); plt.close()

# ============================================================
# Training / Validation Steps
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, ep):
    model.train()
    total = 0.0
    # ④ tqdmでの進捗表示
    pbar = tqdm(loader, desc=f"Epoch {ep:03d} [Train]", leave=False)
    for imgs, tgt_deg, _ in pbar:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        tgt_vec = deg2sincos(tgt_deg).to(device)
        
        loss = criterion(model(imgs), tgt_vec)
        loss.backward()
        optimizer.step()
        
        total += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return total / len(loader.dataset)

def validate(model, loader, criterion, device, ep):
    model.eval()
    total, mae_list = 0.0, []
    # ④ tqdmでの進捗表示
    pbar = tqdm(loader, desc=f"Epoch {ep:03d} [Valid]", leave=False)
    with torch.no_grad():
        for imgs, tgt_deg, _ in pbar:
            imgs = imgs.to(device)
            outputs = model(imgs)
            
            loss = criterion(outputs, deg2sincos(tgt_deg).to(device))
            total += loss.item() * imgs.size(0)
            
            # 角度誤差の計算
            err = circular_error(sincos2deg(outputs.cpu()), tgt_deg)
            mae_list.extend(err.numpy())
            pbar.set_postfix(mae=f"{np.mean(mae_list) if mae_list else 0:.3f}")
            
    return total / len(loader.dataset), float(np.mean(mae_list))

# ============================================================
# Experiment Driver
# ============================================================

def run_experiment(cfg):
    set_seed(cfg["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_log_dir(cfg)

    img_sz = cfg["IMG_SIZE"][0]
    color_mode = cfg.get("COLOR_MODE", "lab")

    # データローダーの準備
    train_ld = DataLoader(ImageRegressionDataset(cfg["TRAIN_DIR"], img_sz, color_mode=color_mode), 
                          cfg["BATCH_SIZE"], True, num_workers=cfg.get("NUM_WORKERS", 4))
    valid_ld = DataLoader(ImageRegressionDataset(cfg["VALID_DIR"], img_sz, color_mode=color_mode), 
                          cfg["BATCH_SIZE"], False, num_workers=cfg.get("NUM_WORKERS", 4))

    # モデルのビルド
    width = parse_width_from_name(cfg["MODEL_NAME"], 1.0)
    model = ResNet18Dilated(in_ch=3, out_dim=2, width_mult=width, 
                            hidden_dim=cfg["HIDDEN_DIM"], dropout_p=cfg["DROPOUT_P"]).to(device)

    # ② 学習済みモデルのロード機能
    if cfg.get("USE_PRETRAINED", False):
        pretrained_path = cfg.get("PRETRAINED_PATH", "")
        if pretrained_path and Path(pretrained_path).exists():
            print(f"📖 Loading weights from: {pretrained_path}")
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
        else:
            print(f"⚠️ Pretrained model not found: {pretrained_path}")

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["MAX_LR"], weight_decay=cfg.get("WEIGHT_DECAY", 0.01))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["EPOCHS"])

    train_losses, val_losses, val_maes, best_mae, no_improve = [], [], [], float("inf"), 0
    
    print(f"🚀 Experiment {cfg['id']}: ColorMode={color_mode}, LR={cfg['MAX_LR']}")
    
    for ep in range(1, cfg["EPOCHS"] + 1):
        tl = train_epoch(model, train_ld, optimizer, criterion, device, ep)
        vl, mae = validate(model, valid_ld, criterion, device, ep)
        scheduler.step()
        
        train_losses.append(tl); val_losses.append(vl); val_maes.append(mae)
        print(f"Epoch {ep:03d}: train_loss={tl:.4f} valid_loss={vl:.4f} MAE={mae:.3f}")
        
        # ① latest.pth を毎エポック保存
        torch.save(model.state_dict(), run_dir / "checkpoints" / "latest.pth")
        
        # ベストモデルの保存
        if mae < best_mae:
            best_mae, no_improve = mae, 0
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best.pth")
        else:
            no_improve += 1
        
        # 早期終了
        if cfg.get("EARLY_STOP") and no_improve >= int(cfg.get("PATIENCE", 10)):
            print(f"Early stop at {ep}"); break

    # 学習曲線の保存
    plot_training_curves(train_losses, val_losses, val_maes, run_dir)
    print(f"✨ Finished {cfg['id']}: Best MAE={best_mae:.3f}")

# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_lab.yaml")
    args = parser.parse_args()
    
    configs = load_configs(args.config)
    for cfg in configs:
        try:
            run_experiment(cfg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] Exp {cfg['id']} failed: {e}", file=sys.stderr)