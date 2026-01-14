#!/usr/bin/env python3
"""
train.py ― Research project trainer
===================================
* 読み込んだ `config.yaml` で複数実験を一括学習。
* ログは `logs/<dataset>/<model_tag>/<mode>_<size>/<run_stamp>/` に自動整理。
* roll 単一軸の場合は **sin・cos 2 出力** で学習し、角度 (0–360°) を推定。
* ResNet18 を **幅縮小 (WIDTH_MULT) 対応** した `ResNet18Scaled` 実装を内蔵。
* `error_vs_true.png` など各種図を `figs/` に保存。
* **RESIZE_MODE** をconfig.yamlで指定可能（datasets_converter.pyと連動）

必要ライブラリ: PyTorch >=2.0, torchvision, pandas, matplotlib, PyYAML, tqdm, numpy
"""
from __future__ import annotations

import argparse, csv, json, math, random, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib; matplotlib.use("Agg")  # head‑less
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
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
    """(B,) -> (B,2)   sin θ , cos θ"""
    x_rad = deg2rad(x)
    return torch.stack((torch.sin(x_rad), torch.cos(x_rad)), dim=-1)


def sincos2deg(v: torch.Tensor):
    """(B,2) -> (B,)  angle in deg [0,360)"""
    rad = torch.atan2(v[..., 0], v[..., 1])  # atan2(sin, cos)
    deg = rad * 180.0 / math.pi
    return deg.remainder(360.0)


def circular_error(pred_deg: torch.Tensor, true_deg: torch.Tensor):
    """Short‑way absolute error in deg, (‑180,180] → |·|"""
    diff = (pred_deg - true_deg + 180.0) % 360.0 - 180.0
    return diff.abs()

# ============================================================
# Lightweight ResNet18 with width scaling (from reference code)
# ============================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18Scaled(nn.Module):
    """ResNet18 where every channel is multiplied by `width_mult`."""

    def __init__(self, in_ch: int = 3, out_dim: int = 2, width_mult: float = 1.0, hidden_dim: int = 256, dropout_p: float = 0.3):
        super().__init__()
        base_channels = np.array([64, 128, 256, 512])
        chs = np.maximum(1, (base_channels * width_mult).astype(int)).tolist()
        self.inplanes = chs[0]

        # Stem (3×3 conv)
        self.conv1 = nn.Conv2d(in_ch, chs[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(chs[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # Stages
        self.layer1 = self._make_layer(chs[0], 2, 1)
        self.layer2 = self._make_layer(chs[1], 2, 2)
        self.layer3 = self._make_layer(chs[2], 2, 2)
        self.layer4 = self._make_layer(chs[3], 2, 2)

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
        print(f"📏 ResNet18Scaled width={width_mult}: {total/1e6:.2f}M params")

    # ------------------------------------------------------------------
    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        v = self.fc(x)  # (B, out_dim)
        # 角度用ベクトルは L2 正規化
        v = v.view(v.size(0), -1, 2)
        v = nn.functional.normalize(v, dim=2)
        return v.view(v.size(0), -1)

# ============================================================
# Dataset (PIL + torchvision transforms)
# ============================================================

class ImageRegressionDataset(Dataset):
    """Loads images and returns (img, target_deg, fname)"""

    def __init__(self, img_dir: Path | str, img_size: int, normalize: bool = True):
        self.img_dir = Path(img_dir)
        label_csv = self.img_dir.parent / "labels.csv"
        if not label_csv.exists():
            raise FileNotFoundError(f"Label CSV not found: {label_csv}")
        df = pd.read_csv(label_csv)
        self.paths = df["filename"].apply(lambda x: self.img_dir / x).tolist()
        self.targets_deg = df["roll"].astype(np.float32).values  # degree

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
# Model factory
# ============================================================

def parse_width_from_name(name: str, default: float = 1.0):
    """Extract width multiplier from strings like 'ResNet18_0p75' (→0.75)."""
    if "_" not in name:
        return default
    try:
        token = name.split("_")[-1]
        return float(token.replace("p", "."))
    except ValueError:
        return default


def get_model(name: str, out_dim: int, hidden_dim: int = 256, dropout_p: float = 0.3):
    if name.startswith("ResNet18Scaled") or name.startswith("ResNet18"):
        width = parse_width_from_name(name, 1.0)
        return ResNet18Scaled(3, out_dim, width_mult=width, hidden_dim=hidden_dim, dropout_p=dropout_p)
    raise ValueError(f"Unknown MODEL_NAME: {name}")

# ============================================================
# Config loader & log dir helper (MODIFIED to support RESIZE_MODE)
# ============================================================

# ============================================================
# Config loader  (★ UPDATED ★)
# ============================================================
def load_configs(cfg_path: str | Path) -> List[Dict]:
    with open(cfg_path, encoding="utf-8") as f:
        cfgs = yaml.safe_load(f)

    processed = []
    for cfg in cfgs:
        # --- 追加: 個別ルートを取得（無ければ DATASET_ROOT を流用） ---
        train_root = Path(cfg.get("TRAIN_DATASET_ROOT", cfg["DATASET_ROOT"]))
        valid_root = Path(cfg.get("VALID_DATASET_ROOT", cfg["DATASET_ROOT"]))

        size        = int(cfg["IMG_SIZE"][0])
        mode        = cfg["INPUT_MODE"]
        resize_mode = cfg.get("RESIZE_MODE", "area")

        def build(root: Path, split: str):
            return str(root / "cache" / mode / f"sz{size}_{resize_mode}" / split / "imgs")

        cfg["TRAIN_DIR"] = build(train_root, "train")
        cfg["VALID_DIR"] = build(valid_root, "valid")
        cfg["EVAL_DIR"]  = cfg["VALID_DIR"]

        # ログ作成用に保持
        cfg["_TRAIN_ROOT"] = str(train_root)
        cfg["_VALID_ROOT"] = str(valid_root)
        cfg["RESIZE_MODE"] = resize_mode
        processed.append(cfg)

    return processed


# ============================================================
# Log dir helper  (★ UPDATED ★)
# ============================================================
def make_log_dir(cfg: Dict) -> Path:
    # ルートごとのタグを生成
    train_tag = Path(cfg["_TRAIN_ROOT"]).name
    valid_tag = Path(cfg["_VALID_ROOT"]).name
    dataset_tag = f"{train_tag}_vs_{valid_tag}" if train_tag != valid_tag else train_tag

    size         = cfg["IMG_SIZE"][0]
    resize_mode  = cfg.get("RESIZE_MODE", "area")
    mode_size    = f"{cfg['INPUT_MODE']}_sz{size}_{resize_mode}"
    model_tag    = cfg["MODEL_NAME"]
    stamp        = datetime.now().strftime("%Y%m%d_%H%M")
    run_id       = f"{stamp}_{cfg['id']}"

    run_dir = Path("logs") / dataset_tag / model_tag / mode_size / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 以降（yaml 保存・サブフォルダ作成・latest シンボリックリンク）は元のまま
    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    for sub in ["checkpoints", "onnx", "figs", "samples"]:
        (run_dir / sub).mkdir(exist_ok=True)

    latest = Path("logs") / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.resolve())
    except OSError:
        pass
    return run_dir

# ============================================================
# Train / Validate helpers
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, sincos_mode):
    model.train()
    total = 0.0
    for imgs, tgt_deg, _ in loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        if sincos_mode:
            tgt_vec = deg2sincos(tgt_deg).to(device)
            outputs = model(imgs)
            loss = criterion(outputs, tgt_vec)
        else:
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, tgt_deg.to(device))
        loss.backward()
        optimizer.step()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)

def validate(model, loader, criterion, device, sincos_mode):
    model.eval()
    total = 0.0
    mae = []
    with torch.no_grad():
        for imgs, tgt_deg, _ in loader:
            imgs = imgs.to(device)
            if sincos_mode:
                tgt_vec = deg2sincos(tgt_deg).to(device)
                outputs = model(imgs)
                loss = criterion(outputs, tgt_vec)
                pred_deg = sincos2deg(outputs.cpu())
                err = circular_error(pred_deg, tgt_deg)
            else:
                outputs = model(imgs).squeeze()
                loss = criterion(outputs, tgt_deg.to(device))
                err = (outputs.cpu() - tgt_deg).abs()
            total += loss.item() * imgs.size(0)
            mae.extend(err.numpy())
    return total / len(loader.dataset), float(np.mean(mae))

# ============================================================
# Plot helpers
# ============================================================

def plot_training_curves(train_losses, val_losses, val_maes, out_dir: Path):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "figs" / "loss_curve.png"); plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_maes, label="val_MAE")
    plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "figs" / "metric_curve.png"); plt.close()


def plot_err_vs_true(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(6, 4))
    plt.scatter(df["true_roll"], df["err_roll"], s=8, alpha=0.6)
    plt.xlabel("True roll (deg)")
    plt.ylabel("|Error| (deg)")
    plt.tight_layout()
    plt.savefig(out_dir / "figs" / "error_vs_true.png"); plt.close()

# ============================================================
# Main experiment driver
# ============================================================

def run_experiment(cfg):
    set_seed(cfg["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_log_dir(cfg)

    # Dataset & loaders
    img_sz = cfg["IMG_SIZE"][0]
    train_ds = ImageRegressionDataset(cfg["TRAIN_DIR"], img_sz)
    valid_ds = ImageRegressionDataset(cfg["VALID_DIR"], img_sz)

    g = torch.Generator().manual_seed(cfg["SEED"])
    train_ld = DataLoader(train_ds, cfg["BATCH_SIZE"], True, num_workers=cfg["NUM_WORKERS"], generator=g)
    valid_ld = DataLoader(valid_ds, cfg["BATCH_SIZE"], False, num_workers=cfg["NUM_WORKERS"], generator=g)

    # Model
    single_roll = cfg["OUTPUT_AXES"] == ["roll"]
    out_dim = 2 if single_roll else len(cfg["OUTPUT_AXES"])
    model = get_model(cfg["MODEL_NAME"], out_dim, cfg["HIDDEN_DIM"], cfg["DROPOUT_P"]).to(device)

    criterion = nn.SmoothL1Loss() if single_roll else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["MAX_LR"], weight_decay=cfg["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["EPOCHS"])

    train_losses, val_losses, val_maes = [], [], []
    best_mae = float("inf")
    no_improve = 0  # early stopping用カウンター

    # 学習ループ
    with (run_dir / "train_log.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_mae"])
        
        for ep in range(1, cfg["EPOCHS"] + 1):
            # 学習とバリデーションを実行
            tl = train_epoch(model, train_ld, optimizer, criterion, device, single_roll)
            vl, mae = validate(model, valid_ld, criterion, device, single_roll)
            scheduler.step()
            
            # 結果を記録
            train_losses.append(tl)
            val_losses.append(vl)
            val_maes.append(mae)
            w.writerow([ep, tl, vl, mae])
            f.flush()
            print(f"Epoch {ep:03d}: train={tl:.4f} valid={vl:.4f} MAE={mae:.3f}")
            
            # モデルを保存
            torch.save(model.state_dict(), run_dir / "checkpoints" / "last.pth")
            
            # Early stopping判定（maeが計算された後に実行）
            delta = float(cfg.get("DELTA", 0.0))  # 文字列の場合でも数値に変換
            if mae + delta < best_mae:
                best_mae = mae
                no_improve = 0
                torch.save(model.state_dict(), run_dir / "checkpoints" / "best.pth")
                print(f"New best MAE: {best_mae:.3f}")
            else:
                no_improve += 1
                print(f"No improvement for {no_improve} epochs")

            # Early stop判定
            early_stop = cfg.get("EARLY_STOP", False)
            patience = int(cfg.get("PATIENCE", 10))  # 文字列の場合でも数値に変換
            if early_stop and no_improve >= patience:
                print(f"Early stopping at epoch {ep} (no improvement for {patience} epochs)")
                break

    # 以下、評価部分は変更なし
    # Curves
    plot_training_curves(train_losses, val_losses, val_maes, run_dir)

    
    # Evaluation (use VALID split)
    eval_ds = ImageRegressionDataset(cfg["EVAL_DIR"], img_sz)
    eval_ld = DataLoader(eval_ds, cfg["BATCH_SIZE"], False, num_workers=cfg["NUM_WORKERS"])

    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pth", map_location=device))
    model.eval()

    records, abs_errs = [], []
    with torch.no_grad():
        for imgs, tgt_deg, names in tqdm(eval_ld, desc="Evaluating"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            if single_roll:
                pred_deg = sincos2deg(outputs.cpu())
                err = circular_error(pred_deg, tgt_deg)
            else:
                pred_deg = outputs.squeeze().cpu()
                err = (pred_deg - tgt_deg).abs()
            abs_errs.extend(err.numpy())
            for n, t, p, e in zip(names, tgt_deg.numpy(), pred_deg.numpy(), err.numpy()):
                records.append({"filename": n, "true_roll": float(t), "pred_roll": float(p), "err_roll": float(e)})

    df = pd.DataFrame(records)
    df.to_csv(run_dir / "eval_results.csv", index=False)
    plot_err_vs_true(df, run_dir)

    # Summary metrics
    metrics = {"best_val_MAE": best_mae, "eval_MAE": float(np.mean(abs_errs)), "eval_count": len(abs_errs)}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Finished exp {cfg['id']}: best_val_MAE={best_mae:.3f} eval_MAE={metrics['eval_MAE']:.3f}")

# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v2.yaml")
    args = parser.parse_args()

    cfgs = load_configs(args.config)
    for cfg in cfgs:
        try:
            run_experiment(cfg)
        except Exception as e:
            print(f"[ERROR] Experiment {cfg['id']} failed: {e}", file=sys.stderr)
            