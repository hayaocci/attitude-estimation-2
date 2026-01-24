#!/usr/bin/env python3
"""
train_dilated_lab.py ― Space Debris Attitude Estimation Trainer
===================================================================
Features:
- Normal ResNet-18 *and* Dilated ResNet-18 (switchable by MODEL_NAME).
- Up to 4 validation datasets per experiment.
- CIELAB (Lab) color space support for illumination robustness.
- Save 'latest.pth' and 'best.pth' (best.pth is decided ONLY by Val#1).
- Pretrained model loading toggle.
- Simplified log directory: lab_logs/{id}
- Real-time progress tracking with tqdm.
- [NEW] Linear LR warmup + CosineAnnealingLR scheduler
  (WARMUP_EPOCHS in config, default=5).
"""
from __future__ import annotations

import argparse, math, random, sys
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
# Model Architecture (ResNet: Normal & Dilated)
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
    共通部分:
    - conv1 / bn1 / relu / maxpool
    - _make_layer
    - avgpool + MLP ヘッド (+ L2 正規化)
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

        # 派生クラス側で layer1〜4 を定義する想定
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
        # 出力を単位ベクトルに正規化 (L2 Normalize)
        v = v.view(v.size(0), -1, 2)
        v = nn.functional.normalize(v, dim=2)
        return v.view(v.size(0), -1)


class ResNet18Normal(ResNet18Base):
    """
    ★ ノーマル版 ResNet-18
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
# Dataset Logic
# ============================================================

class ImageRegressionDataset(Dataset):
    def __init__(self, img_dir: Path | str, img_size: int,
                 input_mode: str,              # gray / bin4 / rgb など
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
        # 1. 画像読み込み & リサイズ
        img_pil = Image.open(self.paths[idx]).convert("RGB")
        img_pil = img_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
        img_np = np.array(img_pil)

        # Labベースで INPUT_MODE ごとに1ch/3chを切替
        if self.color_mode == "lab":
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)

            if self.input_mode in ("gray", "bin4"):
                # Lのみ (H, W, 1) → 1ch
                L = img_lab[:, :, 0:1]
                L = (L / 255.0 - 0.5) / 0.5
                img_tensor = torch.from_numpy(L.transpose(2, 0, 1))  # (1, H, W)
            else:
                # rgbなど → Labの L,a,b をそのまま3ch使用
                lab = (img_lab / 255.0 - 0.5) / 0.5
                img_tensor = torch.from_numpy(lab.transpose(2, 0, 1))  # (3, H, W)
        else:
            # 非Labモード（ほぼ使わない想定だが互換のため残す）
            img_tensor = transforms.ToTensor()(img_pil)
            img_tensor = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(img_tensor)

        target_deg = torch.tensor(self.targets_deg[idx])
        return img_tensor, target_deg, self.paths[idx].name

# ============================================================
# Utility Functions
# ============================================================

def parse_width_from_name(name: str, default: float = 1.0):
    if "_" not in name:
        return default
    try:
        token = name.split("_")[-1]
        return float(token.replace("p", "."))
    except ValueError:
        return default

def load_configs(cfg_path: str | Path) -> List[Dict]:
    """
    config_lab.yaml の各エントリを展開する。
    - TRAIN_DATASET_ROOT / VALID_DATASET_ROOT / DATASET_ROOT から
      TRAIN_DIR, VALID_DIR を生成（従来通り）
    - さらに EXTRA_VALID_DATASET_ROOTS があれば、そこから追加の VALID_DIR を生成し、
      VALID_DIRS に最大4つまで格納する。
    """
    with open(cfg_path, encoding="utf-8") as f:
        cfgs = yaml.safe_load(f)
    processed = []
    for cfg in cfgs:
        train_root = Path(cfg.get("TRAIN_DATASET_ROOT", cfg["DATASET_ROOT"]))
        valid_root_main = Path(cfg.get("VALID_DATASET_ROOT", cfg["DATASET_ROOT"]))
        size, mode = int(cfg["IMG_SIZE"][0]), cfg["INPUT_MODE"]
        resize_mode = cfg.get("RESIZE_MODE", "area")
        
        def build(root: Path, split: str):
            return str(root / "cache" / mode / f"sz{size}_{resize_mode}" / split / "imgs")

        train_dir = build(train_root, "train")
        valid_dir_main = build(valid_root_main, "valid")

        # 追加の validation root（最大3つ分）
        extra_valid_roots = cfg.get("EXTRA_VALID_DATASET_ROOTS", [])
        valid_dirs: List[str] = [valid_dir_main]
        valid_roots_str: List[str] = [str(valid_root_main)]

        for root_str in extra_valid_roots[:3]:
            root_path = Path(root_str)
            valid_dirs.append(build(root_path, "valid"))
            valid_roots_str.append(str(root_path))

        cfg["TRAIN_DIR"] = train_dir
        cfg["VALID_DIR"] = valid_dir_main          # 従来互換
        cfg["VALID_DIRS"] = valid_dirs             # 最大4つの valid dir
        cfg["_TRAIN_ROOT"] = str(train_root)
        cfg["_VALID_ROOTS"] = valid_roots_str

        processed.append(cfg)
    return processed

def make_log_dir(cfg: Dict) -> Path:
    # 保存先を lab_logs/{id} に設定
    run_dir = Path("lab_logs") / cfg["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    
    for sub in ["checkpoints", "figs"]:
        (run_dir / sub).mkdir(exist_ok=True)
    return run_dir

def plot_training_curves_basic(train_losses, val_losses_main, val_maes_main, out_dir: Path):
    """
    従来通りの「train_loss と Val#1 の val_loss / MAE」のみの学習曲線。
    """
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax[0].plot(epochs, train_losses, label="train_loss")
    ax[0].plot(epochs, val_losses_main, label="val_loss_1 (primary)")
    ax[0].set_title("Loss (SmoothL1)")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # MAE (Val#1)
    ax[1].plot(epochs, val_maes_main, label="val_MAE_1 (deg)", color="orange")
    ax[1].set_title("MAE (Degrees, Val#1)")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("MAE [deg]")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "figs" / "curves.png")
    plt.close()

def plot_training_curves_multi(train_losses, val_losses_all: List[List[float]], out_dir: Path):
    """
    train_loss とすべての validation セットの val_loss を1枚の図に重ねて描画。
    - val_losses_all[i] が Val#(i+1) の val_loss 推移。
    """
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(epochs, train_losses, label="train_loss")

    for i, vloss_list in enumerate(val_losses_all):
        if not vloss_list:
            continue
        # 念のため長さの食い違いに対応（early stop 時など）
        e_len = min(len(epochs), len(vloss_list))
        ax.plot(epochs[:e_len], vloss_list[:e_len], label=f"val_loss_{i+1}")

    ax.set_title("Train vs All Validation Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "figs" / "curves_multi_val.png")
    plt.close()

# ============================================================
# Training / Validation Steps
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, ep):
    model.train()
    total = 0.0
    # tqdmでの進捗表示
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
    # tqdmでの進捗表示
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
    input_mode = cfg.get("INPUT_MODE", "rgb")   # INPUT_MODE参照

    # ---------------------------
    # データローダーの準備
    # ---------------------------
    train_ld = DataLoader(
        ImageRegressionDataset(
            cfg["TRAIN_DIR"], img_sz,
            input_mode=input_mode, color_mode=color_mode
        ),
        cfg["BATCH_SIZE"], True, num_workers=cfg.get("NUM_WORKERS", 4)
    )

    # 複数 validation データセット (最大4つ)
    valid_dirs: List[str] = cfg.get("VALID_DIRS", [cfg["VALID_DIR"]])
    if len(valid_dirs) > 4:
        valid_dirs = valid_dirs[:4]

    valid_ld_list: List[DataLoader] = []
    for vdir in valid_dirs:
        vld = DataLoader(
            ImageRegressionDataset(
                vdir, img_sz,
                input_mode=input_mode, color_mode=color_mode
            ),
            cfg["BATCH_SIZE"], False, num_workers=cfg.get("NUM_WORKERS", 4)
        )
        valid_ld_list.append(vld)

    # ---------------------------
    # モデルのビルド
    # ---------------------------
    model_name = cfg.get("MODEL_NAME", "resnet18_dilated")
    width = parse_width_from_name(model_name, 1.0)

    imode = input_mode.lower()
    in_ch = 1 if imode in ("gray", "bin4") else 3

    name_lower = model_name.lower()
    if "dilated" in name_lower or "dil" in name_lower:
        ModelClass = ResNet18Dilated
    else:
        ModelClass = ResNet18Normal

    model = ModelClass(
        in_ch=in_ch,
        out_dim=2,
        width_mult=width,
        hidden_dim=cfg["HIDDEN_DIM"],
        dropout_p=cfg["DROPOUT_P"]
    ).to(device)

    # 学習済みモデルのロード機能
    if cfg.get("USE_PRETRAINED", False):
        pretrained_path = cfg.get("PRETRAINED_PATH", "")
        if pretrained_path and Path(pretrained_path).exists():
            print(f"📖 Loading weights from: {pretrained_path}")
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
        else:
            print(f"⚠️ Pretrained model not found: {pretrained_path}")

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["MAX_LR"],
        weight_decay=cfg.get("WEIGHT_DECAY", 0.01),
    )

    # ---------------------------
    # LR Warmup + Cosine Scheduler
    # ---------------------------
    warmup_epochs = int(cfg.get("WARMUP_EPOCHS", 5))
    total_epochs = int(cfg["EPOCHS"])
    # Cosine を回すエポック数（warmup 後）
    cosine_epochs = max(1, total_epochs - max(0, warmup_epochs))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs
    )

    # ---------------------------
    # 学習ループ
    # ---------------------------
    train_losses: List[float] = []
    val_losses_main: List[float] = []    # Val#1 用
    val_maes_main: List[float] = []      # Val#1 用
    num_valid = len(valid_ld_list)
    val_losses_all: List[List[float]] = [[] for _ in range(num_valid)]

    best_mae, no_improve = float("inf"), 0

    # 学習ログ (CSV用)
    train_log: List[Dict] = []

    print(
        f"🚀 Experiment {cfg['id']}: "
        f"Model={model_name}, InputMode={input_mode}, ColorMode={color_mode}, "
        f"LR_MAX={cfg['MAX_LR']}, WARMUP_EPOCHS={warmup_epochs}, "
        f"Scheduler=Warmup+CosineAnnealing(T_max={cosine_epochs})"
    )
    
    for ep in range(1, total_epochs + 1):
        # --------------------
        # LR の決定（Warmup or Cosine）
        # --------------------
        if warmup_epochs > 0 and ep <= warmup_epochs:
            # 線形 Warmup: 0 → MAX_LR
            new_lr = cfg["MAX_LR"] * ep / float(warmup_epochs)
            for g in optimizer.param_groups:
                g["lr"] = new_lr

        # このエポックで実際に使用する LR を記録
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Train ---
        tl = train_epoch(model, train_ld, optimizer, criterion, device, ep)

        # --- Multi-Validation ---
        epoch_val_losses: List[float] = []
        epoch_val_maes: List[float] = []
        for vidx, vld in enumerate(valid_ld_list):
            vl_i, mae_i = validate(model, vld, criterion, device, ep)
            epoch_val_losses.append(vl_i)
            epoch_val_maes.append(mae_i)
            val_losses_all[vidx].append(vl_i)

        # primary (Val#1)
        vl_primary = epoch_val_losses[0]
        mae_primary = epoch_val_maes[0]
        val_losses_main.append(vl_primary)
        val_maes_main.append(mae_primary)

        # Cosine Scheduler は Warmup 終了後にのみ進める
        if ep > warmup_epochs:
            scheduler.step()
        
        train_losses.append(tl)

        # ログ出力（コンソール）
        # Val#1 については loss & MAE を表示
        msg = (
            f"Epoch {ep:03d}: lr={current_lr:.6f} "
            f"train_loss={tl:.4f} "
            f"val1_loss={vl_primary:.4f} val1_MAE={mae_primary:.3f}"
        )
        # それ以外の validation セットについては MAE のみ表示 (val2_MAE, val3_MAE, ...)
        if num_valid > 1:
            others = " ".join(
                [f"val{i+1}_MAE={epoch_val_maes[i]:.3f}"
                 for i in range(1, num_valid)]
            )
            msg += " | " + others
        print(msg)

        # ★ ログ1行分を追加（CSV用：loss も MAE もすべて保存）
        log_row = {
            "epoch": ep,
            "train_loss": tl,
            "val_loss": vl_primary,          # 従来の列名は Val#1 に対応
            "val_MAE_deg": mae_primary,      # 同上
            "lr": current_lr,
        }
        # すべての Val セットについて個別列も記録
        for i, (vl_i, mae_i) in enumerate(zip(epoch_val_losses, epoch_val_maes), start=1):
            log_row[f"val_loss_{i}"] = vl_i
            log_row[f"val_MAE_deg_{i}"] = mae_i

        train_log.append(log_row)
        
        # latest / best の保存（best は Val#1 ベースで決定）
        torch.save(model.state_dict(), run_dir / "checkpoints" / "latest.pth")
        if mae_primary < best_mae:
            best_mae, no_improve = mae_primary, 0
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best.pth")
        else:
            no_improve += 1
        
        # 早期終了 (Val#1 ベース)
        if cfg.get("EARLY_STOP") and no_improve >= int(cfg.get("PATIENCE", 10)):
            print(f"Early stop at {ep}")
            break

    # ---------------------------
    # ログ保存 & 学習曲線描画
    # ---------------------------
    log_df = pd.DataFrame(train_log)
    log_df.to_csv(run_dir / "train_log.csv", index=False)
    print(f"📝 Saved training log to {run_dir / 'train_log.csv'}")

    # 学習曲線の保存
    plot_training_curves_basic(train_losses, val_losses_main, val_maes_main, run_dir)
    plot_training_curves_multi(train_losses, val_losses_all, run_dir)
    print(f"✨ Finished {cfg['id']}: Best MAE (Val#1) = {best_mae:.3f}")

# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_lab_v1.yaml")
    args = parser.parse_args()
    
    configs = load_configs(args.config)
    for cfg in configs:
        try:
            run_experiment(cfg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] Exp {cfg['id']} failed: {e}", file=sys.stderr)
