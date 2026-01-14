"""
dataset_manager.py
==================
対応構成：
train09_test/
├── imgs/
│   ├── 0000.png ...
├── labels.csv          # filename,roll,pitch,yaw
├── labels_roll.cache   # ← Roll 用キャッシュ
└── labels_all.cache    # ← 姿勢3軸用キャッシュ
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


# ──────────────────────────────────────────────────────────────
# 共通: キャッシュ構築 & 読み込み（キャッシュ名で分離）
# ──────────────────────────────────────────────────────────────
def _build_or_load_cache(
    root_dir: Path,
    csv_name: str = "labels.csv",
    cache_name: str = "labels.cache",
    fields: Tuple[str, ...] = ("rolls",),
) -> Dict:
    csv_path = root_dir / csv_name
    cache_path = root_dir / cache_name

    if cache_path.exists():
        meta = torch.load(cache_path)
        print(f"[dataset] Cache loaded ({meta['n']} samples) from {cache_name}")
        print("======cache exists, no need to rebuild======")
        return meta

    # ---------- build ----------
    print(f"[dataset] Building cache → {cache_path}")
    paths: List[str] = []
    values: Dict[str, List[float]] = {f: [] for f in fields}
    shapes: List[Tuple[int, int]] = []

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # filename,roll,pitch,yaw
        for row in reader:
            if len(row) < 4:
                continue
            fname = row[0].strip()
            paths.append(fname)
            if "rolls" in fields:
                values["rolls"].append(float(row[1]))
            if "pitches" in fields:
                values["pitches"].append(float(row[2]))
            if "yaws" in fields:
                values["yaws"].append(float(row[3]))

            with Image.open(root_dir / "imgs" / fname) as im:
                shapes.append(im.size[::-1])  # (H, W)

    meta = {
        "paths": paths,
        **values,
        "shapes": shapes,
        "n": len(paths),
        "version": 1,
    }
    torch.save(meta, cache_path)
    print(f"[dataset] Cache saved ({meta['n']} samples) to {cache_name}")
    return meta


# ──────────────────────────────────────────────────────────────
# 1) roll だけ使うデータセット
# ──────────────────────────────────────────────────────────────
class RollDatasetCached(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        transform: callable | None = None,
        rebuild_cache: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.cache_name = "labels_roll.cache"
        cache_path = self.root_dir / self.cache_name
        if rebuild_cache and cache_path.exists():
            cache_path.unlink()
        self.meta = _build_or_load_cache(
            self.root_dir,
            cache_name=self.cache_name,
            fields=("rolls",),
        )

        self.img_dir = self.root_dir / "imgs"
        self.transform = (
            transform
            if transform is not None
            else T.Compose([T.Resize((224, 224)), T.ToTensor()])
        )

    def __len__(self):
        return self.meta["n"]

    def __getitem__(self, idx: int):
        fname = self.meta["paths"][idx]
        roll = self.meta["rolls"][idx]
        with Image.open(self.img_dir / fname).convert("RGB") as img:
            img_tensor: Tensor = self.transform(img)
        return img_tensor, torch.tensor([roll], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
# 2) roll・pitch・yaw を同時に返すデータセット
# ──────────────────────────────────────────────────────────────
class AttitudeDatasetCached(Dataset):
    """
    返り値: (image_tensor, pose_tensor) where pose = [roll, pitch, yaw]
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: callable | None = None,
        rebuild_cache: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.cache_name = "labels_all.cache"
        cache_path = self.root_dir / self.cache_name
        if rebuild_cache and cache_path.exists():
            cache_path.unlink()
        self.meta = _build_or_load_cache(
            self.root_dir,
            cache_name=self.cache_name,
            fields=("rolls", "pitches", "yaws"),
        )

        self.img_dir = self.root_dir / "imgs"
        self.transform = (
            transform
            if transform is not None
            else T.Compose([T.Resize((224, 224)), T.ToTensor()])
        )

    def __len__(self):
        return self.meta["n"]

    def __getitem__(self, idx: int):
        fname = self.meta["paths"][idx]
        roll = self.meta["rolls"][idx]
        pitch = self.meta["pitches"][idx]
        yaw = self.meta["yaws"][idx]

        with Image.open(self.img_dir / fname).convert("RGB") as img:
            img_tensor: Tensor = self.transform(img)

        pose_tensor = torch.tensor([roll, pitch, yaw], dtype=torch.float32)
        return img_tensor, pose_tensor


# ──────────────────────────────────────────────────────────────
# 簡易テスト
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("RollDatasetCached ----------------------------")
    ds_roll = RollDatasetCached("C:/workspace/Github/attitude-estimation/dataset/train09_test")
    print("len:", len(ds_roll))
    img, r = ds_roll[0]
    print("shape:", img.shape, "roll:", r)

    print("\nAttitudeDatasetCached ----------------------------")
    ds_pose = AttitudeDatasetCached("C:/workspace/Github/attitude-estimation/dataset/train09_test")
    print("len:", len(ds_pose))
    img, pose = ds_pose[0]
    print("shape:", img.shape, "pose:", pose)
