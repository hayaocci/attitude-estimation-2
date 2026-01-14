# model_resnet_small.py

import torch
import torch.nn as nn
import torchvision.models as models
import math


class ResNet18_Roll(nn.Module):
    """
    入力: RGB画像 [B, 3, 224, 224]
    出力: Roll角 [B, 1], 単位はラジアン。[-π, π] の範囲に正規化。
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 最終全結合層（元は1000クラス用）をRoll出力用に変更
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Tanh()  # [-1, 1] に正規化
        )

    def forward(self, x):
        x = self.backbone(x)         # [-1, 1]
        x = x * math.pi              # [-π, π]
        return x
