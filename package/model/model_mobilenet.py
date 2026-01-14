# model_mobilenet.py
import torch
import torch.nn as nn
import torchvision.models as models
import math

class MobileNetV3_Roll(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        # MobileNetV3 large をベースとして使用
        base = models.mobilenet_v3_large(pretrained=pretrained)
        self.backbone = base.features  # 特徴抽出部
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(960, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # 最終出力: 1次元（rollの角度 [rad]）
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = torch.tanh(x) * math.pi  # 出力を [-π, π] に制限
        return x
