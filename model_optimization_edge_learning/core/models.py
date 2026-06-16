"""Small, readable models for learning training and optimization internals."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class TinyBackbone(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.stem = ConvBlock(3, width, stride=1)
        self.stage1 = ConvBlock(width, width * 2, stride=2)
        self.stage2 = ConvBlock(width * 2, width * 4, stride=2)
        self.out_channels = width * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return x


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, width: int = 32) -> None:
        super().__init__()
        self.backbone = TinyBackbone(width=width)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        return self.head(pooled)


class TinyDetector(nn.Module):
    """A minimal anchor-free detector head for learning tensor shapes.

    It predicts one box, one objectness logit, and class logits per feature-map
    location. This is intentionally small and incomplete compared with YOLO.
    """

    def __init__(self, num_classes: int = 3, width: int = 32) -> None:
        super().__init__()
        self.backbone = TinyBackbone(width=width)
        channels = self.backbone.out_channels
        self.box_head = nn.Conv2d(channels, 4, kernel_size=1)
        self.obj_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.cls_head = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        boxes = torch.sigmoid(self.box_head(features))
        objectness = self.obj_head(features)
        class_logits = self.cls_head(features)
        return {"boxes_cxcywh": boxes, "objectness": objectness, "class_logits": class_logits}


class LoRALinear(nn.Module):
    """A minimal LoRA layer for understanding parameter-efficient finetuning."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.requires_grad = False
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        if self.bias is not None:
            self.bias.requires_grad = False

        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        update = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
        return base + update

    def merge_weight(self) -> torch.Tensor:
        return self.weight + (self.lora_b @ self.lora_a) * self.scaling

