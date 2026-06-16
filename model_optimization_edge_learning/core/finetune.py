"""Finetuning primitives: freezing, BatchNorm handling, and param groups."""

from __future__ import annotations

import torch
from torch import nn

try:
    from .models import TinyClassifier, LoRALinear
except ImportError:
    from models import TinyClassifier, LoRALinear


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def freeze_backbone(model: nn.Module) -> None:
    if not hasattr(model, "backbone"):
        raise ValueError("Model does not expose a .backbone module.")
    set_trainable(model.backbone, False)


def unfreeze_backbone(model: nn.Module) -> None:
    if not hasattr(model, "backbone"):
        raise ValueError("Model does not expose a .backbone module.")
    set_trainable(model.backbone, True)


def freeze_batch_norm(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            child.eval()
            set_trainable(child, False)


def trainable_parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def build_finetune_param_groups(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float = 1e-4,
) -> list[dict]:
    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    return groups


def replace_classifier_with_lora(model: TinyClassifier, rank: int = 4, alpha: float = 8.0) -> None:
    old_head = model.head
    lora_head = LoRALinear(old_head.in_features, old_head.out_features, rank=rank, alpha=alpha, bias=old_head.bias is not None)
    with torch.no_grad():
        lora_head.weight.copy_(old_head.weight)
        if old_head.bias is not None and lora_head.bias is not None:
            lora_head.bias.copy_(old_head.bias)
    model.head = lora_head


def demo() -> None:
    model = TinyClassifier(num_classes=3)
    print("Before freezing:", trainable_parameter_count(model))

    freeze_backbone(model)
    freeze_batch_norm(model)
    print("After freezing backbone and BN:", trainable_parameter_count(model))

    groups = build_finetune_param_groups(model, backbone_lr=1e-4, head_lr=1e-3)
    print("Optimizer groups:", [{"lr": group["lr"], "params": len(group["params"])} for group in groups])

    replace_classifier_with_lora(model, rank=2, alpha=4.0)
    print("After replacing head with LoRA:", trainable_parameter_count(model))


if __name__ == "__main__":
    demo()

