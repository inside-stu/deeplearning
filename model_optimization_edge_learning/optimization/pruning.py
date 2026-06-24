"""Pruning examples: masks, sparsity, and fine-tuning hooks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import prune


def conv_linear_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    modules: list[tuple[nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            modules.append((module, "weight"))
    return modules


def apply_global_l1_pruning(model: nn.Module, amount: float = 0.3) -> None:
    prune.global_unstructured(
        conv_linear_modules(model),
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )


def remove_pruning_reparam(model: nn.Module) -> None:
    for module, parameter_name in conv_linear_modules(model):
        if hasattr(module, f"{parameter_name}_mask"):
            prune.remove(module, parameter_name)


def module_sparsity(module: nn.Module) -> float:
    zeros = 0
    total = 0

    for buffer_name, _ in module.named_buffers(recurse=False):
        if not buffer_name.endswith("_mask"):
            continue

        parameter_name = buffer_name[: -len("_mask")]
        effective_parameter = getattr(module, parameter_name)
        total += effective_parameter.numel()
        zeros += (effective_parameter == 0).sum().item()

    if total > 0:
        return zeros / total

    for parameter in module.parameters():
        total += parameter.numel()
        zeros += (parameter == 0).sum().item()
    return zeros / max(total, 1)


def channel_l1_importance(conv: nn.Conv2d) -> torch.Tensor:
    if not isinstance(conv, nn.Conv2d):
        raise TypeError("channel_l1_importance expects nn.Conv2d.")
    return conv.weight.detach().abs().sum(dim=(1, 2, 3))


def lowest_importance_channels(conv: nn.Conv2d, amount: int) -> torch.Tensor:
    importance = channel_l1_importance(conv)
    amount = min(amount, importance.numel())
    return torch.argsort(importance)[:amount]
