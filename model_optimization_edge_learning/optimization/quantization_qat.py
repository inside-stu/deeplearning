"""PyTorch quantization-aware training example."""

from __future__ import annotations

import torch
from torch import nn


def prepare_qat_model(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    model.train()
    torch.backends.quantized.engine = backend
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    return torch.ao.quantization.prepare_qat(model, inplace=False)


def qat_train_one_epoch(
    prepared_model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    prepared_model.train()
    prepared_model.to(device)
    total_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = prepared_model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.shape[0]
    return total_loss / len(dataloader.dataset)


def convert_qat_model(prepared_model: nn.Module) -> nn.Module:
    prepared_model.cpu()
    prepared_model.eval()
    return torch.ao.quantization.convert(prepared_model, inplace=False)

