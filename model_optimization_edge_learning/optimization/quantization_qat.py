"""PyTorch quantization-aware training example."""

from __future__ import annotations

import torch
from torch import nn


def quantization_api():
    ao_quant = getattr(getattr(torch, "ao", None), "quantization", None)
    if ao_quant is not None and hasattr(ao_quant, "QuantStub") and hasattr(ao_quant, "prepare_qat"):
        return ao_quant
    return torch.quantization


def fuse_model_if_available(model: nn.Module, is_qat: bool = False) -> None:
    fuse_model = getattr(model, "fuse_model", None)
    if callable(fuse_model):
        try:
            fuse_model(is_qat=is_qat)
        except TypeError:
            fuse_model()


def prepare_qat_model(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    model.train()
    # QAT flow: fuse train-time Conv/BN/ReLU first, then insert fake quant modules.
    quant = quantization_api()
    fuse_model_if_available(model, is_qat=True)
    torch.backends.quantized.engine = backend
    model.qconfig = quant.get_default_qat_qconfig(backend)
    return quant.prepare_qat(model, inplace=False)


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
    return quantization_api().convert(prepared_model, inplace=False)
