"""PyTorch post-training static quantization example."""

from __future__ import annotations

import torch
from torch import nn


def fuse_model_if_available(model: nn.Module, is_qat: bool = False) -> None:
    fuse_model = getattr(model, "fuse_model", None)
    if callable(fuse_model):
        try:
            fuse_model(is_qat=is_qat)
        except TypeError:
            fuse_model()


def prepare_ptq_model(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    model.eval()
    # PTQ flow: fuse Conv/BN/ReLU first, then insert observers with prepare().
    fuse_model_if_available(model, is_qat=False)
    torch.backends.quantized.engine = backend
    model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    return torch.ao.quantization.prepare(model, inplace=False)


@torch.no_grad()
def calibrate(prepared_model: nn.Module, dataloader, device: torch.device, max_batches: int = 20) -> None:
    prepared_model.to(device)
    prepared_model.eval()
    for batch_index, (images, _) in enumerate(dataloader):
        if batch_index >= max_batches:
            break
        prepared_model(images.to(device))


def convert_ptq_model(prepared_model: nn.Module) -> nn.Module:
    prepared_model.cpu()
    prepared_model.eval()
    return torch.ao.quantization.convert(prepared_model, inplace=False)


def ptq_flow(model: nn.Module, calibration_loader, device: torch.device, backend: str = "fbgemm") -> nn.Module:
    prepared = prepare_ptq_model(model, backend=backend)
    calibrate(prepared, calibration_loader, device=device)
    return convert_ptq_model(prepared)
