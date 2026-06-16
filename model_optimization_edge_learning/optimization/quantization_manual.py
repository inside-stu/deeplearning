"""Manual tensor quantization for understanding scale and zero point."""

from __future__ import annotations

import torch


def calculate_qparams(
    tensor: torch.Tensor,
    num_bits: int = 8,
    symmetric: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if symmetric:
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1
        max_abs = tensor.abs().max().clamp_min(1e-8)
        scale = max_abs / qmax
        zero_point = torch.tensor(0.0, device=tensor.device)
    else:
        qmin = 0
        qmax = 2**num_bits - 1
        min_val = tensor.min()
        max_val = tensor.max()
        scale = ((max_val - min_val) / (qmax - qmin)).clamp_min(1e-8)
        zero_point = torch.round(qmin - min_val / scale).clamp(qmin, qmax)
    return scale, zero_point, qmin, qmax


def quantize_tensor(tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    return torch.round(tensor / scale + zero_point).clamp(qmin, qmax)


def dequantize_tensor(qtensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    return (qtensor - zero_point) * scale


def fake_quantize(tensor: torch.Tensor, num_bits: int = 8, symmetric: bool = False) -> torch.Tensor:
    scale, zero_point, qmin, qmax = calculate_qparams(tensor, num_bits=num_bits, symmetric=symmetric)
    qtensor = quantize_tensor(tensor, scale, zero_point, qmin, qmax)
    return dequantize_tensor(qtensor, scale, zero_point)


def quantization_error(original: torch.Tensor, restored: torch.Tensor) -> dict[str, float]:
    diff = (original - restored).abs()
    return {"mae": diff.mean().item(), "max_abs_error": diff.max().item()}


def demo() -> None:
    tensor = torch.tensor([-1.0, -0.2, 0.0, 0.3, 1.2, 2.0])
    scale, zero_point, qmin, qmax = calculate_qparams(tensor, symmetric=False)
    qtensor = quantize_tensor(tensor, scale, zero_point, qmin, qmax)
    restored = dequantize_tensor(qtensor, scale, zero_point)
    print("original:", tensor)
    print("scale:", scale.item(), "zero_point:", zero_point.item())
    print("quantized:", qtensor)
    print("restored:", restored)
    print("error:", quantization_error(tensor, restored))


if __name__ == "__main__":
    demo()

