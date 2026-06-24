"""Model analysis helpers: size, parameter count, sparsity, and latency."""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path

import torch
from torch import nn


def parameter_count(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def sparsity(model: nn.Module) -> float:
    zeros = 0
    total = 0
    skipped_parameters: set[int] = set()

    for module in model.modules():
        for buffer_name, _ in module.named_buffers(recurse=False):
            if not buffer_name.endswith("_mask"):
                continue

            parameter_name = buffer_name[: -len("_mask")]
            effective_parameter = getattr(module, parameter_name)
            original_parameter = getattr(module, f"{parameter_name}_orig", None)
            if original_parameter is not None:
                skipped_parameters.add(id(original_parameter))

            total += effective_parameter.numel()
            zeros += (effective_parameter == 0).sum().item()

    for parameter in model.parameters():
        if id(parameter) in skipped_parameters:
            continue
        total += parameter.numel()
        zeros += (parameter == 0).sum().item()
    return zeros / max(total, 1)


def state_dict_size_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(model.state_dict(), temp_path)
        return temp_path.stat().st_size / (1024 * 1024)
    finally:
        temp_path.unlink(missing_ok=True)


@torch.no_grad()
def benchmark_forward(
    model: nn.Module,
    sample: torch.Tensor,
    warmup: int = 20,
    repeats: int = 100,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    sample = sample.to(device)

    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000)

    mean_ms = statistics.mean(latencies_ms)
    p50_ms = statistics.median(latencies_ms)
    p95_ms = sorted(latencies_ms)[round(0.95 * (len(latencies_ms) - 1))]
    return {"mean_ms": mean_ms, "p50_ms": p50_ms, "p95_ms": p95_ms, "fps": 1000 / mean_ms}
