"""Manual ONNX Runtime inference and latency benchmark."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np


def build_session(model_path: str | Path, provider: str = "cuda"):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install onnxruntime-gpu or onnxruntime before running inference.") from exc

    providers = {
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }[provider]
    return ort.InferenceSession(str(model_path), providers=providers)


def run_benchmark(
    model_path: str | Path,
    shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    provider: str = "cuda",
    warmup: int = 20,
    repeats: int = 100,
) -> dict[str, float | list[str]]:
    session = build_session(model_path, provider=provider)
    input_name = session.get_inputs()[0].name
    sample = np.random.rand(*shape).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: sample})

    latencies_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        session.run(None, {input_name: sample})
        latencies_ms.append((time.perf_counter() - start) * 1000)

    mean_ms = statistics.mean(latencies_ms)
    p50_ms = statistics.median(latencies_ms)
    p95_ms = sorted(latencies_ms)[round(0.95 * (len(latencies_ms) - 1))]
    return {
        "providers": session.get_providers(),
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "fps": 1000 / mean_ms,
    }


def parse_shape(raw: str) -> tuple[int, int, int, int]:
    parts = tuple(int(part.strip()) for part in raw.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Shape must be N,C,H,W.")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX Runtime benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--shape", type=parse_shape, default=(1, 3, 32, 32))
    parser.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_benchmark(args.model, shape=args.shape, provider=args.provider, warmup=args.warmup, repeats=args.repeats)
    for key, value in metrics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

