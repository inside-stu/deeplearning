"""Benchmark an ONNX model with ONNX Runtime using synthetic input.

This measures model forward latency only. It does not include image decode,
resize, normalization, NMS, or drawing.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime latency.")
    parser.add_argument("--model", required=True, help="Path to .onnx model.")
    parser.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--shape", default="1,3,640,640", help="Input shape as N,C,H,W.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    args = parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install onnxruntime-gpu") from exc

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"ONNX model not found: {model_path}")

    providers = {
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }[args.provider]

    session = ort.InferenceSession(str(model_path), providers=providers)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    shape = tuple(int(part) for part in args.shape.split(","))
    dtype = np.float16 if args.dtype == "float16" else np.float32
    sample = np.random.random(shape).astype(dtype)

    for _ in range(args.warmup):
        session.run(None, {input_name: sample})

    latencies_ms: list[float] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        session.run(None, {input_name: sample})
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    p50_ms = statistics.median(latencies_ms)
    p95_ms = percentile(latencies_ms, 95)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    print(f"model={model_path}")
    print(f"providers={session.get_providers()}")
    print(f"shape={shape}")
    print(f"mean_ms={mean_ms:.3f}")
    print(f"p50_ms={p50_ms:.3f}")
    print(f"p95_ms={p95_ms:.3f}")
    print(f"fps={fps:.2f}")


if __name__ == "__main__":
    main()
