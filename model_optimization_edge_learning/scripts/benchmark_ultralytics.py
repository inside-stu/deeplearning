"""Stage 2 framework reference: benchmark an Ultralytics YOLO model."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO prediction latency.")
    parser.add_argument("--weights", required=True, help="Path to .pt/.onnx/.engine model.")
    parser.add_argument("--source", required=True, help="Image, folder, video, or camera source.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install ultralytics") from exc

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    predict_args = {
        "source": args.source,
        "imgsz": args.imgsz,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
    }

    for _ in range(args.warmup):
        list(model.predict(**predict_args))

    latencies_ms: list[float] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        list(model.predict(**predict_args))
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    p50_ms = statistics.median(latencies_ms)
    p95_ms = percentile(latencies_ms, 95)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    print(f"weights={weights}")
    print(f"source={args.source}")
    print(f"mean_ms={mean_ms:.3f}")
    print(f"p50_ms={p50_ms:.3f}")
    print(f"p95_ms={p95_ms:.3f}")
    print(f"fps={fps:.2f}")


if __name__ == "__main__":
    main()
