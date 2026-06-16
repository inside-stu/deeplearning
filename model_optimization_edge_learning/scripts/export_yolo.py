"""Stage 2 framework reference: export a YOLO model to ONNX or TensorRT.

Examples:
    python scripts/export_yolo.py --weights runs/detect/train/weights/best.pt --format onnx --imgsz 640 --simplify
    python scripts/export_yolo.py --weights runs/detect/train/weights/best.pt --format engine --imgsz 640 --half --device 0

For the handwritten export path, see deployment/export_onnx_manual.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an Ultralytics YOLO model.")
    parser.add_argument("--weights", required=True, help="Path to .pt weights.")
    parser.add_argument("--format", choices=["onnx", "engine"], default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--half", action="store_true", help="Enable FP16 export when supported.")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 export when supported.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shapes.")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph.")
    parser.add_argument("--data", default=None, help="Dataset yaml for INT8 calibration.")
    parser.add_argument("--batch", type=int, default=1)
    return parser.parse_args()


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
    export_args = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "int8": args.int8,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
        "batch": args.batch,
    }
    if args.device is not None:
        export_args["device"] = args.device
    if args.data is not None:
        export_args["data"] = args.data

    output = model.export(**export_args)
    print(f"Exported: {output}")


if __name__ == "__main__":
    main()
