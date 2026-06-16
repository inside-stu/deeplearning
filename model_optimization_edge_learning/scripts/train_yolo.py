"""Stage 2 framework reference: train a YOLO model with Ultralytics.

Example:
    python scripts/train_yolo.py --model yolo11n.pt --data configs/dataset.yaml --epochs 50 --imgsz 640 --device 0 --name week01_baseline

This file intentionally keeps the framework-level interface. Study the
handwritten implementation in core/ first, then use this script to map those
concepts back to Ultralytics engineering practice.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO model.")
    parser.add_argument("--model", default="yolo11n.pt", help="Model name or local .pt path.")
    parser.add_argument("--data", required=True, help="YOLO dataset yaml path.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU id, 'cpu', or Ultralytics device string.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--freeze", type=int, default=0, help="Number of layers to freeze.")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="train")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install ultralytics") from exc

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset yaml not found: {data_path}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        freeze=args.freeze,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        resume=args.resume,
    )
    print(results)


if __name__ == "__main__":
    main()
