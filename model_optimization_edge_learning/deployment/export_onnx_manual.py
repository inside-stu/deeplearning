"""Manual PyTorch to ONNX export without framework wrappers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from core.models import TinyClassifier
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from core.models import TinyClassifier


def export_classifier_onnx(
    checkpoint: str | Path | None,
    output: str | Path,
    num_classes: int = 3,
    image_size: int = 32,
    dynamic_batch: bool = True,
    opset: int = 17,
) -> Path:
    model = TinyClassifier(num_classes=num_classes)
    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"images": {0: "batch"}, "logits": {0: "batch"}}

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TinyClassifier to ONNX.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="models/tiny_classifier.onnx")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--static-batch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = export_classifier_onnx(
        checkpoint=args.checkpoint,
        output=args.output,
        num_classes=args.num_classes,
        image_size=args.image_size,
        dynamic_batch=not args.static_batch,
        opset=args.opset,
    )
    print(f"Exported ONNX: {output}")


if __name__ == "__main__":
    main()

