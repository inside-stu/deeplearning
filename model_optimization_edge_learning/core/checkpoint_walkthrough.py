"""Checkpoint save/load walkthrough for beginners.

This script demonstrates the full loop:
1. Create or reuse a checkpoint.
2. Build a fresh model with the same architecture.
3. Load checkpoint["model"] into that model.
4. Run the same input through two loaded models.
5. Check whether their outputs are identical.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

try:
    from .models import TinyClassifier
    from .train_loop import build_dataloaders, load_checkpoint, save_checkpoint, train_one_epoch
except ImportError:
    from models import TinyClassifier
    from train_loop import build_dataloaders, load_checkpoint, save_checkpoint, train_one_epoch


def create_checkpoint_if_missing(
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int,
    image_size: int,
    batch_size: int,
    lr: float,
) -> None:
    if checkpoint_path.exists():
        return

    print(f"Checkpoint not found, training one tiny epoch: {checkpoint_path}")
    train_loader, _ = build_dataloaders(batch_size=batch_size, num_classes=num_classes, image_size=image_size)
    model = TinyClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        metrics={"train_loss": train_loss, "note": "created by checkpoint_walkthrough"},
    )


def compare_loaded_models(
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int,
    image_size: int,
    atol: float,
) -> dict[str, float | bool | int | object]:
    torch.manual_seed(123)
    sample = torch.randn(1, 3, image_size, image_size, device=device)

    model_a = TinyClassifier(num_classes=num_classes).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model_a)
    model_a.eval()

    model_b = TinyClassifier(num_classes=num_classes).to(device)
    model_b.load_state_dict(checkpoint["model"])
    model_b.eval()

    random_model = TinyClassifier(num_classes=num_classes).to(device)
    random_model.eval()

    with torch.no_grad():
        output_a = model_a(sample)
        output_b = model_b(sample)
        random_output = random_model(sample)

    loaded_diff = (output_a - output_b).abs().max().item()
    random_diff = (output_a - random_output).abs().max().item()
    return {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics"),
        "loaded_models_allclose": torch.allclose(output_a, output_b, atol=atol),
        "loaded_models_max_abs_diff": loaded_diff,
        "random_vs_loaded_max_abs_diff": random_diff,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain and verify checkpoint loading.")
    parser.add_argument("--checkpoint", default="models/handwritten_tiny_classifier.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--fail-if-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)

    if args.fail_if_missing and not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint_path}")

    create_checkpoint_if_missing(
        checkpoint_path=checkpoint_path,
        device=device,
        num_classes=args.num_classes,
        image_size=args.image_size,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    result = compare_loaded_models(
        checkpoint_path=checkpoint_path,
        device=device,
        num_classes=args.num_classes,
        image_size=args.image_size,
        atol=args.atol,
    )

    print(f"checkpoint={checkpoint_path}")
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
