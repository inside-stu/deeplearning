"""A handwritten PyTorch training loop for stage 1 learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

try:
    from .datasets import CsvImageClassificationDataset, SyntheticClassificationDataset, classification_collate
    from .metrics import accuracy, confusion_matrix, precision_recall_from_confusion
    from .models import TinyClassifier
except ImportError:
    from datasets import CsvImageClassificationDataset, SyntheticClassificationDataset, classification_collate
    from metrics import accuracy, confusion_matrix, precision_recall_from_confusion
    from models import TinyClassifier


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    autocast_enabled = scaler is not None and device.type == "cuda"

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None and autocast_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.shape[0]

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.shape[0]
        total_accuracy += accuracy(logits, labels) * images.shape[0]
        matrix += confusion_matrix(logits, labels, num_classes=num_classes)

    precision, recall = precision_recall_from_confusion(matrix)
    return {
        "loss": total_loss / len(dataloader.dataset),
        "accuracy": total_accuracy / len(dataloader.dataset),
        "precision": precision,
        "recall": recall,
        "confusion_matrix": matrix,
    }


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def build_dataloaders(
    batch_size: int,
    num_classes: int,
    image_size: int,
    dataset: str = "synthetic",
    train_csv: str | Path | None = None,
    val_csv: str | Path | None = None,
    data_root: str | Path | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    if dataset == "synthetic":
        full_dataset = SyntheticClassificationDataset(length=512, num_classes=num_classes, image_size=image_size)
        train_dataset, val_dataset = random_split(full_dataset, [400, 112], generator=torch.Generator().manual_seed(42))
    elif dataset == "csv":
        if train_csv is None or val_csv is None:
            raise ValueError("--train-csv and --val-csv are required when --dataset csv.")
        train_dataset = CsvImageClassificationDataset(train_csv, root=data_root, image_size=image_size)
        val_dataset = CsvImageClassificationDataset(val_csv, root=data_root, image_size=image_size)
    else:
        raise ValueError(f"Unsupported dataset route: {dataset}")

    if max_train_samples is not None:
        train_dataset = limit_dataset(train_dataset, max_train_samples, seed=42)
    if max_val_samples is not None:
        val_dataset = limit_dataset(val_dataset, max_val_samples, seed=43)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=classification_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=classification_collate)
    return train_loader, val_loader


def limit_dataset(dataset, max_samples: int, seed: int) -> Subset:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    if max_samples >= len(dataset):
        return Subset(dataset, list(range(len(dataset))))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Handwritten classifier training loop.")
    parser.add_argument("--dataset", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--train-csv", default=None, help="Training CSV for --dataset csv.")
    parser.add_argument("--val-csv", default=None, help="Validation CSV for --dataset csv.")
    parser.add_argument("--data-root", default=None, help="Root directory for image paths in CSV files.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional subset size for quick smoke tests.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional validation subset size for quick smoke tests.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint", default="models/handwritten_tiny_classifier.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    train_loader, val_loader = build_dataloaders(
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        image_size=args.image_size,
        dataset=args.dataset,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        data_root=args.data_root,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    model = TinyClassifier(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        metrics = evaluate_classifier(model, val_loader, criterion, device, num_classes=args.num_classes)
        scheduler.step()

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} val_acc={metrics['accuracy']:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if metrics["accuracy"] >= best_accuracy:
            best_accuracy = float(metrics["accuracy"])
            save_checkpoint(args.checkpoint, model, optimizer, epoch, metrics)


if __name__ == "__main__":
    main()
