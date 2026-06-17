"""Evaluate a saved classifier checkpoint and print sample predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

try:
    from .datasets import CsvImageClassificationDataset
    from .models import TinyClassifier
    from .train_loop import build_dataloaders, evaluate_classifier, load_checkpoint
except ImportError:
    from datasets import CsvImageClassificationDataset
    from models import TinyClassifier
    from train_loop import build_dataloaders, evaluate_classifier, load_checkpoint


def unwrap_subset(dataset):
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        return dataset.dataset, list(dataset.indices)
    return dataset, None


def sample_name(dataset, index: int) -> str:
    base_dataset, indices = unwrap_subset(dataset)
    real_index = indices[index] if indices is not None else index
    if isinstance(base_dataset, CsvImageClassificationDataset):
        return base_dataset.samples[real_index][0]
    return f"sample_{real_index}"


def compact_saved_metrics(metrics) -> dict[str, object]:
    if not isinstance(metrics, dict):
        return {}
    compact: dict[str, object] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            compact[key] = f"tensor(shape={tuple(value.shape)})"
        elif isinstance(value, float):
            compact[key] = round(value, 6)
        else:
            compact[key] = value
    return compact


def class_support_text(support: torch.Tensor) -> str:
    parts = [f"{index}:{int(count)}" for index, count in enumerate(support.tolist()) if count > 0]
    return "{" + ", ".join(parts) + "}"


def present_class_mean(values: torch.Tensor, support: torch.Tensor) -> float:
    present_mask = support > 0
    if not present_mask.any():
        return 0.0
    return values[present_mask].mean().item()


def select_sample_indices(dataset_size: int, count: int, mode: str, seed: int) -> list[int]:
    total = min(count, dataset_size)
    if mode == "random":
        generator = torch.Generator().manual_seed(seed)
        return torch.randperm(dataset_size, generator=generator)[:total].tolist()
    return list(range(total))


@torch.no_grad()
def print_sample_predictions(
    model: nn.Module,
    dataset,
    device: torch.device,
    sample_indices: list[int],
    topk: int,
) -> None:
    model.eval()
    print("sample_predictions:")
    for row_index, index in enumerate(sample_indices):
        image, label = dataset[index]
        logits = model(image.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
        k = min(topk, probabilities.numel())
        values, indices = torch.topk(probabilities, k=k)
        topk_text = ", ".join(f"{int(cls)}:{float(prob):.4f}" for cls, prob in zip(indices, values))
        prediction = int(indices[0])
        print(
            f"  {row_index:03d} dataset_index={index} path={sample_name(dataset, index)} "
            f"target={int(label)} pred={prediction} correct={prediction == int(label)} top{k}=[{topk_text}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TinyClassifier checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--train-csv", default=None, help="Only needed because build_dataloaders creates both splits.")
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    _, val_loader = build_dataloaders(
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        image_size=args.image_size,
        dataset=args.dataset,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        data_root=args.data_root,
        max_val_samples=args.max_val_samples,
    )

    model = TinyClassifier(num_classes=args.num_classes).to(device)
    checkpoint = load_checkpoint(Path(args.checkpoint), model)
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_classifier(model, val_loader, criterion, device, num_classes=args.num_classes)
    confusion = metrics["confusion_matrix"]
    support = confusion.sum(dim=1)
    present_classes = [index for index, count in enumerate(support.tolist()) if count > 0]
    sample_indices = select_sample_indices(
        dataset_size=len(val_loader.dataset),
        count=args.sample_count,
        mode=args.sample_mode,
        seed=args.sample_seed,
    )

    print(f"checkpoint={args.checkpoint}")
    print(f"epoch={checkpoint.get('epoch')}")
    print(f"saved_metrics={compact_saved_metrics(checkpoint.get('metrics'))}")
    print(f"val_loss={metrics['loss']:.4f}")
    print(f"val_accuracy_micro={metrics['accuracy']:.4f}")
    print(f"macro_precision_all_classes={metrics['precision'].mean().item():.4f}")
    print(f"macro_recall_all_classes={metrics['recall'].mean().item():.4f}")
    print(f"macro_precision_present_classes={present_class_mean(metrics['precision'], support):.4f}")
    print(f"macro_recall_present_classes={present_class_mean(metrics['recall'], support):.4f}")
    print(f"present_classes={present_classes}")
    print(f"class_support={class_support_text(support)}")
    print(f"confusion_matrix_shape={tuple(confusion.shape)}")
    print(f"sample_mode={args.sample_mode}")
    if args.sample_mode == "random":
        print(f"sample_seed={args.sample_seed}")

    print_sample_predictions(
        model=model,
        dataset=val_loader.dataset,
        device=device,
        sample_indices=sample_indices,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
