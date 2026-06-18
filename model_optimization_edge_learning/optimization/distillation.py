"""Handwritten teacher-student distillation."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from core.models import TinyClassifier
from core.train_loop import build_dataloaders, evaluate_classifier, train_one_epoch


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine hard-label CE loss and soft-label KL distillation loss."""
    hard_loss = F.cross_entropy(student_logits, labels)

    # KLDivLoss expects log probabilities as input and probabilities as target.
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)

    distill_loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
    return distill_loss, {
        "hard_loss": hard_loss.item(),
        "soft_loss": soft_loss.item(),
        "distill_loss": distill_loss.item(),
        "loss": distill_loss.item(),
    }


def feature_distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    projector: nn.Module | None = None,
) -> torch.Tensor:
    """Make student intermediate features mimic teacher intermediate features."""
    if projector is not None:
        student_features = projector(student_features)
    return F.mse_loss(student_features, teacher_features.detach())


def classifier_logits_and_features(model: TinyClassifier, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = model.backbone(images)
    pooled = model.pool(features).flatten(1)
    logits = model.head(pooled)
    return logits, pooled


def trainable_parameters(*modules: nn.Module | None) -> Iterable[nn.Parameter]:
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if parameter.requires_grad:
                yield parameter


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_student_one_epoch(
    student: TinyClassifier,
    teacher: TinyClassifier,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 4.0,
    alpha: float = 0.7,
    feature_weight: float = 0.1,
    projector: nn.Module | None = None,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    if projector is not None:
        projector.train()

    totals = {
        "hard_loss": 0.0,
        "soft_loss": 0.0,
        "distill_loss": 0.0,
        "feature_loss": 0.0,
        "total_loss": 0.0,
    }

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.shape[0]
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_logits, teacher_features = classifier_logits_and_features(teacher, images)

        student_logits, student_features = classifier_logits_and_features(student, images)
        logit_loss, parts = distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            temperature=temperature,
            alpha=alpha,
        )

        if feature_weight > 0:
            feature_loss = feature_distillation_loss(student_features, teacher_features, projector=projector)
        else:
            feature_loss = student_logits.new_tensor(0.0)

        loss = logit_loss + feature_weight * feature_loss
        loss.backward()
        optimizer.step()

        totals["hard_loss"] += parts["hard_loss"] * batch_size
        totals["soft_loss"] += parts["soft_loss"] * batch_size
        totals["distill_loss"] += parts["distill_loss"] * batch_size
        totals["feature_loss"] += feature_loss.item() * batch_size
        totals["total_loss"] += loss.item() * batch_size

    dataset_size = len(dataloader.dataset)
    return {key: value / dataset_size for key, value in totals.items()}


def train_supervised_model(
    name: str,
    model: TinyClassifier,
    train_loader,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
    num_classes: int,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    metrics: dict[str, object] = {}
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate_classifier(model, val_loader, criterion, device, num_classes=num_classes)
        print(
            f"{name} epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} val_acc={metrics['accuracy']:.4f}"
        )
    return metrics


def train_distilled_student(
    student: TinyClassifier,
    teacher: TinyClassifier,
    train_loader,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
    num_classes: int,
    temperature: float,
    alpha: float,
    feature_weight: float,
    projector: nn.Module | None,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(trainable_parameters(student, projector), lr=lr)
    metrics: dict[str, object] = {}
    for epoch in range(1, epochs + 1):
        losses = train_student_one_epoch(
            student=student,
            teacher=teacher,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=temperature,
            alpha=alpha,
            feature_weight=feature_weight,
            projector=projector,
        )
        metrics = evaluate_classifier(student, val_loader, criterion, device, num_classes=num_classes)
        print(
            f"distilled_student epoch={epoch} "
            f"hard_loss={losses['hard_loss']:.4f} soft_loss={losses['soft_loss']:.4f} "
            f"distill_loss={losses['distill_loss']:.4f} feature_loss={losses['feature_loss']:.4f} "
            f"total_loss={losses['total_loss']:.4f} val_loss={metrics['loss']:.4f} "
            f"val_acc={metrics['accuracy']:.4f}"
        )
    return metrics


def create_feature_projector(student: TinyClassifier, teacher: TinyClassifier, feature_weight: float) -> nn.Module | None:
    if feature_weight <= 0:
        return None
    student_channels = student.backbone.out_channels
    teacher_channels = teacher.backbone.out_channels
    if student_channels == teacher_channels:
        return None
    return nn.Linear(student_channels, teacher_channels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a handwritten distillation demo.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher-width", type=int, default=32)
    parser.add_argument("--student-width", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--feature-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader = build_dataloaders(
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        image_size=args.image_size,
        dataset="csv",
        train_csv=r"data\classification_toy\train.csv",
        val_csv=r"data\classification_toy\val.csv",
        data_root=r"data\classification_toy"
    )
    criterion = nn.CrossEntropyLoss()

    teacher = TinyClassifier(num_classes=args.num_classes, width=args.teacher_width).to(device)
    student_init = TinyClassifier(num_classes=args.num_classes, width=args.student_width)
    normal_student = TinyClassifier(num_classes=args.num_classes, width=args.student_width).to(device)
    distilled_student = TinyClassifier(num_classes=args.num_classes, width=args.student_width).to(device)
    normal_student.load_state_dict(copy.deepcopy(student_init.state_dict()))
    distilled_student.load_state_dict(copy.deepcopy(student_init.state_dict()))

    projector = create_feature_projector(distilled_student, teacher, args.feature_weight)
    if projector is not None:
        projector = projector.to(device)

    print("distillation_setup:")
    print(f"  teacher_width={args.teacher_width} teacher_params={parameter_count(teacher)}")
    print(f"  student_width={args.student_width} student_params={parameter_count(normal_student)}")
    print(f"  temperature={args.temperature} alpha={args.alpha} feature_weight={args.feature_weight}")
    print(f"  projector={projector.__class__.__name__ if projector is not None else 'None'}")

    teacher_metrics = train_supervised_model(
        "teacher",
        teacher,
        train_loader,
        val_loader,
        criterion,
        device,
        args.epochs,
        args.lr,
        args.num_classes,
    )
    normal_student_metrics = train_supervised_model(
        "normal_student",
        normal_student,
        train_loader,
        val_loader,
        criterion,
        device,
        args.epochs,
        args.lr,
        args.num_classes,
    )
    distilled_student_metrics = train_distilled_student(
        student=distilled_student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        num_classes=args.num_classes,
        temperature=args.temperature,
        alpha=args.alpha,
        feature_weight=args.feature_weight,
        projector=projector,
    )

    print("final_comparison:")
    print(f"  teacher_val_acc={teacher_metrics['accuracy']:.4f} val_loss={teacher_metrics['loss']:.4f}")
    print(
        f"  normal_student_val_acc={normal_student_metrics['accuracy']:.4f} "
        f"val_loss={normal_student_metrics['loss']:.4f}"
    )
    print(
        f"  distilled_student_val_acc={distilled_student_metrics['accuracy']:.4f} "
        f"val_loss={distilled_student_metrics['loss']:.4f}"
    )


if __name__ == "__main__":
    main()
