"""Handwritten teacher-student distillation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> tuple[torch.Tensor, dict[str, float]]:
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)
    loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
    return loss, {"hard_loss": hard_loss.item(), "soft_loss": soft_loss.item(), "loss": loss.item()}


def feature_distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    projector: nn.Module | None = None,
) -> torch.Tensor:
    if projector is not None:
        student_features = projector(student_features)
    return F.mse_loss(student_features, teacher_features.detach())


def train_student_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> float:
    student.train()
    teacher.eval()
    total_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_logits = teacher(images)
        student_logits = student(images)
        loss, _ = distillation_loss(student_logits, teacher_logits, labels, temperature=temperature, alpha=alpha)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.shape[0]

    return total_loss / len(dataloader.dataset)

