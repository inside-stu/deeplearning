"""Metrics and detection primitives implemented without framework magic."""

from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


@torch.no_grad()
def confusion_matrix(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    predictions = logits.argmax(dim=1)
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=targets.device)
    for target, prediction in zip(targets.view(-1), predictions.view(-1)):
        matrix[target.long(), prediction.long()] += 1
    return matrix.cpu()


def precision_recall_from_confusion(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    true_positive = matrix.diag().float()
    predicted_positive = matrix.sum(dim=0).float().clamp_min(1)
    actual_positive = matrix.sum(dim=1).float().clamp_min(1)
    precision = true_positive / predicted_positive
    recall = true_positive / actual_positive
    return precision, recall


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    half_w = w / 2
    half_h = h / 2
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)


def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    width = (boxes[..., 2] - boxes[..., 0]).clamp_min(0)
    height = (boxes[..., 3] - boxes[..., 1]).clamp_min(0)
    return width * height


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (right_bottom - left_top).clamp_min(0)
    intersection = wh[..., 0] * wh[..., 1]
    union = box_area_xyxy(boxes1)[:, None] + box_area_xyxy(boxes2)[None, :] - intersection
    return intersection / union.clamp_min(1e-6)


def nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        ious = box_iou_xyxy(boxes[current].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep)


def simple_average_precision(sorted_matches: torch.Tensor, sorted_scores: torch.Tensor, total_gt: int) -> float:
    if total_gt <= 0 or sorted_matches.numel() == 0:
        return 0.0
    order = sorted_scores.argsort(descending=True)
    matches = sorted_matches[order].float()
    tp = matches.cumsum(dim=0)
    fp = (1.0 - matches).cumsum(dim=0)
    recall = tp / total_gt
    precision = tp / (tp + fp).clamp_min(1e-6)
    recall_points = torch.linspace(0, 1, 11, device=matches.device)
    ap = torch.stack([precision[recall >= point].max() if (recall >= point).any() else torch.tensor(0.0) for point in recall_points])
    return ap.mean().item()

