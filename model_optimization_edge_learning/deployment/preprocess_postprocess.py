"""Readable preprocessing and postprocessing helpers."""

from __future__ import annotations

import numpy as np


def normalize_image_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError("Expected HWC image.")
    image = image.astype(np.float32) / 255.0
    return np.transpose(image, (2, 0, 1))[None, ...]


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=axis, keepdims=True)


def topk_classification(logits: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    probabilities = softmax(logits, axis=-1).reshape(-1)
    indices = np.argsort(probabilities)[::-1][:k]
    return [(int(index), float(probabilities[index])) for index in indices]

