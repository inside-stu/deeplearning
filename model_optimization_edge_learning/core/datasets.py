"""Small datasets used by the handwritten training lessons.

Two dataset routes are intentionally provided:
1. SyntheticClassificationDataset: no files needed, good for learning loops.
2. CsvImageClassificationDataset: local images + CSV, good for real Dataset practice.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


def image_to_tensor(image, image_size: int | None = None) -> torch.Tensor:
    image = image.convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class CsvImageClassificationDataset(Dataset):
    """Read image paths and class labels from a CSV file.

    CSV columns default to:
        image,label
        images/0001.jpg,0
    """

    def __init__(
        self,
        csv_path: str | Path,
        root: str | Path | None = None,
        transform: Callable | None = None,
        image_size: int | None = 32,
        image_column: str = "image",
        label_column: str = "label",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root = Path(root) if root is not None else self.csv_path.parent
        self.transform = transform
        self.image_size = image_size
        self.samples: list[tuple[str, int]] = []

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                self.samples.append((row[image_column], int(row[label_column])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for image datasets.") from exc

        relative_path, label = self.samples[index]
        image = Image.open(self.root / relative_path)
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = image_to_tensor(image, image_size=self.image_size)
        return image_tensor, torch.tensor(label, dtype=torch.long)


class SyntheticClassificationDataset(Dataset):
    """Tiny deterministic dataset for smoke tests and learning loops.

    This dataset has no directory on disk. It creates tensors in memory and
    labels them from image brightness, so the training loop can run immediately.
    """

    def __init__(
        self,
        length: int = 256,
        num_classes: int = 3,
        image_size: int = 32,
        seed: int = 42,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(length, 3, image_size, image_size, generator=generator)
        # Labels are intentionally learnable: the image mean decides the class.
        means = self.images.mean(dim=(1, 2, 3))
        self.labels = torch.clamp((means * num_classes).long(), max=num_classes - 1)
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def classification_collate(batch):
    """Merge a list of (image, label) samples into one batched tensor pair."""
    images, labels = zip(*batch)
    return torch.stack(list(images), dim=0), torch.stack(list(labels), dim=0)
