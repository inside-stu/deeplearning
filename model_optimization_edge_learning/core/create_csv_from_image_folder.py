"""Create train/val CSV files from image folders.

Supported label styles:
1. Filename prefix: train/7_123.png -> label 7
2. Class subfolder: train/7/123.png -> label 7

The output CSV uses paths relative to --data-root, which is exactly what
CsvImageClassificationDataset expects.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def infer_label(path: Path, split_dir: Path, label_source: str) -> int:
    if label_source == "filename_prefix":
        prefix = path.stem.split("_", 1)[0]
        return int(prefix)
    if label_source == "parent_dir":
        return int(path.parent.name)
    if label_source == "auto":
        prefix = path.stem.split("_", 1)[0]
        if prefix.isdigit():
            return int(prefix)
        parent = path.parent.name
        if parent.isdigit():
            return int(parent)
    raise ValueError(f"Cannot infer label for {path} with label_source={label_source}")


def collect_rows(data_root: Path, split: str, label_source: str) -> list[dict[str, int | str]]:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    rows: list[dict[str, int | str]] = []
    for path in sorted(split_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = infer_label(path, split_dir=split_dir, label_source=label_source)
        rows.append({"image": path.relative_to(data_root).as_posix(), "label": label})
    if not rows:
        raise ValueError(f"No image files found under {split_dir}")
    return rows


def write_csv(path: Path, rows: list[dict[str, int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "label"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CSV files for image classification folders.")
    parser.add_argument("--data-root", required=True, help="Root containing train/test or train/val folders.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test", help="Use 'test' for MNIST-style extracted data.")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--label-source", choices=["auto", "filename_prefix", "parent_dir"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    train_csv = Path(args.train_csv) if args.train_csv else data_root / "train.csv"
    val_csv = Path(args.val_csv) if args.val_csv else data_root / "val.csv"

    train_rows = collect_rows(data_root, split=args.train_split, label_source=args.label_source)
    val_rows = collect_rows(data_root, split=args.val_split, label_source=args.label_source)
    write_csv(train_csv, train_rows)
    write_csv(val_csv, val_rows)

    labels = sorted({int(row["label"]) for row in train_rows + val_rows})
    print(f"data_root={data_root}")
    print(f"train_csv={train_csv} rows={len(train_rows)}")
    print(f"val_csv={val_csv} rows={len(val_rows)}")
    print(f"labels={labels}")


if __name__ == "__main__":
    main()
