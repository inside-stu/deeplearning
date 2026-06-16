"""Create a tiny local image classification dataset for week 1.

The generated dataset is intentionally simple and offline:
- class 0: red square
- class 1: green circle
- class 2: blue triangle

It creates train.csv and val.csv with image paths relative to the output root.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw


CLASS_NAMES = {
    0: "red_square",
    1: "green_circle",
    2: "blue_triangle",
}


def draw_sample(label: int, image_size: int, rng: random.Random) -> Image.Image:
    image = Image.new("RGB", (image_size, image_size), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    margin = rng.randint(4, 10)
    jitter = rng.randint(-3, 3)
    box = (
        margin + jitter,
        margin,
        image_size - margin,
        image_size - margin + jitter,
    )

    if label == 0:
        draw.rectangle(box, fill=(220, 40, 40))
    elif label == 1:
        draw.ellipse(box, fill=(40, 180, 70))
    elif label == 2:
        points = [
            (image_size // 2, margin),
            (image_size - margin, image_size - margin),
            (margin, image_size - margin),
        ]
        draw.polygon(points, fill=(60, 90, 220))
    else:
        raise ValueError(f"Unsupported label: {label}")
    return image


def write_split(
    output: Path,
    split: str,
    samples_per_class: int,
    image_size: int,
    rng: random.Random,
) -> None:
    image_dir = output / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{split}.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "label"])
        writer.writeheader()
        for label, class_name in CLASS_NAMES.items():
            for index in range(samples_per_class):
                filename = f"{class_name}_{index:03d}.png"
                relative_path = Path("images") / split / filename
                image = draw_sample(label, image_size=image_size, rng=rng)
                image.save(output / relative_path)
                writer.writerow({"image": relative_path.as_posix(), "label": label})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny local classification dataset.")
    parser.add_argument("--output", default="data/classification_toy")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--val-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    write_split(output, "train", args.train_per_class, args.image_size, rng)
    write_split(output, "val", args.val_per_class, args.image_size, rng)

    print(f"output={output}")
    print(f"train_csv={output / 'train.csv'}")
    print(f"val_csv={output / 'val.csv'}")
    print(f"classes={CLASS_NAMES}")


if __name__ == "__main__":
    main()
