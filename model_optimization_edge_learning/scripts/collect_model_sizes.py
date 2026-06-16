"""Print model artifact sizes for metric tables."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect model file sizes.")
    parser.add_argument("paths", nargs="+", help="Model files or directories.")
    return parser.parse_args()


def iter_files(paths: list[str]):
    suffixes = {".pt", ".onnx", ".engine", ".plan", ".torchscript"}
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in suffixes:
                    yield child


def main() -> None:
    args = parse_args()
    print("path,size_mb")
    for path in iter_files(args.paths):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{path},{size_mb:.3f}")


if __name__ == "__main__":
    main()
