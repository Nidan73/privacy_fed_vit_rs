from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils import project_relative, resolve_path, scan_image_paths


def inspect_dataset(data_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    image_paths = scan_image_paths(data_dir)
    class_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    unreadable: list[str] = []
    class_modes: dict[str, Counter[str]] = defaultdict(Counter)
    class_sizes: dict[str, Counter[str]] = defaultdict(Counter)
    class_unreadable: Counter[str] = Counter()

    for image_path in tqdm(image_paths, desc="Checking images"):
        class_name = image_path.parent.name
        class_counts[class_name] += 1

        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                mode = image.mode
                size = f"{image.size[0]}x{image.size[1]}"
        except Exception:
            unreadable.append(project_relative(image_path))
            class_unreadable[class_name] += 1
            continue

        mode_counts[mode] += 1
        size_counts[size] += 1
        class_modes[class_name][mode] += 1
        class_sizes[class_name][size] += 1

    rows = []
    for class_name in sorted(class_counts):
        rows.append(
            {
                "class_name": class_name,
                "image_count": class_counts[class_name],
                "corrupted_image_count": class_unreadable[class_name],
                "modes": "; ".join(
                    f"{mode}:{count}" for mode, count in class_modes[class_name].most_common()
                ),
                "sizes": "; ".join(
                    f"{size}:{count}" for size, count in class_sizes[class_name].most_common()
                ),
            }
        )

    summary = {
        "data_dir": str(resolve_path(data_dir)),
        "total_classes": len(class_counts),
        "total_images": len(image_paths),
        "unreadable_images": len(unreadable),
        "mode_counts": dict(mode_counts),
        "size_counts": dict(size_counts),
        "unreadable_paths": unreadable,
    }
    return pd.DataFrame(rows), summary


def print_summary(class_df: pd.DataFrame, summary: dict) -> None:
    print("\nDataset Summary")
    print("===============")
    print(f"Data directory: {summary['data_dir']}")
    print(f"Total classes: {summary['total_classes']}")
    print(f"Total images: {summary['total_images']}")
    print(f"Unreadable/corrupted images: {summary['unreadable_images']}")
    print(f"Image modes: {summary['mode_counts']}")
    print(f"Image sizes: {summary['size_counts']}")

    if class_df.empty:
        print("\nNo image files were found.")
        return

    print("\nImages per class:")
    for row in class_df.itertuples(index=False):
        print(f"  {row.class_name}: {row.image_count}")

    if summary["unreadable_paths"]:
        print("\nUnreadable image paths:")
        for path in summary["unreadable_paths"]:
            print(f"  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a remote sensing image dataset.")
    parser.add_argument("--data_dir", required=True, help="Dataset root directory to scan.")
    parser.add_argument(
        "--output_csv",
        default="results/metrics/dataset_summary.csv",
        help="Path where the per-class dataset summary CSV is saved.",
    )
    args = parser.parse_args()

    class_df, summary = inspect_dataset(args.data_dir)

    output_path = resolve_path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_df.to_csv(output_path, index=False)

    print_summary(class_df, summary)
    print(f"\nSaved class summary CSV to: {output_path}")

    if summary["total_images"] == 0:
        raise SystemExit("No images found. Dataset preparation cannot continue.")


if __name__ == "__main__":
    main()
