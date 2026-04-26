from __future__ import annotations

import argparse
import io
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils import resolve_path


HF_DATASETS = ("blanchon/UC_Merced", "torchgeo/ucmerced")

EXPECTED_CLASSES = [
    "agricultural",
    "airplane",
    "baseballdiamond",
    "beach",
    "buildings",
    "chaparral",
    "denseresidential",
    "forest",
    "freeway",
    "golfcourse",
    "harbor",
    "intersection",
    "mediumresidential",
    "mobilehomepark",
    "overpass",
    "parkinglot",
    "river",
    "runway",
    "sparseresidential",
    "storagetanks",
    "tenniscourt",
]


def normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


EXPECTED_CLASS_LOOKUP = {normalized_name(name): name for name in EXPECTED_CLASSES}


def canonical_class_name(value: Any) -> str:
    normalized = normalized_name(value)
    return EXPECTED_CLASS_LOOKUP.get(normalized, normalized)


def image_count_by_class(data_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            counts[path.parent.name] += 1
    return counts


def has_valid_ucmerced_layout(data_dir: Path) -> bool:
    counts = image_count_by_class(data_dir)
    return (
        len(counts) == 21
        and sum(counts.values()) == 2100
        and all(counts.get(class_name) == 100 for class_name in EXPECTED_CLASSES)
    )


def flatten_dataset(dataset: Any) -> Any:
    if hasattr(dataset, "keys"):
        split_names = list(dataset.keys())
        if len(split_names) == 1:
            return dataset[split_names[0]]

        from datasets import concatenate_datasets

        return concatenate_datasets([dataset[split_name] for split_name in split_names])
    return dataset


def load_ucmerced_dataset() -> tuple[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install it with: pip install datasets"
        ) from exc

    last_error: Exception | None = None
    for dataset_name in HF_DATASETS:
        try:
            print(f"Trying Hugging Face dataset: {dataset_name}")
            return dataset_name, flatten_dataset(load_dataset(dataset_name))
        except Exception as exc:
            print(f"Failed to load {dataset_name}: {exc}")
            last_error = exc

    raise RuntimeError(
        "Hugging Face access failed for both blanchon/UC_Merced and torchgeo/ucmerced. "
        "Please manually download UC Merced Land Use from Kaggle or Academic Torrents."
    ) from last_error


def inspect_fields(dataset: Any) -> tuple[str, str | None, dict[int, str]]:
    print("\nDataset fields:")
    print(dataset.features)

    image_field = detect_image_field(dataset)
    label_field, label_mapping = detect_label_field(dataset)

    print(f"\nDetected image field: {image_field}")
    print(f"Detected label/class field: {label_field}")
    if label_mapping:
        print(f"Detected label mapping with {len(label_mapping)} classes.")
    else:
        print("No direct label mapping detected; class names will be inferred from records when possible.")

    return image_field, label_field, label_mapping


def detect_image_field(dataset: Any) -> str:
    try:
        from datasets import Image as DatasetImage
    except ImportError:
        DatasetImage = None

    for field_name, feature in dataset.features.items():
        if DatasetImage is not None and isinstance(feature, DatasetImage):
            return field_name

    preferred_names = ("image", "img", "pixel_values")
    for name in preferred_names:
        if name in dataset.column_names:
            return name

    sample = dataset[0]
    for field_name, value in sample.items():
        if isinstance(value, Image.Image):
            return field_name
        if isinstance(value, dict) and ("bytes" in value or "path" in value):
            return field_name

    raise ValueError("Could not detect an image field in the Hugging Face dataset.")


def detect_label_field(dataset: Any) -> tuple[str | None, dict[int, str]]:
    label_mapping: dict[int, str] = {}

    for field_name, feature in dataset.features.items():
        names = getattr(feature, "names", None)
        if names:
            label_mapping = {index: canonical_class_name(name) for index, name in enumerate(names)}
            return field_name, label_mapping

    for field_name in ("label", "labels", "class", "class_name", "category"):
        if field_name in dataset.column_names:
            return field_name, {}

    return None, {}


def image_from_value(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path") is not None:
            return Image.open(value["path"])
    if isinstance(value, (str, Path)):
        return Image.open(value)
    raise TypeError(f"Unsupported image value type: {type(value)}")


def infer_class_name(
    record: dict[str, Any],
    label_field: str | None,
    label_mapping: dict[int, str],
) -> str:
    if label_field is not None:
        raw_label = record[label_field]
        if isinstance(raw_label, int) and raw_label in label_mapping:
            return label_mapping[raw_label]
        if isinstance(raw_label, str):
            return canonical_class_name(raw_label)
        if raw_label in label_mapping:
            return label_mapping[raw_label]

    for field_name in ("class_name", "class", "category", "label_name", "filename", "file_name", "path"):
        value = record.get(field_name)
        if value is None:
            continue
        if field_name in {"filename", "file_name", "path"}:
            parts = Path(str(value)).parts
            for part in reversed(parts[:-1]):
                candidate = canonical_class_name(part)
                if candidate in EXPECTED_CLASSES:
                    return candidate
        else:
            candidate = canonical_class_name(value)
            if candidate:
                return candidate

    raise ValueError("Could not infer class name for a dataset record.")


def unique_output_path(class_dir: Path, base_name: str, suffix: str = ".jpg") -> Path:
    candidate = class_dir / f"{base_name}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = class_dir / f"{base_name}_{counter:03d}{suffix}"
        counter += 1
    return candidate


def save_dataset(dataset: Any, output_dir: Path, image_field: str, label_field: str | None, label_mapping: dict[int, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_counts: Counter[str] = Counter()
    mapping_rows = []

    if label_mapping:
        mapping_rows = [
            {"label": label, "class_name": class_name}
            for label, class_name in sorted(label_mapping.items())
        ]

    for index, record in enumerate(tqdm(dataset, desc="Saving UCMerced images")):
        class_name = infer_class_name(record, label_field, label_mapping)
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        class_counts[class_name] += 1
        output_path = unique_output_path(class_dir, f"{class_name}_{class_counts[class_name]:03d}")

        with image_from_value(record[image_field]) as image:
            image.convert("RGB").save(output_path, format="JPEG", quality=95)

    if mapping_rows:
        pd.DataFrame(mapping_rows).to_csv(output_dir / "class_mapping.csv", index=False)

    print("\nSaved image counts:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")
    print(f"\nTotal saved images: {sum(class_counts.values())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert UC Merced from Hugging Face.")
    parser.add_argument(
        "--output_dir",
        default="data/raw/uc_merced",
        help="Output directory for class-folder UC Merced images.",
    )
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)

    if has_valid_ucmerced_layout(output_dir):
        print(f"Valid UC Merced dataset already exists at: {output_dir}")
        print("Skipping download.")
        return

    existing_counts = image_count_by_class(output_dir)
    if existing_counts:
        print("Existing UC Merced folder is incomplete or not in the expected layout.")
        print("The converter will avoid duplicate filenames and continue saving into class folders.")

    dataset_name, dataset = load_ucmerced_dataset()
    print(f"\nLoaded dataset source: {dataset_name}")
    print(f"Number of records: {len(dataset)}")

    image_field, label_field, label_mapping = inspect_fields(dataset)
    save_dataset(dataset, output_dir, image_field, label_field, label_mapping)

    if has_valid_ucmerced_layout(output_dir):
        print("\nUC Merced conversion completed successfully.")
    else:
        counts = image_count_by_class(output_dir)
        print("\nUC Merced conversion finished, but the expected 21 x 100 image layout was not found.")
        print(f"Detected classes: {len(counts)}")
        print(f"Detected images: {sum(counts.values())}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
