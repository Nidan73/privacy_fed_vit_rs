from __future__ import annotations

import argparse
import io
import re
from collections import Counter
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils import resolve_path


HF_DATASETS = ("jonathan-roberts1/NWPU-RESISC45", "timm/resisc45")

EXPECTED_CLASSES = [
    "airplane",
    "airport",
    "baseball_diamond",
    "basketball_court",
    "beach",
    "bridge",
    "chaparral",
    "church",
    "circular_farmland",
    "cloud",
    "commercial_area",
    "dense_residential",
    "desert",
    "forest",
    "freeway",
    "golf_course",
    "ground_track_field",
    "harbor",
    "industrial_area",
    "intersection",
    "island",
    "lake",
    "meadow",
    "medium_residential",
    "mobile_home_park",
    "mountain",
    "overpass",
    "palace",
    "parking_lot",
    "railway",
    "railway_station",
    "rectangular_farmland",
    "river",
    "roundabout",
    "runway",
    "sea_ice",
    "ship",
    "snowberg",
    "sparse_residential",
    "stadium",
    "storage_tank",
    "tennis_court",
    "terrace",
    "thermal_power_station",
    "wetland",
]


def normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


EXPECTED_CLASS_LOOKUP = {normalized_name(name): name for name in EXPECTED_CLASSES}
EXPECTED_CLASS_LOOKUP.update(
    {
        "storagetanks": "storage_tank",
        "storageanks": "storage_tank",
        "sparseresidential": "sparse_residential",
        "denseresidential": "dense_residential",
        "mediumresidential": "medium_residential",
        "mobilehomepark": "mobile_home_park",
        "baseballdiamond": "baseball_diamond",
        "basketballcourt": "basketball_court",
        "golfcourse": "golf_course",
        "groundtrackfield": "ground_track_field",
        "parkinglot": "parking_lot",
        "railwaystation": "railway_station",
        "seaice": "sea_ice",
        "tenniscourt": "tennis_court",
        "thermalpowerstation": "thermal_power_station",
    }
)


def canonical_class_name(value: Any) -> str:
    normalized = normalized_name(value)
    if normalized in EXPECTED_CLASS_LOOKUP:
        return EXPECTED_CLASS_LOOKUP[normalized]

    candidate = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return candidate


def image_count_by_class(data_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            counts[path.parent.name] += 1
    return counts


def has_valid_nwpu_layout(data_dir: Path) -> bool:
    counts = image_count_by_class(data_dir)
    return (
        len(counts) == 45
        and sum(counts.values()) == 31_500
        and all(counts.get(class_name) == 700 for class_name in EXPECTED_CLASSES)
    )


def flatten_dataset(dataset: Any) -> Any:
    if hasattr(dataset, "keys"):
        split_names = list(dataset.keys())
        if len(split_names) == 1:
            return dataset[split_names[0]]

        from datasets import concatenate_datasets

        return concatenate_datasets([dataset[split_name] for split_name in split_names])
    return dataset


def load_nwpu_dataset() -> tuple[str, Any]:
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

    raise RuntimeError("Hugging Face access failed for all configured NWPU-RESISC45 sources.") from last_error


def detect_image_field(dataset: Any) -> str:
    try:
        from datasets import Image as DatasetImage
    except ImportError:
        DatasetImage = None

    for field_name, feature in dataset.features.items():
        if DatasetImage is not None and isinstance(feature, DatasetImage):
            return field_name

    for field_name in ("image", "img", "jpeg", "jpg", "png", "pixel_values"):
        if field_name in dataset.column_names:
            return field_name

    sample = dataset[0]
    for field_name, value in sample.items():
        if isinstance(value, Image.Image):
            return field_name
        if isinstance(value, dict) and ("bytes" in value or "path" in value):
            return field_name
        if isinstance(value, (str, Path)) and Path(str(value)).suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }:
            return field_name

    raise ValueError("Could not detect an image field in the Hugging Face dataset.")


def detect_label_field(dataset: Any) -> tuple[str | None, dict[int, str]]:
    for field_name, feature in dataset.features.items():
        names = getattr(feature, "names", None)
        if names:
            label_mapping = {
                index: canonical_class_name(class_name)
                for index, class_name in enumerate(names)
            }
            return field_name, label_mapping

    for field_name in ("label", "labels", "class", "class_name", "category", "target"):
        if field_name in dataset.column_names:
            return field_name, {}

    return None, {}


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
        print("No direct label mapping detected; class names will be inferred from records.")

    return image_field, label_field, label_mapping


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


def class_from_path_value(value: Any) -> str | None:
    if value is None:
        return None
    parts = Path(str(value)).parts
    for part in reversed(parts[:-1]):
        candidate = canonical_class_name(part)
        if candidate in EXPECTED_CLASSES:
            return candidate
    return None


def infer_class_name(
    record: dict[str, Any],
    label_field: str | None,
    label_mapping: dict[int, str],
) -> str:
    if label_field is not None:
        raw_label = record[label_field]
        if isinstance(raw_label, Integral) and int(raw_label) in label_mapping:
            return label_mapping[int(raw_label)]
        if raw_label in label_mapping:
            return label_mapping[raw_label]
        if isinstance(raw_label, str):
            candidate = canonical_class_name(raw_label)
            if candidate:
                return candidate

    for field_name in ("class_name", "class", "category", "label_name", "filename", "file_name", "path"):
        value = record.get(field_name)
        if value is None:
            continue
        if field_name in {"filename", "file_name", "path"}:
            candidate = class_from_path_value(value)
            if candidate:
                return candidate
        else:
            candidate = canonical_class_name(value)
            if candidate:
                return candidate

    raise ValueError("Could not infer class name for a dataset record.")


def safe_stem(value: Any) -> str:
    stem = Path(str(value)).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return stem or "image"


def record_base_name(record: dict[str, Any], image_field: str, class_name: str, class_count: int) -> str:
    for field_name in ("filename", "file_name", "path", "image_path"):
        value = record.get(field_name)
        if value:
            return safe_stem(value)

    image_value = record.get(image_field)
    if isinstance(image_value, dict) and image_value.get("path"):
        return safe_stem(image_value["path"])
    if isinstance(image_value, (str, Path)):
        return safe_stem(image_value)

    return f"{class_name}_{class_count:05d}"


def unique_output_path(class_dir: Path, base_name: str, suffix: str = ".jpg") -> Path:
    candidate = class_dir / f"{base_name}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = class_dir / f"{base_name}_{counter:03d}{suffix}"
        counter += 1
    return candidate


def mapping_rows_from_observed(
    label_mapping: dict[int, str],
    observed_label_mapping: dict[int, str],
    class_counts: Counter[str],
) -> list[dict[str, Any]]:
    if label_mapping:
        return [
            {"label": label, "class_name": class_name}
            for label, class_name in sorted(label_mapping.items())
        ]
    if observed_label_mapping:
        return [
            {"label": label, "class_name": class_name}
            for label, class_name in sorted(observed_label_mapping.items())
        ]
    return [
        {"label": label, "class_name": class_name}
        for label, class_name in enumerate(sorted(class_counts))
    ]


def save_dataset(
    dataset: Any,
    output_dir: Path,
    image_field: str,
    label_field: str | None,
    label_mapping: dict[int, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_counts = image_count_by_class(output_dir)
    observed_label_mapping: dict[int, str] = {}

    for record in tqdm(dataset, desc="Saving NWPU-RESISC45 images"):
        class_name = infer_class_name(record, label_field, label_mapping)
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        next_count = class_counts[class_name] + 1
        base_name = record_base_name(record, image_field, class_name, next_count)
        output_path = unique_output_path(class_dir, base_name)

        with image_from_value(record[image_field]) as image:
            image.convert("RGB").save(output_path, format="JPEG", quality=95)

        class_counts[class_name] += 1
        if label_field is not None and isinstance(record[label_field], Integral):
            observed_label_mapping[int(record[label_field])] = class_name

    mapping_rows = mapping_rows_from_observed(label_mapping, observed_label_mapping, class_counts)
    pd.DataFrame(mapping_rows).to_csv(output_dir / "class_mapping.csv", index=False)

    print("\nSaved image counts:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")
    print(f"\nTotal saved images: {sum(class_counts.values())}")


def print_manual_instructions(output_dir: Path) -> None:
    print("\nNWPU-RESISC45 download did not complete.")
    print("Manual fallback:")
    print("1. Download NWPU-RESISC45 from Hugging Face or the official dataset source.")
    print(f"2. Arrange images as class folders under: {output_dir}")
    print("3. Expected layout is 45 class folders with 700 images per class.")
    print("4. Re-run dataset_check.py, split_data.py, client_partition.py, and sanity_check_splits.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert NWPU-RESISC45 from Hugging Face.")
    parser.add_argument(
        "--output_dir",
        default="data/raw/nwpu_resisc45",
        help="Output directory for class-folder NWPU-RESISC45 images.",
    )
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)

    if has_valid_nwpu_layout(output_dir):
        print(f"Valid NWPU-RESISC45 dataset already exists at: {output_dir}")
        print("Skipping download.")
        return

    existing_counts = image_count_by_class(output_dir)
    if existing_counts:
        print("Existing NWPU folder is incomplete or not in the expected layout.")
        print("The converter will avoid duplicate filenames and continue saving into class folders.")

    try:
        dataset_name, dataset = load_nwpu_dataset()
        print(f"\nLoaded dataset source: {dataset_name}")
        print(f"Number of records: {len(dataset)}")

        image_field, label_field, label_mapping = inspect_fields(dataset)
        save_dataset(dataset, output_dir, image_field, label_field, label_mapping)
    except Exception as exc:
        print(f"\nDownload/conversion failed: {exc}")
        print_manual_instructions(output_dir)
        raise SystemExit(1) from exc

    if has_valid_nwpu_layout(output_dir):
        print("\nNWPU-RESISC45 conversion completed successfully.")
    else:
        counts = image_count_by_class(output_dir)
        print("\nNWPU-RESISC45 conversion finished, but the expected 45 x 700 image layout was not found.")
        print(f"Detected classes: {len(counts)}")
        print(f"Detected images: {sum(counts.values())}")
        print_manual_instructions(output_dir)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
