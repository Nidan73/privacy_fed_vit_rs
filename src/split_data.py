from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, TEST_RATIO, TRAIN_RATIO, VAL_RATIO
from utils import project_relative, resolve_path, scan_image_paths


def build_manifest(data_dir: str | Path) -> pd.DataFrame:
    image_paths = scan_image_paths(data_dir)
    if not image_paths:
        raise FileNotFoundError(f"No image files found under: {resolve_path(data_dir)}")

    class_names = sorted({path.parent.name for path in image_paths})
    class_to_label = {class_name: label for label, class_name in enumerate(class_names)}

    rows = [
        {
            "image_path": project_relative(path),
            "label": class_to_label[path.parent.name],
            "class_name": path.parent.name,
        }
        for path in image_paths
    ]
    return pd.DataFrame(rows).sort_values(["class_name", "image_path"]).reset_index(drop=True)


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    invalid = {name: value for name, value in ratios.items() if value <= 0 or value >= 1}
    if invalid:
        raise ValueError(f"Split ratios must be between 0 and 1: {invalid}")

    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.8f}")


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_ratios(train_ratio, val_ratio, test_ratio)

    train_df, temp_df = train_test_split(
        df,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=df["label"],
        shuffle=True,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
        stratify=temp_df["label"],
        shuffle=True,
    )
    return (
        train_df.sort_values("image_path").reset_index(drop=True),
        val_df.sort_values("image_path").reset_index(drop=True),
        test_df.sort_values("image_path").reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str | Path
) -> dict[str, Path]:
    output_root = resolve_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": output_root / "train.csv",
        "val": output_root / "val.csv",
        "test": output_root / "test.csv",
    }
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)
    return paths


def print_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    total = len(train_df) + len(val_df) + len(test_df)
    print("\nSplit Summary")
    print("=============")
    print(f"Total images: {total}")
    print(f"Train: {len(train_df)} ({len(train_df) / total:.1%})")
    print(f"Val:   {len(val_df)} ({len(val_df) / total:.1%})")
    print(f"Test:  {len(test_df)} ({len(test_df) / total:.1%})")

    print("\nPer-class counts:")
    combined_classes = sorted(
        set(train_df["class_name"]) | set(val_df["class_name"]) | set(test_df["class_name"])
    )
    for class_name in combined_classes:
        train_count = int((train_df["class_name"] == class_name).sum())
        val_count = int((val_df["class_name"] == class_name).sum())
        test_count = int((test_df["class_name"] == class_name).sum())
        print(f"  {class_name}: train={train_count}, val={val_count}, test={test_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified train/val/test CSV splits.")
    parser.add_argument("--data_dir", required=True, help="Dataset root directory to scan.")
    parser.add_argument("--output_dir", required=True, help="Directory where split CSVs are saved.")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test_ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    manifest = build_manifest(args.data_dir)
    train_df, val_df, test_df = create_splits(
        manifest,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    paths = save_splits(train_df, val_df, test_df, args.output_dir)

    print_split_summary(train_df, val_df, test_df)
    print("\nSaved split CSV files:")
    for split_name, path in paths.items():
        print(f"  {split_name}: {path}")


if __name__ == "__main__":
    main()
