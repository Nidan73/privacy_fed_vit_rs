from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED
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


def create_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["label"],
        shuffle=True,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
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
    args = parser.parse_args()

    manifest = build_manifest(args.data_dir)
    train_df, val_df, test_df = create_splits(manifest)
    paths = save_splits(train_df, val_df, test_df, args.output_dir)

    print_split_summary(train_df, val_df, test_df)
    print("\nSaved split CSV files:")
    for split_name, path in paths.items():
        print(f"  {split_name}: {path}")


if __name__ == "__main__":
    main()
