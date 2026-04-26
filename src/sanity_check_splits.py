from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from utils import resolve_path


EXPECTED_TRAIN_PER_CLASS = 70
EXPECTED_VAL_PER_CLASS = 15
EXPECTED_TEST_PER_CLASS = 15


def load_split_csv(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{split_name} CSV not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"image_path", "label", "class_name"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{split_name} CSV missing required columns: {sorted(missing_columns)}")
    return df


def duplicate_paths(df: pd.DataFrame) -> list[str]:
    duplicated = df[df["image_path"].duplicated()]["image_path"].tolist()
    return sorted(set(duplicated))


def path_set(df: pd.DataFrame) -> set[str]:
    return set(df["image_path"].astype(str))


def check_all_paths_exist(df: pd.DataFrame) -> list[str]:
    missing = []
    for image_path in df["image_path"].astype(str):
        if not resolve_path(image_path).exists():
            missing.append(image_path)
    return missing


def label_mapping(df: pd.DataFrame) -> dict[int, str]:
    mapping = {}
    for row in df[["label", "class_name"]].drop_duplicates().itertuples(index=False):
        label = int(row.label)
        class_name = str(row.class_name)
        if label in mapping and mapping[label] != class_name:
            raise ValueError(f"Inconsistent mapping inside split for label {label}: {mapping[label]} vs {class_name}")
        mapping[label] = class_name
    return dict(sorted(mapping.items()))


def validate_label_range(df: pd.DataFrame, valid_labels: set[int], split_name: str) -> list[int]:
    labels = {int(label) for label in df["label"].unique()}
    invalid = sorted(labels - valid_labels)
    if invalid:
        print(f"WARNING: {split_name} has labels outside valid range: {invalid}")
    return invalid


def per_class_counts(df: pd.DataFrame) -> dict[str, int]:
    return {
        str(class_name): int(count)
        for class_name, count in df["class_name"].value_counts().sort_index().items()
    }


def count_mismatches(counts: dict[str, int], expected_count: int) -> dict[str, int]:
    return {
        class_name: count
        for class_name, count in counts.items()
        if count != expected_count
    }


def read_client_csvs(client_dir: Path, prefix: str) -> list[tuple[int, Path, pd.DataFrame]]:
    if not client_dir.exists():
        raise FileNotFoundError(f"{prefix} client directory not found: {client_dir}")

    client_paths = sorted(client_dir.glob("client_*.csv"))
    if not client_paths:
        raise FileNotFoundError(f"No client CSVs found in {client_dir}")

    clients = []
    for path in client_paths:
        client_id_text = path.stem.split("_")[-1]
        try:
            client_id = int(client_id_text)
        except ValueError as exc:
            raise ValueError(f"Could not parse client id from {path.name}") from exc
        clients.append((client_id, path, load_split_csv(path, f"{prefix} {path.name}")))
    return clients


def client_distribution_rows(clients: list[tuple[int, Path, pd.DataFrame]], partition_name: str) -> list[dict[str, Any]]:
    rows = []
    for client_id, path, df in clients:
        counts = df["class_name"].value_counts().sort_index()
        for class_name, count in counts.items():
            rows.append(
                {
                    "partition": partition_name,
                    "client_id": client_id,
                    "client_csv": path.as_posix(),
                    "class_name": class_name,
                    "count": int(count),
                }
            )
    return rows


def check_client_partition(
    clients: list[tuple[int, Path, pd.DataFrame]],
    partition_name: str,
    train_paths: set[str],
    val_paths: set[str],
    test_paths: set[str],
) -> dict[str, Any]:
    all_client_paths: list[str] = []
    sample_counts = {}
    train_only = True
    no_val_test = True

    print(f"\n{partition_name} Client Sample Counts")
    print("=" * (len(partition_name) + 21))
    for client_id, _, df in clients:
        current_paths = list(df["image_path"].astype(str))
        sample_counts[f"client_{client_id}"] = len(current_paths)
        all_client_paths.extend(current_paths)
        outside_train = sorted(set(current_paths) - train_paths)
        val_overlap = sorted(set(current_paths) & val_paths)
        test_overlap = sorted(set(current_paths) & test_paths)
        train_only = train_only and not outside_train
        no_val_test = no_val_test and not val_overlap and not test_overlap
        print(f"client_{client_id}: {len(current_paths)}")

    path_counter = Counter(all_client_paths)
    duplicate_paths_across_clients = sorted(path for path, count in path_counter.items() if count > 1)
    combined_paths = set(all_client_paths)
    missing_train_paths = sorted(train_paths - combined_paths)
    extra_paths = sorted(combined_paths - train_paths)
    covers_train_exactly_once = (
        not duplicate_paths_across_clients
        and not missing_train_paths
        and not extra_paths
        and len(all_client_paths) == len(train_paths)
    )

    return {
        "sample_counts": sample_counts,
        "train_only": train_only,
        "no_val_test_images": no_val_test,
        "covers_train_exactly_once": covers_train_exactly_once,
        "duplicate_paths_across_clients": duplicate_paths_across_clients,
        "missing_train_paths_count": len(missing_train_paths),
        "extra_paths_count": len(extra_paths),
        "total_client_rows": len(all_client_paths),
        "unique_client_paths": len(combined_paths),
    }


def print_distribution_table(rows: list[dict[str, Any]], title: str) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"\n{title}: no rows")
        return

    pivot = (
        df.pivot_table(index="client_id", columns="class_name", values="count", fill_value=0, aggfunc="sum")
        .astype(int)
        .sort_index(axis=1)
    )
    print(f"\n{title}")
    print("=" * len(title))
    print(pivot.to_string())


def inspect_evaluation_config() -> dict[str, Any]:
    centralized_source = resolve_path("src/train_centralized.py").read_text(encoding="utf-8")
    fedavg_source = resolve_path("src/train_fedavg.py").read_text(encoding="utf-8")
    config_source = resolve_path("configs/ablation_plan.yaml").read_text(encoding="utf-8")

    return {
        "a0_uses_configured_test_csv": 'test_dataset = RemoteSensingCSVDataset(config["test_csv"]' in centralized_source,
        "a1_uses_configured_test_csv": 'test_loader = build_eval_loader(config["test_csv"]' in fedavg_source,
        "a1_uses_configured_val_csv": 'val_loader = build_eval_loader(config["val_csv"]' in fedavg_source,
        "a1_client_training_uses_client_split_dir": 'client_csvs = load_client_csvs(config["client_split_dir"]' in fedavg_source,
        "a1_config_iid_client_split_dir": "client_split_dir: data/splits/clients_iid" in config_source,
        "a1_no_val_or_test_client_training_source_pattern": (
            "build_client_loader(client_csv" in fedavg_source
            and 'build_client_loader(config["val_csv"]' not in fedavg_source
            and 'build_client_loader(config["test_csv"]' not in fedavg_source
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check split and client partition leakage for UCMerced.")
    parser.add_argument("--splits_dir", default="data/splits")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    splits_dir = resolve_path(args.splits_dir)
    results_dir = resolve_path(args.results_dir)

    train_df = load_split_csv(splits_dir / "train.csv", "train")
    val_df = load_split_csv(splits_dir / "val.csv", "val")
    test_df = load_split_csv(splits_dir / "test.csv", "test")

    split_dfs = {"train": train_df, "val": val_df, "test": test_df}
    split_paths = {name: path_set(df) for name, df in split_dfs.items()}

    duplicate_report = {name: duplicate_paths(df) for name, df in split_dfs.items()}
    missing_paths = {name: check_all_paths_exist(df) for name, df in split_dfs.items()}

    overlap_report = {
        "train_val": sorted(split_paths["train"] & split_paths["val"]),
        "train_test": sorted(split_paths["train"] & split_paths["test"]),
        "val_test": sorted(split_paths["val"] & split_paths["test"]),
    }

    mappings = {name: label_mapping(df) for name, df in split_dfs.items()}
    reference_mapping = mappings["train"]
    mapping_consistent = all(mapping == reference_mapping for mapping in mappings.values())
    valid_labels = set(reference_mapping)
    invalid_labels = {
        name: validate_label_range(df, valid_labels, name)
        for name, df in split_dfs.items()
    }

    class_counts = {name: per_class_counts(df) for name, df in split_dfs.items()}
    class_count_mismatches = {
        "train": count_mismatches(class_counts["train"], EXPECTED_TRAIN_PER_CLASS),
        "val": count_mismatches(class_counts["val"], EXPECTED_VAL_PER_CLASS),
        "test": count_mismatches(class_counts["test"], EXPECTED_TEST_PER_CLASS),
    }

    iid_clients = read_client_csvs(splits_dir / "clients_iid", "IID")
    noniid_clients = read_client_csvs(splits_dir / "clients_noniid", "non-IID")
    iid_report = check_client_partition(
        iid_clients,
        "IID",
        split_paths["train"],
        split_paths["val"],
        split_paths["test"],
    )
    noniid_report = check_client_partition(
        noniid_clients,
        "Non-IID",
        split_paths["train"],
        split_paths["val"],
        split_paths["test"],
    )

    iid_rows = client_distribution_rows(iid_clients, "iid")
    noniid_rows = client_distribution_rows(noniid_clients, "noniid")
    print_distribution_table(iid_rows, "IID Per-Client Class Distribution")
    print_distribution_table(noniid_rows, "Non-IID Per-Client Class Distribution")

    evaluation_config = inspect_evaluation_config()
    warnings = []
    if any(duplicate_report.values()):
        warnings.append("Duplicate image_path values found inside one or more split CSVs.")
    if any(overlap_report.values()):
        warnings.append("Overlap found between train/val/test splits.")
    if any(missing_paths.values()):
        warnings.append("One or more image paths do not exist on disk.")
    if any(invalid_labels.values()):
        warnings.append("Invalid labels found outside the train label range.")
    if not mapping_consistent:
        warnings.append("class_name/label mapping is inconsistent across splits.")
    if any(class_count_mismatches.values()):
        warnings.append("Per-class split counts do not match expected UCMerced 70/15/15 counts.")
    if not iid_report["covers_train_exactly_once"]:
        warnings.append("IID clients do not cover train.csv exactly once.")
    if not noniid_report["covers_train_exactly_once"]:
        warnings.append("Non-IID clients do not cover train.csv exactly once.")
    if not iid_report["train_only"] or not noniid_report["train_only"]:
        warnings.append("One or more client partitions contain images outside train.csv.")
    if not iid_report["no_val_test_images"] or not noniid_report["no_val_test_images"]:
        warnings.append("One or more client partitions contain val/test images.")
    if not all(evaluation_config.values()):
        warnings.append("Evaluation config source inspection found a failed check.")

    report = {
        "split_csvs_exist": True,
        "split_row_counts": {name: len(df) for name, df in split_dfs.items()},
        "duplicate_paths_inside_splits": {name: len(paths) for name, paths in duplicate_report.items()},
        "overlap_counts": {name: len(paths) for name, paths in overlap_report.items()},
        "all_image_paths_exist": {name: len(paths) == 0 for name, paths in missing_paths.items()},
        "missing_image_path_counts": {name: len(paths) for name, paths in missing_paths.items()},
        "label_mapping_consistent": mapping_consistent,
        "label_mapping": reference_mapping,
        "invalid_labels": invalid_labels,
        "class_counts": class_counts,
        "class_count_mismatches": class_count_mismatches,
        "iid_clients": iid_report,
        "noniid_clients": noniid_report,
        "evaluation_config": evaluation_config,
        "warnings": warnings,
        "passed": not warnings,
    }

    metrics_dir = results_dir / "metrics"
    tables_dir = results_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    report_path = metrics_dir / "split_sanity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    pd.DataFrame(iid_rows).to_csv(tables_dir / "iid_client_class_distribution.csv", index=False)
    pd.DataFrame(noniid_rows).to_csv(tables_dir / "noniid_client_class_distribution.csv", index=False)

    print("\nSplit Sanity Summary")
    print("====================")
    print(f"train/val overlap: {len(overlap_report['train_val'])}")
    print(f"train/test overlap: {len(overlap_report['train_test'])}")
    print(f"val/test overlap: {len(overlap_report['val_test'])}")
    print(f"IID train-only: {iid_report['train_only']}")
    print(f"IID covers train exactly once: {iid_report['covers_train_exactly_once']}")
    print(f"Non-IID train-only: {noniid_report['train_only']}")
    print(f"Non-IID covers train exactly once: {noniid_report['covers_train_exactly_once']}")
    print(f"A0 uses configured test_csv: {evaluation_config['a0_uses_configured_test_csv']}")
    print(f"A1 uses configured test_csv: {evaluation_config['a1_uses_configured_test_csv']}")
    print(f"A1 uses configured val_csv: {evaluation_config['a1_uses_configured_val_csv']}")
    print(f"A1 trains from client_split_dir only: {evaluation_config['a1_client_training_uses_client_split_dir']}")
    print(f"Warnings: {len(warnings)}")
    for warning in warnings:
        print(f"- {warning}")
    print(f"\nSaved report: {report_path}")
    print(f"Saved IID table: {tables_dir / 'iid_client_class_distribution.csv'}")
    print(f"Saved non-IID table: {tables_dir / 'noniid_client_class_distribution.csv'}")

    if warnings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
