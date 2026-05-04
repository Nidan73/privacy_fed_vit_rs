from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DEFAULT_NUM_CLIENTS, RANDOM_SEED
from utils import resolve_path


def split_dataframe(df: pd.DataFrame, num_parts: int) -> list[pd.DataFrame]:
    split_indices = np.array_split(np.arange(len(df)), num_parts)
    return [df.iloc[indices].copy() for indices in split_indices]


def split_counts_from_proportions(num_samples: int, proportions: np.ndarray) -> np.ndarray:
    raw_counts = proportions * num_samples
    counts = np.floor(raw_counts).astype(int)
    remainder = num_samples - int(counts.sum())
    if remainder > 0:
        fractional_order = np.argsort(raw_counts - counts)[::-1]
        counts[fractional_order[:remainder]] += 1
    return counts


def shuffle_or_empty(parts: list[pd.DataFrame], columns: list[str], seed: int) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def create_iid_partitions(train_df: pd.DataFrame, num_clients: int, seed: int = RANDOM_SEED) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    client_parts = [[] for _ in range(num_clients)]

    for _, class_df in train_df.groupby("label", sort=True):
        shuffled = class_df.sample(frac=1, random_state=int(rng.integers(0, 1_000_000)))
        for client_id, part in enumerate(split_dataframe(shuffled, num_clients)):
            client_parts[client_id].append(part)

    return [shuffle_or_empty(parts, list(train_df.columns), seed) for parts in client_parts]


def create_noniid_partitions(train_df: pd.DataFrame, num_clients: int, seed: int = RANDOM_SEED) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    client_parts = [[] for _ in range(num_clients)]

    # Simple class-skew strategy:
    # each class is assigned a primary client that receives about 60% of that class,
    # while the remaining samples are spread across the other clients. This creates
    # client distributions that are visibly non-IID without adding Dirichlet logic yet.
    for class_index, (_, class_df) in enumerate(train_df.groupby("label", sort=True)):
        primary_client = class_index % num_clients
        shuffled = class_df.sample(frac=1, random_state=int(rng.integers(0, 1_000_000)))

        primary_count = int(round(0.60 * len(shuffled)))
        primary_rows = shuffled.iloc[:primary_count]
        remaining_rows = shuffled.iloc[primary_count:]

        client_parts[primary_client].append(primary_rows)

        other_clients = [client_id for client_id in range(num_clients) if client_id != primary_client]
        if other_clients:
            for client_id, part in zip(other_clients, split_dataframe(remaining_rows, len(other_clients))):
                client_parts[client_id].append(part)

    return [shuffle_or_empty(parts, list(train_df.columns), seed) for parts in client_parts]


def create_dirichlet_partitions(
    train_df: pd.DataFrame,
    num_clients: int,
    alpha: float,
    seed: int = RANDOM_SEED,
) -> list[pd.DataFrame]:
    if alpha <= 0:
        raise ValueError("dirichlet_alpha must be greater than 0.")

    rng = np.random.default_rng(seed)
    client_parts = [[] for _ in range(num_clients)]

    for _, class_df in train_df.groupby("label", sort=True):
        shuffled = class_df.sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
        proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=float))
        counts = split_counts_from_proportions(len(shuffled), proportions)

        start = 0
        for client_id, count in enumerate(counts):
            end = start + int(count)
            if end > start:
                client_parts[client_id].append(shuffled.iloc[start:end].copy())
            start = end

    return [shuffle_or_empty(parts, list(train_df.columns), seed) for parts in client_parts]


def dirichlet_partition_name(alpha: float) -> str:
    alpha_text = f"{alpha:g}".replace(".", "").replace("-", "m")
    return f"clients_dirichlet_alpha{alpha_text}"


def save_client_partitions(
    partitions: list[pd.DataFrame], output_dir: Path, partition_name: str
) -> list[Path]:
    partition_dir = output_dir / partition_name
    partition_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for client_id, client_df in enumerate(partitions):
        path = partition_dir / f"client_{client_id}.csv"
        client_df.to_csv(path, index=False)
        saved_paths.append(path)
    return saved_paths


def print_partition_summary(partitions: list[pd.DataFrame], title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    for client_id, client_df in enumerate(partitions):
        class_counts = client_df["class_name"].value_counts().sort_index()
        zero_classes = int(len(set().union(*(set(df["class_name"]) for df in partitions))) - len(class_counts))
        counts_text = ", ".join(f"{name}:{count}" for name, count in class_counts.items())
        print(f"client_{client_id}: total={len(client_df)} | classes={len(class_counts)} | zero_classes={zero_classes}")
        print(f"  {counts_text}")


def partition_overlap_count(partitions: list[pd.DataFrame]) -> int:
    all_paths: list[str] = []
    for client_df in partitions:
        all_paths.extend(client_df["image_path"].astype(str).tolist())
    path_counts = Counter(all_paths)
    return sum(count - 1 for count in path_counts.values() if count > 1)


def client_distribution_rows(
    partitions: list[pd.DataFrame],
    partition_name: str,
    saved_paths: list[Path],
    all_classes: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for client_id, (client_df, path) in enumerate(zip(partitions, saved_paths)):
        counts = client_df["class_name"].value_counts()
        for class_name in all_classes:
            rows.append(
                {
                    "partition": partition_name,
                    "client_id": client_id,
                    "client_csv": path.as_posix(),
                    "class_name": class_name,
                    "count": int(counts.get(class_name, 0)),
                }
            )
    return rows


def build_partition_summary(
    partitions: list[pd.DataFrame],
    train_df: pd.DataFrame,
    all_classes: list[str],
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    train_paths = set(train_df["image_path"].astype(str))
    all_client_paths: list[str] = []
    client_sample_counts = {}
    classes_per_client = {}
    min_samples_per_class = {}
    max_samples_per_class = {}
    zero_class_entries = {}

    for client_id, client_df in enumerate(partitions):
        client_name = f"client_{client_id}"
        paths = client_df["image_path"].astype(str).tolist()
        all_client_paths.extend(paths)

        class_counts = client_df["class_name"].value_counts()
        full_counts = [int(class_counts.get(class_name, 0)) for class_name in all_classes]
        client_sample_counts[client_name] = int(len(client_df))
        classes_per_client[client_name] = int(sum(count > 0 for count in full_counts))
        min_samples_per_class[client_name] = int(min(full_counts)) if full_counts else 0
        max_samples_per_class[client_name] = int(max(full_counts)) if full_counts else 0
        zero_class_entries[client_name] = int(sum(count == 0 for count in full_counts))

    path_counts = Counter(all_client_paths)
    duplicate_count = sum(count - 1 for count in path_counts.values() if count > 1)
    unique_client_paths = set(all_client_paths)
    missing_train_paths = train_paths - unique_client_paths
    extra_paths = unique_client_paths - train_paths
    val_overlap = set() if val_df is None else unique_client_paths & set(val_df["image_path"].astype(str))
    test_overlap = set() if test_df is None else unique_client_paths & set(test_df["image_path"].astype(str))

    return {
        "client_sample_counts": client_sample_counts,
        "classes_per_client": classes_per_client,
        "min_samples_per_class_per_client": min_samples_per_class,
        "max_samples_per_class_per_client": max_samples_per_class,
        "zero_class_entries_per_client": zero_class_entries,
        "total_train_rows": int(len(train_df)),
        "total_client_rows": int(len(all_client_paths)),
        "unique_client_paths": int(len(unique_client_paths)),
        "total_train_coverage": int(len(train_paths & unique_client_paths)),
        "duplicate_count": int(duplicate_count),
        "overlap_between_clients": int(partition_overlap_count(partitions)),
        "missing_train_count": int(len(missing_train_paths)),
        "extra_not_train_count": int(len(extra_paths)),
        "covers_train_exactly_once": bool(
            len(all_client_paths) == len(train_paths)
            and not duplicate_count
            and not missing_train_paths
            and not extra_paths
        ),
        "val_images_in_clients": int(len(val_overlap)),
        "test_images_in_clients": int(len(test_overlap)),
    }


def read_optional_split(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def save_partition_reports(
    partitions: list[pd.DataFrame],
    saved_paths: list[Path],
    output_dir: Path,
    partition_name: str,
    train_df: pd.DataFrame,
) -> tuple[Path, Path, dict[str, Any]]:
    dataset_name = output_dir.name
    report_name = partition_name.removeprefix("clients_")
    metrics_dir = resolve_path("results/metrics")
    tables_dir = resolve_path("results/tables")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_classes = sorted(train_df["class_name"].astype(str).unique())
    val_df = read_optional_split(output_dir / "val.csv")
    test_df = read_optional_split(output_dir / "test.csv")

    rows = client_distribution_rows(partitions, partition_name, saved_paths, all_classes)
    summary = build_partition_summary(partitions, train_df, all_classes, val_df=val_df, test_df=test_df)

    table_path = tables_dir / f"{dataset_name}_{report_name}_client_class_distribution.csv"
    summary_path = metrics_dir / f"{dataset_name}_{report_name}_partition_summary.json"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return table_path, summary_path, summary


def warn_on_small_clients(partitions: list[pd.DataFrame], min_client_samples: int | None) -> None:
    sample_counts = [len(df) for df in partitions]
    average_count = sum(sample_counts) / max(len(sample_counts), 1)
    tiny_threshold = max(1, int(0.05 * average_count))
    if min_client_samples is not None:
        for client_id, count in enumerate(sample_counts):
            if count < min_client_samples:
                print(
                    f"WARNING: client_{client_id} has {count} samples, "
                    f"below --min_client_samples={min_client_samples}."
                )
    for client_id, count in enumerate(sample_counts):
        if count == 0:
            print(f"WARNING: client_{client_id} received zero samples.")
        elif min_client_samples is None and count < tiny_threshold:
            print(
                f"WARNING: client_{client_id} has {count} samples, "
                f"below 5% of the average client size ({average_count:.1f})."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create federated client train partitions.")
    parser.add_argument("--train_csv", required=True, help="Path to data/splits/train.csv.")
    parser.add_argument(
        "--num_clients",
        type=int,
        default=DEFAULT_NUM_CLIENTS,
        help="Number of federated clients.",
    )
    parser.add_argument("--output_dir", required=True, help="Base output directory for client CSVs.")
    parser.add_argument(
        "--partition_type",
        choices=["iid", "mild_class_skew", "dirichlet"],
        help=(
            "Partition mode to create. Omit for backward-compatible behavior that "
            "creates both clients_iid and clients_noniid."
        ),
    )
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3, help="Dirichlet alpha for label-skew split.")
    parser.add_argument(
        "--min_client_samples",
        type=int,
        help="Optional warning threshold for very small client partitions.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducible partitions.")
    args = parser.parse_args()

    if args.num_clients < 2:
        raise ValueError("num_clients must be at least 2.")

    train_csv = resolve_path(args.train_csv)
    output_dir = resolve_path(args.output_dir)
    train_df = pd.read_csv(train_csv)

    required_columns = {"image_path", "label", "class_name"}
    missing_columns = required_columns - set(train_df.columns)
    if missing_columns:
        raise ValueError(f"Train CSV is missing required columns: {sorted(missing_columns)}")

    if args.partition_type is None:
        iid_partitions = create_iid_partitions(train_df, args.num_clients, seed=args.seed)
        noniid_partitions = create_noniid_partitions(train_df, args.num_clients, seed=args.seed)

        iid_paths = save_client_partitions(iid_partitions, output_dir, "clients_iid")
        noniid_paths = save_client_partitions(noniid_partitions, output_dir, "clients_noniid")

        print_partition_summary(iid_partitions, "IID Client Partition Summary")
        print_partition_summary(noniid_partitions, "Mild Class-Skew Client Partition Summary")

        print("\nSaved IID client CSV files:")
        for path in iid_paths:
            print(f"  {path}")

        print("\nSaved mild class-skew client CSV files:")
        for path in noniid_paths:
            print(f"  {path}")
        return

    if args.partition_type == "iid":
        partitions = create_iid_partitions(train_df, args.num_clients, seed=args.seed)
        partition_name = "clients_iid"
        title = "IID Client Partition Summary"
    elif args.partition_type == "mild_class_skew":
        partitions = create_noniid_partitions(train_df, args.num_clients, seed=args.seed)
        partition_name = "clients_noniid"
        title = "Mild Class-Skew Client Partition Summary"
    else:
        partitions = create_dirichlet_partitions(
            train_df,
            args.num_clients,
            alpha=args.dirichlet_alpha,
            seed=args.seed,
        )
        partition_name = dirichlet_partition_name(args.dirichlet_alpha)
        title = f"Dirichlet alpha={args.dirichlet_alpha:g} Client Partition Summary"

    warn_on_small_clients(partitions, args.min_client_samples)
    saved_paths = save_client_partitions(partitions, output_dir, partition_name)
    print_partition_summary(partitions, title)

    print(f"\nSaved {args.partition_type} client CSV files:")
    for path in saved_paths:
        print(f"  {path}")

    if args.partition_type == "dirichlet":
        table_path, summary_path, summary = save_partition_reports(
            partitions,
            saved_paths,
            output_dir,
            partition_name,
            train_df,
        )
        print("\nDirichlet Partition Report")
        print("==========================")
        print(f"Client sample counts: {summary['client_sample_counts']}")
        print(f"Classes per client: {summary['classes_per_client']}")
        print(f"Zero-class entries: {summary['zero_class_entries_per_client']}")
        print(f"Covers train exactly once: {summary['covers_train_exactly_once']}")
        print(f"Val images in clients: {summary['val_images_in_clients']}")
        print(f"Test images in clients: {summary['test_images_in_clients']}")
        print(f"Saved distribution table: {table_path}")
        print(f"Saved partition summary: {summary_path}")


if __name__ == "__main__":
    main()
