from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_NUM_CLIENTS, RANDOM_SEED
from utils import resolve_path


def split_dataframe(df: pd.DataFrame, num_parts: int) -> list[pd.DataFrame]:
    split_indices = np.array_split(np.arange(len(df)), num_parts)
    return [df.iloc[indices].copy() for indices in split_indices]


def create_iid_partitions(train_df: pd.DataFrame, num_clients: int) -> list[pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    client_parts = [[] for _ in range(num_clients)]

    for _, class_df in train_df.groupby("label", sort=True):
        shuffled = class_df.sample(frac=1, random_state=int(rng.integers(0, 1_000_000)))
        for client_id, part in enumerate(split_dataframe(shuffled, num_clients)):
            client_parts[client_id].append(part)

    return [
        pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        for parts in client_parts
    ]


def create_noniid_partitions(train_df: pd.DataFrame, num_clients: int) -> list[pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
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

    return [
        pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        for parts in client_parts
    ]


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
        counts_text = ", ".join(f"{name}:{count}" for name, count in class_counts.items())
        print(f"client_{client_id}: total={len(client_df)} | {counts_text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create IID and non-IID client train partitions.")
    parser.add_argument("--train_csv", required=True, help="Path to data/splits/train.csv.")
    parser.add_argument(
        "--num_clients",
        type=int,
        default=DEFAULT_NUM_CLIENTS,
        help="Number of federated clients.",
    )
    parser.add_argument("--output_dir", required=True, help="Base output directory for client CSVs.")
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

    iid_partitions = create_iid_partitions(train_df, args.num_clients)
    noniid_partitions = create_noniid_partitions(train_df, args.num_clients)

    iid_paths = save_client_partitions(iid_partitions, output_dir, "clients_iid")
    noniid_paths = save_client_partitions(noniid_partitions, output_dir, "clients_noniid")

    print_partition_summary(iid_partitions, "IID Client Partition Summary")
    print_partition_summary(noniid_partitions, "Non-IID Client Partition Summary")

    print("\nSaved IID client CSV files:")
    for path in iid_paths:
        print(f"  {path}")

    print("\nSaved non-IID client CSV files:")
    for path in noniid_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
