from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_COLUMNS = [
    "experiment_id",
    "base_experiment_id",
    "seed",
    "test_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "best_val_accuracy",
    "best_epoch_or_round",
    "global_rounds",
    "local_epochs",
    "privacy",
    "selected_ckks_num_parameters",
    "ckks_num_chunks",
    "ckks_encryption_time_total",
    "ckks_aggregation_time_total",
    "ckks_decryption_time_total",
    "total_training_time_seconds",
]

KNOWN_BASE_EXPERIMENTS = [
    "A0_centralized_vit_base",
    "A1_fedavg_iid_vit_base",
    "A2_fedavg_noniid_vit_base",
    "A3_fedavg_ckks_iid_vit_base",
    "A4_fedavg_ckks_noniid_vit_base",
]


def normalize_pattern(pattern: str | None) -> str:
    if not pattern:
        return "*_metrics.json"
    if any(char in pattern for char in "*?[]"):
        return pattern
    return f"*{pattern}*"


def infer_base_experiment_id(metrics: dict[str, Any]) -> str | None:
    if metrics.get("base_experiment_id"):
        return str(metrics["base_experiment_id"])

    experiment_id = str(metrics.get("experiment_id", ""))
    for base_id in KNOWN_BASE_EXPERIMENTS:
        if experiment_id == base_id or experiment_id.startswith(f"{base_id}_"):
            return base_id
    return experiment_id or None


def infer_seed(metrics: dict[str, Any]) -> int | None:
    if metrics.get("seed") is not None:
        return int(metrics["seed"])

    experiment_id = str(metrics.get("experiment_id", ""))
    match = re.search(r"(?:^|_)s(?P<seed>\d+)(?:_|$)", experiment_id)
    if match:
        return int(match.group("seed"))
    return None


def best_epoch_or_round(metrics: dict[str, Any]) -> Any:
    if metrics.get("best_epoch") is not None:
        return metrics["best_epoch"]
    return metrics.get("best_round")


def nested_test_metric(metrics: dict[str, Any], key: str) -> Any:
    test_metrics = metrics.get("test_metrics")
    if isinstance(test_metrics, dict):
        return test_metrics.get(key)
    return None


def first_present(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if metrics.get(key) is not None:
            return metrics[key]
    return None


def test_metric_value(metrics: dict[str, Any], root_key: str, nested_key: str) -> Any:
    value = metrics.get(root_key)
    if value is not None:
        return value
    return nested_test_metric(metrics, nested_key)


def load_metric_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        print(f"Skipping unreadable JSON {path}: {exc}")
        return None

    if not isinstance(payload, dict):
        print(f"Skipping non-object JSON {path}")
        return None
    if not payload.get("experiment_id"):
        print(f"Skipping non-experiment metrics file {path.name}")
        return None
    return payload


def metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": metrics.get("experiment_id"),
        "base_experiment_id": infer_base_experiment_id(metrics),
        "seed": infer_seed(metrics),
        "test_accuracy": test_metric_value(metrics, "test_accuracy", "accuracy"),
        "test_macro_f1": test_metric_value(metrics, "test_macro_f1", "macro_f1"),
        "test_weighted_f1": test_metric_value(metrics, "test_weighted_f1", "weighted_f1"),
        "best_val_accuracy": metrics.get("best_val_accuracy"),
        "best_epoch_or_round": best_epoch_or_round(metrics),
        "global_rounds": metrics.get("global_rounds"),
        "local_epochs": metrics.get("local_epochs"),
        "privacy": metrics.get("privacy"),
        "selected_ckks_num_parameters": metrics.get("selected_ckks_num_parameters"),
        "ckks_num_chunks": metrics.get("ckks_num_chunks"),
        "ckks_encryption_time_total": metrics.get("ckks_encryption_time_total"),
        "ckks_aggregation_time_total": metrics.get("ckks_aggregation_time_total"),
        "ckks_decryption_time_total": metrics.get("ckks_decryption_time_total"),
        "total_training_time_seconds": first_present(
            metrics,
            "total_training_time_seconds",
            "training_time_seconds",
            "training_time_sec",
        ),
    }


def collect_metric_rows(metrics_dir: Path, pattern: str | None = None) -> list[dict[str, Any]]:
    glob_pattern = normalize_pattern(pattern)
    metric_paths = sorted(metrics_dir.glob(glob_pattern))

    rows: list[dict[str, Any]] = []
    for path in metric_paths:
        metrics = load_metric_json(path)
        if metrics is None:
            continue
        rows.append(metric_row(metrics))
    return rows


def write_summary(rows: list[dict[str, Any]], output_csv: Path) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment metric JSON files into one CSV.")
    parser.add_argument("--metrics_dir", default="results/metrics", help="Directory containing metric JSON files.")
    parser.add_argument("--pattern", help="Optional glob pattern or substring for metric JSON files.")
    parser.add_argument(
        "--output_csv",
        default="results/tables/experiment_summary.csv",
        help="Path for the aggregated summary CSV.",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_csv = Path(args.output_csv)
    rows = collect_metric_rows(metrics_dir, args.pattern)
    df = write_summary(rows, output_csv)

    print(f"Scanned metrics_dir: {metrics_dir}")
    print(f"Rows written: {len(df)}")
    print(f"Output CSV: {output_csv}")
    if not df.empty:
        print(df[["experiment_id", "seed", "test_accuracy", "test_macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
