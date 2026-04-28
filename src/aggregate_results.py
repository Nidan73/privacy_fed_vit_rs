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
    "output_suffix",
    "comparable_group_id",
    "model_name",
    "fl_method",
    "privacy",
    "global_rounds",
    "local_epochs",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "selected_ckks_num_parameters",
    "ckks_num_chunks",
    "test_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "best_val_accuracy",
    "best_epoch",
    "best_round",
]

KNOWN_BASE_EXPERIMENTS = [
    "A0_centralized_vit_base",
    "A1_fedavg_iid_vit_base",
    "A2_fedavg_noniid_vit_base",
    "A3_fedavg_ckks_iid_vit_base",
    "A4_fedavg_ckks_noniid_vit_base",
]

DEFAULT_EPOCHS_BY_BASE = {
    "A0_centralized_vit_base": 5,
}


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


def infer_output_suffix(metrics: dict[str, Any], base_experiment_id: str | None) -> str | None:
    if metrics.get("output_suffix"):
        return str(metrics["output_suffix"])

    experiment_id = str(metrics.get("experiment_id", ""))
    if base_experiment_id and experiment_id.startswith(f"{base_experiment_id}_"):
        return experiment_id[len(base_experiment_id) + 1 :]
    return None


def infer_fl_method(metrics: dict[str, Any], base_experiment_id: str | None) -> str | None:
    if metrics.get("fl_method"):
        return str(metrics["fl_method"])
    if base_experiment_id == "A0_centralized_vit_base":
        return "none"
    if base_experiment_id and base_experiment_id.startswith(("A1_", "A2_", "A3_", "A4_")):
        return "fedavg"
    return None


def infer_privacy(metrics: dict[str, Any], base_experiment_id: str | None, fl_method: str | None) -> str | None:
    if metrics.get("privacy"):
        return str(metrics["privacy"])
    if fl_method == "none":
        return "none"
    if base_experiment_id and base_experiment_id.startswith(("A1_", "A2_")):
        return "none"
    return None


def infer_int_from_experiment_id(metrics: dict[str, Any], pattern: str) -> int | None:
    experiment_id = str(metrics.get("experiment_id", ""))
    match = re.search(pattern, experiment_id)
    if match:
        return int(match.group(1))
    return None


def infer_epochs(metrics: dict[str, Any], base_experiment_id: str | None) -> int | None:
    value = first_present(metrics, "epochs", "actual_epochs_ran")
    if value is not None:
        return int(value)

    value = infer_int_from_experiment_id(metrics, r"_(\d+)ep(?:_|$)")
    if value is not None:
        return value

    if base_experiment_id in DEFAULT_EPOCHS_BY_BASE and metrics.get("experiment_id") == base_experiment_id:
        return DEFAULT_EPOCHS_BY_BASE[base_experiment_id]
    return None


def infer_global_rounds(metrics: dict[str, Any]) -> int | None:
    value = first_present(metrics, "global_rounds", "actual_global_rounds")
    if value is not None:
        return int(value)
    return infer_int_from_experiment_id(metrics, r"_(\d+)r(?:\d+e)?(?:_|$)")


def infer_local_epochs(metrics: dict[str, Any]) -> int | None:
    value = metrics.get("local_epochs")
    if value is not None:
        return int(value)
    return infer_int_from_experiment_id(metrics, r"_(?:\d+)r(\d+)e(?:_|$)")


def client_split_short_name(client_split_dir: Any) -> str | None:
    if client_split_dir is None:
        return None

    normalized = str(client_split_dir).replace("\\", "/").rstrip("/")
    name = normalized.split("/")[-1]
    if name == "clients_iid":
        return "iid"
    if name == "clients_noniid":
        return "noniid"
    return sanitize_group_part(name)


def format_value(value: Any, missing: str = "unknown") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return missing
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def sanitize_group_part(value: Any) -> str:
    text = format_value(value)
    text = text.replace("\\", "/").strip("/")
    text = text.replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def comparable_group_id(row: dict[str, Any], client_split_dir: Any) -> str:
    base_id = sanitize_group_part(row["base_experiment_id"])
    model_name = sanitize_group_part(row["model_name"])
    batch_size = format_value(row["batch_size"])
    learning_rate = format_value(row["learning_rate"])
    weight_decay = format_value(row["weight_decay"])

    if row["fl_method"] == "none":
        return (
            f"{base_id}_epochs{format_value(row['epochs'])}_{model_name}"
            f"_bs{batch_size}_lr{learning_rate}_wd{weight_decay}"
        )

    split_name = client_split_short_name(client_split_dir)
    if row["privacy"] in {None, "none"}:
        return (
            f"{base_id}_r{format_value(row['global_rounds'])}_e{format_value(row['local_epochs'])}"
            f"_{model_name}_{format_value(split_name)}"
            f"_bs{batch_size}_lr{learning_rate}_wd{weight_decay}"
        )

    return (
        f"{base_id}_r{format_value(row['global_rounds'])}_e{format_value(row['local_epochs'])}"
        f"_{model_name}_{format_value(split_name)}_{sanitize_group_part(row['privacy'])}"
        f"_chunks{format_value(row['ckks_num_chunks'])}"
        f"_params{format_value(row['selected_ckks_num_parameters'])}"
        f"_bs{batch_size}_lr{learning_rate}_wd{weight_decay}"
    )


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
    base_experiment_id = infer_base_experiment_id(metrics)
    fl_method = infer_fl_method(metrics, base_experiment_id)
    privacy = infer_privacy(metrics, base_experiment_id, fl_method)
    row = {
        "experiment_id": metrics.get("experiment_id"),
        "base_experiment_id": base_experiment_id,
        "seed": infer_seed(metrics),
        "output_suffix": infer_output_suffix(metrics, base_experiment_id),
        "comparable_group_id": None,
        "model_name": metrics.get("model_name"),
        "fl_method": fl_method,
        "privacy": privacy,
        "global_rounds": infer_global_rounds(metrics),
        "local_epochs": infer_local_epochs(metrics),
        "epochs": infer_epochs(metrics, base_experiment_id),
        "batch_size": metrics.get("batch_size"),
        "learning_rate": metrics.get("learning_rate"),
        "weight_decay": metrics.get("weight_decay"),
        "selected_ckks_num_parameters": metrics.get("selected_ckks_num_parameters"),
        "ckks_num_chunks": metrics.get("ckks_num_chunks"),
        "test_accuracy": test_metric_value(metrics, "test_accuracy", "accuracy"),
        "test_macro_f1": test_metric_value(metrics, "test_macro_f1", "macro_f1"),
        "test_weighted_f1": test_metric_value(metrics, "test_weighted_f1", "weighted_f1"),
        "best_val_accuracy": metrics.get("best_val_accuracy"),
        "best_epoch": metrics.get("best_epoch"),
        "best_round": metrics.get("best_round"),
    }
    row["comparable_group_id"] = comparable_group_id(row, metrics.get("client_split_dir"))
    return row


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
        print(
            df[
                [
                    "experiment_id",
                    "seed",
                    "comparable_group_id",
                    "test_accuracy",
                    "test_macro_f1",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
