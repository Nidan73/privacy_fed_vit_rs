from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from utils import resolve_path


REQUIRED_FIELDS = {
    "experiment_id",
    "experiment_name",
    "dataset",
    "train_csv",
    "val_csv",
    "test_csv",
    "model_name",
    "pretrained",
    "fl_method",
    "privacy",
    "num_clients",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "seed",
    "output_dir",
    "metrics_path",
}


def load_ablation_plan(config_path: str | Path) -> dict[str, dict[str, Any]]:
    config_file = resolve_path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    defaults = raw_config.get("defaults", {})
    experiments = raw_config.get("experiments", {})
    if not experiments:
        raise ValueError("No experiments found under the 'experiments' key.")

    merged = {}
    for experiment_id, experiment_config in experiments.items():
        config = {**defaults, **experiment_config}
        config.setdefault("experiment_id", experiment_id)
        validate_experiment(config)
        merged[experiment_id] = config
    return merged


def validate_experiment(config: dict[str, Any]) -> None:
    missing = [field for field in sorted(REQUIRED_FIELDS) if field not in config]
    if missing:
        raise ValueError(f"{config.get('experiment_id', '<unknown>')} missing fields: {missing}")

    if config["experiment_id"] == "":
        raise ValueError("experiment_id must not be empty.")
    if config["fl_method"] not in {"none", "fedavg"}:
        raise ValueError(f"Unsupported fl_method: {config['fl_method']}")
    if config["privacy"] not in {"none", "ckks_secure_aggregation", "selected_layer_ckks"}:
        raise ValueError(f"Unsupported privacy setting: {config['privacy']}")
    if config["fl_method"] == "fedavg" and not config.get("client_split_dir"):
        raise ValueError(f"{config['experiment_id']} requires client_split_dir.")
    if int(config["num_clients"]) < 1:
        raise ValueError("num_clients must be at least 1.")
    if int(config["epochs"]) < 1:
        raise ValueError("epochs must be at least 1.")
    if int(config["batch_size"]) < 1:
        raise ValueError("batch_size must be at least 1.")


def get_experiment(config_path: str | Path, experiment_id: str) -> dict[str, Any]:
    experiments = load_ablation_plan(config_path)
    if experiment_id not in experiments:
        valid_ids = ", ".join(experiments)
        raise KeyError(f"Unknown experiment_id '{experiment_id}'. Valid IDs: {valid_ids}")
    return experiments[experiment_id]


def format_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def print_registry(experiments: dict[str, dict[str, Any]]) -> None:
    print("\nExperiment IDs")
    print("==============")
    for experiment_id in experiments:
        print(f"- {experiment_id}")

    columns = [
        ("experiment_id", 34),
        ("model_name", 22),
        ("fl_method", 10),
        ("privacy", 25),
        ("clients", 7),
        ("epochs", 6),
        ("batch", 5),
    ]
    header = " | ".join(name.ljust(width) for name, width in columns)
    print("\nAblation Registry")
    print("=================")
    print(header)
    print("-" * len(header))

    for config in experiments.values():
        row_values = {
            "experiment_id": config["experiment_id"],
            "model_name": config["model_name"],
            "fl_method": config["fl_method"],
            "privacy": config["privacy"],
            "clients": config["num_clients"],
            "epochs": config["epochs"],
            "batch": config["batch_size"],
        }
        print(
            " | ".join(
                format_bool(row_values[name])[:width].ljust(width)
                for name, width in columns
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print and validate ablation experiments.")
    parser.add_argument("--config", required=True, help="Path to ablation_plan.yaml.")
    args = parser.parse_args()

    experiments = load_ablation_plan(args.config)
    print_registry(experiments)


if __name__ == "__main__":
    main()
