from __future__ import annotations

import argparse
import json
from typing import Any

from experiment_registry import get_experiment


def print_selected_experiment(config: dict[str, Any], dry_run: bool, no_train: bool) -> None:
    print("\nSelected Experiment")
    print("===================")
    print(json.dumps(config, indent=2))
    print(f"\ndry_run: {dry_run}")
    print(f"no_train: {no_train}")


def run_experiment(config: dict[str, Any], dry_run: bool = False, no_train: bool = False) -> dict[str, Any] | None:
    print_selected_experiment(config, dry_run=dry_run, no_train=no_train)

    fl_method = config["fl_method"]
    privacy = config["privacy"]

    if no_train:
        print("\nNo training requested. Configuration validation completed.")
        return None

    if fl_method == "none":
        from train_centralized import run_training

        return run_training(config, dry_run=dry_run)

    if fl_method == "fedavg" and privacy == "none":
        print("\nTODO: FedAvg training pipeline is not implemented yet.")
        print("This branch will use train_fedavg.py after centralized training is validated.")
        return None

    if fl_method == "fedavg" and privacy == "ckks_secure_aggregation":
        print("\nTODO: FedAvg + CKKS secure aggregation pipeline is not implemented yet.")
        print("This branch will connect train_fedavg.py and secure_aggregation_ckks.py later.")
        return None

    raise ValueError(f"Unsupported experiment route: fl_method={fl_method}, privacy={privacy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch one configured ablation experiment.")
    parser.add_argument("--config", default="configs/ablation_plan.yaml", help="Path to config YAML.")
    parser.add_argument("--experiment_id", required=True, help="Experiment ID from the config.")
    parser.add_argument("--dry_run", action="store_true", help="Run a tiny pipeline check only.")
    parser.add_argument("--no_train", action="store_true", help="Validate and print config without training.")
    args = parser.parse_args()

    config = get_experiment(args.config, args.experiment_id)
    run_experiment(config, dry_run=args.dry_run, no_train=args.no_train)


if __name__ == "__main__":
    main()
