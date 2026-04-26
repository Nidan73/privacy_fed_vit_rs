from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from experiment_registry import get_experiment
from utils import resolve_path


OVERRIDE_FIELDS = {
    "epochs": "epochs",
    "global_rounds": "global_rounds",
    "local_epochs": "local_epochs",
    "batch_size": "batch_size",
    "lr": "learning_rate",
    "weight_decay": "weight_decay",
}


def print_selected_experiment(config: dict[str, Any], dry_run: bool, no_train: bool) -> None:
    print("\nSelected Experiment")
    print("===================")
    print(json.dumps(config, indent=2))
    print(f"\ndry_run: {dry_run}")
    print(f"no_train: {no_train}")


def sanitize_suffix(output_suffix: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", output_suffix):
        raise ValueError("output_suffix may contain only letters, numbers, underscores, and hyphens.")
    return output_suffix


def with_output_suffix(config: dict[str, Any], output_suffix: str) -> dict[str, Any]:
    suffix = sanitize_suffix(output_suffix)
    original_experiment_id = config["experiment_id"]
    experiment_id = f"{original_experiment_id}_{suffix}"

    output_dir = Path(config["output_dir"])
    config["base_experiment_id"] = original_experiment_id
    config["experiment_id"] = experiment_id
    config["experiment_name"] = f"{config['experiment_name']} ({suffix})"
    config["output_suffix"] = suffix
    config["output_dir"] = (output_dir.parent / experiment_id).as_posix()
    config["metrics_path"] = f"results/metrics/{experiment_id}_metrics.json"
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    changed_fields = []

    for arg_name, config_name in OVERRIDE_FIELDS.items():
        value = getattr(args, arg_name)
        if value is None:
            continue

        old_value = updated[config_name]
        updated[config_name] = value
        if str(old_value) != str(value):
            changed_fields.append(config_name)

    if changed_fields and not args.output_suffix:
        fields = ", ".join(changed_fields)
        raise SystemExit(
            "Refusing to run with training overrides without --output_suffix. "
            f"Changed fields: {fields}. "
            "Use --output_suffix to avoid overwriting previous experiment results."
        )

    if args.output_suffix:
        updated = with_output_suffix(updated, args.output_suffix)

    if args.resume_from:
        updated["resume_from"] = args.resume_from

    if changed_fields:
        print("\nApplied Overrides")
        print("=================")
        for field in changed_fields:
            print(f"{field}: {config[field]} -> {updated[field]}")
        print(f"experiment_id: {config['experiment_id']} -> {updated['experiment_id']}")
        print(f"output_dir: {updated['output_dir']}")
        print(f"metrics_path: {updated['metrics_path']}")

    if args.resume_from:
        print(f"resume_from: {args.resume_from}")

    return updated


def guard_existing_outputs(config: dict[str, Any], dry_run: bool, no_train: bool) -> None:
    if dry_run or no_train or config.get("resume_from"):
        return

    experiment_id = config["experiment_id"]
    possible_outputs = [
        resolve_path(config["metrics_path"]),
        resolve_path(config["output_dir"]) / "checkpoints" / "best.pt",
        resolve_path(config["output_dir"]) / "checkpoints" / "last.pt",
        resolve_path("results") / "logs" / f"{experiment_id}_epoch_log.csv",
    ]
    existing_outputs = [path for path in possible_outputs if path.exists()]

    if existing_outputs:
        existing_text = "\n".join(f"- {path}" for path in existing_outputs)
        raise FileExistsError(
            "Refusing to overwrite existing experiment outputs. "
            "Use --output_suffix for a new run.\n"
            f"{existing_text}"
        )


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
        from train_fedavg import run_training

        return run_training(config, dry_run=dry_run)

    if fl_method == "fedavg" and privacy == "ckks_secure_aggregation":
        print("\nTODO: FedAvg + CKKS secure aggregation pipeline is not implemented yet.")
        print("This branch will connect train_fedavg.py and secure_aggregation_ckks.py later.")
        return None

    raise ValueError(f"Unsupported experiment route: fl_method={fl_method}, privacy={privacy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch one configured ablation experiment.")
    parser.add_argument("--config", default="configs/ablation_plan.yaml", help="Path to config YAML.")
    parser.add_argument("--experiment_id", required=True, help="Experiment ID from the config.")
    parser.add_argument("--epochs", type=int, help="Override configured epoch count.")
    parser.add_argument("--global_rounds", type=int, help="Override configured FedAvg global rounds.")
    parser.add_argument("--local_epochs", type=int, help="Override configured FedAvg local epochs.")
    parser.add_argument("--batch_size", type=int, help="Override configured batch size.")
    parser.add_argument("--lr", type=float, help="Override configured learning rate.")
    parser.add_argument("--weight_decay", type=float, help="Override configured weight decay.")
    parser.add_argument(
        "--output_suffix",
        help="Append a suffix to experiment outputs, for example --output_suffix 10ep.",
    )
    parser.add_argument("--resume_from", help="Resume centralized training from a checkpoint.")
    parser.add_argument("--dry_run", action="store_true", help="Run a tiny pipeline check only.")
    parser.add_argument("--no_train", action="store_true", help="Validate and print config without training.")
    args = parser.parse_args()

    config = get_experiment(args.config, args.experiment_id)
    config = apply_overrides(config, args)
    guard_existing_outputs(config, dry_run=args.dry_run, no_train=args.no_train)
    run_experiment(config, dry_run=args.dry_run, no_train=args.no_train)


if __name__ == "__main__":
    main()
