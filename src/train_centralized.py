from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RemoteSensingCSVDataset
from model import get_model
from utils import resolve_path, set_seed


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def detect_num_classes(train_csv: str | Path) -> int:
    train_df = pd.read_csv(resolve_path(train_csv))
    return int(train_df["label"].nunique())


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    batch_size = int(config["batch_size"])
    device_is_cuda = torch.cuda.is_available()

    train_dataset = RemoteSensingCSVDataset(config["train_csv"], train=True)
    val_dataset = RemoteSensingCSVDataset(config["val_csv"], train=False)
    test_dataset = RemoteSensingCSVDataset(config["test_csv"], train=False)

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": device_is_cuda,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)
    return train_loader, val_loader, test_loader


def amp_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def build_grad_scaler(enabled: bool):
    if hasattr(torch, "amp"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def accuracy_from_counts(correct: int, total: int) -> float:
    return float(correct / total) if total else 0.0


def f1_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    if not labels:
        return {"macro_f1": 0.0, "weighted_f1": 0.0}
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    epoch: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for batch_index, (images, labels) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size

        progress.set_postfix(
            loss=f"{total_loss / max(total_samples, 1):.4f}",
            acc=f"{accuracy_from_counts(correct, total_samples):.4f}",
        )

    return {
        "loss": float(total_loss / max(total_samples, 1)),
        "accuracy": accuracy_from_counts(correct, total_samples),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    desc: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(loader, desc=desc, leave=False)
    for batch_index, (images, labels) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with amp_autocast(amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size

        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

        progress.set_postfix(
            loss=f"{total_loss / max(total_samples, 1):.4f}",
            acc=f"{accuracy_from_counts(correct, total_samples):.4f}",
        )

    metrics = {
        "loss": float(total_loss / max(total_samples, 1)),
        "accuracy": accuracy_from_counts(correct, total_samples),
    }
    metrics.update(f1_metrics(all_labels, all_predictions))
    return metrics


def artifact_paths(config: dict[str, Any], dry_run: bool) -> dict[str, Path]:
    experiment_id = config.get("experiment_id", "centralized_cli")
    if dry_run:
        output_dir = resolve_path(".cache") / "dry_runs" / experiment_id
        metrics_path = output_dir / "metrics.json"
        log_path = output_dir / "epoch_log.csv"
    else:
        output_dir = resolve_path(config["output_dir"])
        metrics_path = resolve_path(config["metrics_path"])
        log_path = resolve_path("results") / "logs" / f"{experiment_id}_epoch_log.csv"

    return {
        "output_dir": output_dir,
        "checkpoint_path": output_dir / "checkpoints" / "best.pt",
        "metrics_path": metrics_path,
        "log_path": log_path,
    }


def save_epoch_log(rows: list[dict[str, Any]], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_checkpoint(
    model: nn.Module,
    config: dict[str, Any],
    checkpoint_path: Path,
    epoch: int,
    val_metrics: dict[str, float],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_metrics": val_metrics,
            "config": config,
        },
        checkpoint_path,
    )


def print_epoch_summary(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    epoch_time: float,
    cuda_available: bool,
) -> None:
    gpu_memory = "n/a"
    if cuda_available:
        gpu_memory = f"{torch.cuda.max_memory_allocated() / (1024 ** 2):.1f} MB"

    print(
        f"Epoch {epoch} | "
        f"train_loss={train_metrics['loss']:.4f} | "
        f"train_acc={train_metrics['accuracy']:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"val_acc={val_metrics['accuracy']:.4f} | "
        f"gpu_mem={gpu_memory} | "
        f"time={epoch_time:.1f}s"
    )


def run_training(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    config = dict(config)
    set_seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))

    if dry_run:
        print("\nDry-run mode: running 1 epoch with 2 train batches and 2 validation batches.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    num_classes = detect_num_classes(config["train_csv"])

    print("\nCentralized Training Setup")
    print("==========================")
    print(f"experiment_id: {config.get('experiment_id', 'centralized_cli')}")
    print(f"model_name: {config['model_name']}")
    print(f"pretrained: {config['pretrained']}")
    print(f"num_classes: {num_classes}")
    print(f"device: {device}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"amp_enabled: {amp_enabled}")

    train_loader, val_loader, test_loader = build_dataloaders(config)
    print(f"train_batches: {len(train_loader)}")
    print(f"val_batches: {len(val_loader)}")
    print(f"test_batches: {len(test_loader)}")

    model = get_model(
        model_name=config["model_name"],
        num_classes=num_classes,
        pretrained=bool(config["pretrained"]),
    ).to(device)
    parameter_count = count_parameters(model)
    print(f"model_parameter_count: {parameter_count}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = build_grad_scaler(amp_enabled)

    paths = artifact_paths(config, dry_run=dry_run)
    num_epochs = 1 if dry_run else int(config["epochs"])
    max_train_batches = 2 if dry_run else None
    max_val_batches = 2 if dry_run else None
    best_val_accuracy = -1.0
    best_epoch = 0
    epoch_rows: list[dict[str, Any]] = []
    training_start = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            epoch=epoch,
            max_batches=max_train_batches,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            desc=f"Epoch {epoch} val",
            max_batches=max_val_batches,
        )
        epoch_time = time.perf_counter() - epoch_start
        print_epoch_summary(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            epoch_time=epoch_time,
            cuda_available=torch.cuda.is_available(),
        )

        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "epoch_time_sec": epoch_time,
            }
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            save_checkpoint(model, config, paths["checkpoint_path"], epoch, val_metrics)

    training_time = time.perf_counter() - training_start
    test_metrics: dict[str, float] | None = None

    if dry_run:
        print("Dry-run mode: test evaluation skipped.")
    else:
        checkpoint = torch.load(paths["checkpoint_path"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            desc="Test",
        )

    save_epoch_log(epoch_rows, paths["log_path"])

    metrics_payload = {
        "experiment_id": config.get("experiment_id", "centralized_cli"),
        "dry_run": dry_run,
        "model_name": config["model_name"],
        "dataset": config["dataset"],
        "num_classes": num_classes,
        "model_parameter_count": parameter_count,
        "cuda_available": torch.cuda.is_available(),
        "amp_enabled": amp_enabled,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "training_time_sec": training_time,
        "epoch_log_path": str(paths["log_path"]),
        "checkpoint_path": str(paths["checkpoint_path"]),
        "test_metrics": test_metrics,
    }
    save_json(metrics_payload, paths["metrics_path"])

    print("\nCentralized run completed.")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_accuracy: {best_val_accuracy:.4f}")
    print(f"metrics_path: {paths['metrics_path']}")
    print(f"epoch_log_path: {paths['log_path']}")
    print(f"checkpoint_path: {paths['checkpoint_path']}")

    return metrics_payload


def cli_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment_id": args.experiment_id,
        "experiment_name": args.experiment_id,
        "dataset": args.dataset,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "metrics_path": args.metrics_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train centralized ViT-Base baseline.")
    parser.add_argument("--experiment_id", default="centralized_cli")
    parser.add_argument("--dataset", default="UCMerced Land Use")
    parser.add_argument("--train_csv", default="data/splits/train.csv")
    parser.add_argument("--val_csv", default="data/splits/val.csv")
    parser.add_argument("--test_csv", default="data/splits/test.csv")
    parser.add_argument("--model_name", default="vit_base_patch16_224")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="experiments/centralized/centralized_cli")
    parser.add_argument("--metrics_path", default="results/metrics/centralized_cli_metrics.json")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_train", action="store_true")
    args = parser.parse_args()

    config = cli_config(args)
    if args.no_train:
        print(json.dumps(config, indent=2))
        print("\nNo training requested.")
        return

    run_training(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
