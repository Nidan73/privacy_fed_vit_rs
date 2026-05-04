from __future__ import annotations

import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SUPPORTED_AUG_POLICIES, RemoteSensingCSVDataset
from model import get_model
from secure_aggregation_ckks import (
    create_ckks_context,
    get_ckks_slot_count,
    mixed_plaintext_ckks_fedavg,
    plaintext_aggregate_state_dicts,
    resolve_selected_ckks_keys,
)
from utils import resolve_path, set_seed


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


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


def detect_num_classes(*csv_paths: str | Path) -> int:
    labels: set[int] = set()
    for csv_path in csv_paths:
        df = pd.read_csv(resolve_path(csv_path))
        labels.update(int(label) for label in df["label"].unique())
    return len(labels)


def class_names_from_csv(csv_path: str | Path) -> list[str]:
    df = pd.read_csv(resolve_path(csv_path))
    label_names = (
        df[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
        .set_index("label")["class_name"]
        .to_dict()
    )
    return [label_names[label] for label in sorted(label_names)]


def load_client_csvs(client_split_dir: str | Path, num_clients: int) -> list[Path]:
    split_dir = resolve_path(client_split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Client split directory not found: {split_dir}")

    client_paths = [split_dir / f"client_{client_id}.csv" for client_id in range(num_clients)]
    missing = [path for path in client_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing client CSV files: {missing}")
    return client_paths


def build_eval_loader(csv_path: str | Path, batch_size: int, num_workers: int, device_is_cuda: bool) -> DataLoader:
    dataset = RemoteSensingCSVDataset(csv_path, train=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_is_cuda,
    )


def build_client_loader(
    csv_path: str | Path,
    batch_size: int,
    num_workers: int,
    device_is_cuda: bool,
    aug_policy: str,
) -> DataLoader:
    dataset = RemoteSensingCSVDataset(csv_path, train=True, aug_policy=aug_policy)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device_is_cuda,
    )


def round_learning_rate(
    base_learning_rate: float,
    scheduler_name: str,
    round_index: int,
    total_rounds: int,
) -> float:
    if scheduler_name == "none":
        return base_learning_rate
    if scheduler_name == "cosine":
        progress = float(round_index - 1) / max(float(total_rounds), 1.0)
        return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError("scheduler must be one of: none, cosine")


def train_client(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    local_epochs: int,
    round_index: int,
    client_id: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    batches_seen = 0

    for local_epoch in range(1, local_epochs + 1):
        progress = tqdm(
            loader,
            desc=f"Round {round_index} client {client_id} epoch {local_epoch}",
            leave=False,
        )
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
            batches_seen += 1

            progress.set_postfix(
                loss=f"{total_loss / max(total_samples, 1):.4f}",
                acc=f"{accuracy_from_counts(correct, total_samples):.4f}",
            )

    return {
        "loss": float(total_loss / max(total_samples, 1)),
        "accuracy": accuracy_from_counts(correct, total_samples),
        "batches_seen": float(batches_seen),
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
) -> dict[str, Any]:
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

    metrics: dict[str, Any] = {
        "loss": float(total_loss / max(total_samples, 1)),
        "accuracy": accuracy_from_counts(correct, total_samples),
        "labels": all_labels,
        "predictions": all_predictions,
    }
    metrics.update(f1_metrics(all_labels, all_predictions))
    return metrics


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def aggregate_state_dicts(
    client_states: list[dict[str, torch.Tensor]],
    client_sample_counts: list[int],
) -> dict[str, torch.Tensor]:
    return plaintext_aggregate_state_dicts(client_states, client_sample_counts)


def artifact_paths(config: dict[str, Any], dry_run: bool) -> dict[str, Path]:
    experiment_id = config["experiment_id"]
    if dry_run:
        base_dir = resolve_path(".cache") / "dry_runs" / experiment_id
        metrics_dir = base_dir
        logs_dir = base_dir
        output_dir = base_dir
    else:
        output_dir = resolve_path(config["output_dir"])
        metrics_dir = resolve_path("results") / "metrics"
        logs_dir = resolve_path("results") / "logs"

    return {
        "output_dir": output_dir,
        "best_checkpoint_path": output_dir / "checkpoints" / "best.pt",
        "last_checkpoint_path": output_dir / "checkpoints" / "last.pt",
        "metrics_path": metrics_dir / f"{experiment_id}_metrics.json"
        if dry_run
        else resolve_path(config["metrics_path"]),
        "round_log_path": logs_dir / f"{experiment_id}_round_log.csv",
        "client_log_path": logs_dir / f"{experiment_id}_client_log.csv",
        "classification_report_path": metrics_dir / f"{experiment_id}_classification_report.csv",
        "confusion_matrix_path": metrics_dir / f"{experiment_id}_confusion_matrix.csv",
    }


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_best_checkpoint(
    model: nn.Module,
    config: dict[str, Any],
    path: Path,
    round_index: int,
    best_val_accuracy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "round": round_index,
            "model_state_dict": model.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "best_round": round_index,
            "config": config,
        },
        path,
    )


def save_last_checkpoint(
    model: nn.Module,
    config: dict[str, Any],
    path: Path,
    round_index: int,
    best_val_accuracy: float,
    best_round: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "round": round_index,
            "model_state_dict": model.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "best_round": best_round,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def maybe_resume_global_model(
    config: dict[str, Any],
    model: nn.Module,
    device: torch.device,
) -> tuple[int, float, int]:
    resume_from = config.get("resume_from")
    if not resume_from:
        return 1, -1.0, 0

    checkpoint_path = resolve_path(resume_from)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"FedAvg resume checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint_round = int(checkpoint.get("round", checkpoint.get("epoch", 0)))
    best_val_accuracy = float(checkpoint.get("best_val_accuracy", -1.0))
    best_round = int(checkpoint.get("best_round", checkpoint_round))

    print("\nResumed FedAvg")
    print("==============")
    print(f"resume_from: {checkpoint_path}")
    print(f"checkpoint_round: {checkpoint_round}")
    print(f"next_round: {checkpoint_round + 1}")
    print(f"best_round: {best_round}")
    print(f"best_val_accuracy: {best_val_accuracy:.4f}")
    return checkpoint_round + 1, best_val_accuracy, best_round


def gpu_memory_text() -> str:
    if not torch.cuda.is_available():
        return "n/a"
    return f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.2f}GB"


def save_test_reports(
    test_metrics: dict[str, Any],
    class_names: list[str],
    classification_report_path: Path,
    confusion_matrix_path: Path,
) -> None:
    labels = test_metrics["labels"]
    predictions = test_metrics["predictions"]
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_rows = []
    for label_name, metrics in report.items():
        if isinstance(metrics, dict):
            row = {"class_name": label_name, **metrics}
        else:
            row = {"class_name": label_name, "score": metrics}
        report_rows.append(row)

    classification_report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(classification_report_path, index=False)

    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    matrix_df = pd.DataFrame(matrix, index=class_names, columns=class_names)
    matrix_df.to_csv(confusion_matrix_path)


def run_training(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    config = dict(config)
    set_seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))

    experiment_id = config["experiment_id"]
    num_clients = int(config.get("num_clients", 3))
    global_rounds = int(config.get("global_rounds", config.get("epochs", 5)))
    local_epochs = int(config.get("local_epochs", 1))
    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 0))
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    aug_policy = str(config.get("aug_policy", "basic"))
    label_smoothing = float(config.get("label_smoothing", 0.0))
    scheduler_name = str(config.get("scheduler", "none"))
    privacy = config.get("privacy", "none")
    if aug_policy not in SUPPORTED_AUG_POLICIES:
        raise ValueError(f"aug_policy must be one of: {sorted(SUPPORTED_AUG_POLICIES)}")
    if label_smoothing < 0.0 or label_smoothing >= 1.0:
        raise ValueError("label_smoothing must be in the range [0.0, 1.0).")
    if scheduler_name not in {"none", "cosine"}:
        raise ValueError("scheduler must be one of: none, cosine")
    if privacy not in {"none", "selected_layer_ckks"}:
        raise ValueError(f"Unsupported FedAvg privacy mode: {privacy}")

    if dry_run:
        print("\nFedAvg dry-run mode: 1 round, 1 local epoch, 2 client batches, 2 validation batches.")
        global_rounds = 1
        local_epochs = 1

    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = device.type == "cuda"
    device_is_cuda = device.type == "cuda"

    client_csvs = load_client_csvs(config["client_split_dir"], num_clients)
    client_sample_counts = [len(pd.read_csv(path)) for path in client_csvs]
    total_client_samples = sum(client_sample_counts)
    num_classes = detect_num_classes(config["val_csv"], config["test_csv"], *client_csvs)

    print("\nFedAvg Training Setup")
    print("=====================")
    print(f"experiment_id: {experiment_id}")
    print(f"model_name: {config['model_name']}")
    print(f"pretrained: {config['pretrained']}")
    print(f"num_classes: {num_classes}")
    print(f"num_clients: {num_clients}")
    print(f"client_split_dir: {config['client_split_dir']}")
    print(f"client_sample_counts: {client_sample_counts}")
    print(f"global_rounds: {global_rounds}")
    print(f"local_epochs: {local_epochs}")
    print(f"privacy: {privacy}")
    print(f"device: {device}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"amp_enabled: {amp_enabled}")
    print(f"aug_policy: {aug_policy}")
    print(f"label_smoothing: {label_smoothing}")
    print(f"scheduler: {scheduler_name}")

    val_loader = build_eval_loader(config["val_csv"], batch_size, num_workers, device_is_cuda)
    test_loader = build_eval_loader(config["test_csv"], batch_size, num_workers, device_is_cuda)
    train_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    global_model = get_model(
        model_name=config["model_name"],
        num_classes=num_classes,
        pretrained=bool(config["pretrained"]),
    ).to(device)
    parameter_count = count_parameters(global_model)
    print(f"model_parameter_count: {parameter_count}")

    ckks_context = None
    selected_ckks_keys: list[str] = []
    selected_ckks_num_parameters = 0
    ckks_poly_modulus_degree = int(config.get("ckks_poly_modulus_degree", 8192))
    ckks_coeff_mod_bit_sizes = config.get("ckks_coeff_mod_bit_sizes", [60, 40, 40, 60])
    ckks_global_scale_exponent = int(config.get("ckks_global_scale_exponent", 40))
    ckks_global_scale = float(2 ** ckks_global_scale_exponent)
    ckks_chunk_size = int(
        config.get("ckks_chunk_size", get_ckks_slot_count(ckks_poly_modulus_degree))
    )
    ckks_num_chunks = 0

    if privacy == "selected_layer_ckks":
        selected_ckks_keys = resolve_selected_ckks_keys(
            global_model.state_dict(),
            config.get("selected_ckks_keys"),
        )
        selected_ckks_num_parameters = int(
            sum(global_model.state_dict()[key].numel() for key in selected_ckks_keys)
        )
        ckks_context = create_ckks_context(
            poly_modulus_degree=ckks_poly_modulus_degree,
            coeff_mod_bit_sizes=ckks_coeff_mod_bit_sizes,
            global_scale=ckks_global_scale,
        )
        print("selected_layer_ckks: enabled")
        print(f"selected_ckks_keys: {selected_ckks_keys}")
        print(f"selected_ckks_num_parameters: {selected_ckks_num_parameters}")
        print(f"ckks_poly_modulus_degree: {ckks_poly_modulus_degree}")
        print(f"ckks_coeff_mod_bit_sizes: {ckks_coeff_mod_bit_sizes}")
        print(f"ckks_global_scale: 2**{ckks_global_scale_exponent}")
        print(f"ckks_chunk_size: {ckks_chunk_size}")

    paths = artifact_paths(config, dry_run=dry_run)
    start_round, best_val_accuracy, best_round = maybe_resume_global_model(config, global_model, device)
    if start_round > global_rounds:
        raise ValueError(
            f"Resume checkpoint is already at round {start_round - 1}, "
            f"but requested global_rounds={global_rounds}."
        )

    max_client_batches = 2 if dry_run else None
    max_val_batches = 2 if dry_run else None
    client_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    ckks_encryption_time_total = 0.0
    ckks_aggregation_time_total = 0.0
    ckks_decryption_time_total = 0.0
    training_start = time.perf_counter()

    for round_index in range(start_round, global_rounds + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        round_start = time.perf_counter()
        current_learning_rate = round_learning_rate(
            base_learning_rate=learning_rate,
            scheduler_name=scheduler_name,
            round_index=round_index,
            total_rounds=global_rounds,
        )
        global_state = cpu_state_dict(global_model)
        client_states: list[dict[str, torch.Tensor]] = []

        for client_id, client_csv in enumerate(client_csvs):
            client_loader = build_client_loader(
                client_csv,
                batch_size,
                num_workers,
                device_is_cuda,
                aug_policy=aug_policy,
            )
            client_model = get_model(
                model_name=config["model_name"],
                num_classes=num_classes,
                pretrained=False,
            ).to(device)
            client_model.load_state_dict(global_state)

            optimizer = torch.optim.AdamW(
                client_model.parameters(),
                lr=current_learning_rate,
                weight_decay=weight_decay,
            )
            scaler = build_grad_scaler(amp_enabled)
            client_metrics = train_client(
                model=client_model,
                loader=client_loader,
                criterion=train_criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                amp_enabled=amp_enabled,
                local_epochs=local_epochs,
                round_index=round_index,
                client_id=client_id,
                max_batches=max_client_batches,
            )

            sample_count = client_sample_counts[client_id]
            client_states.append(cpu_state_dict(client_model))
            gpu_text = gpu_memory_text()
            print(
                f"Round {round_index}/{global_rounds} | "
                f"Client {client_id} | "
                f"samples={sample_count} | "
                f"lr={current_learning_rate:.6g} | "
                f"loss={client_metrics['loss']:.4f} | "
                f"acc={client_metrics['accuracy'] * 100:.1f}% | "
                f"gpu={gpu_text}"
            )
            client_rows.append(
                {
                    "round": round_index,
                    "client_id": client_id,
                    "client_csv": client_csv.as_posix(),
                    "samples": sample_count,
                    "learning_rate": current_learning_rate,
                    "train_loss": client_metrics["loss"],
                    "train_accuracy": client_metrics["accuracy"],
                    "batches_seen": int(client_metrics["batches_seen"]),
                }
            )

            del client_model, optimizer, scaler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ckks_info: dict[str, Any] = {
            "ckks_encryption_time": 0.0,
            "ckks_aggregation_time": 0.0,
            "ckks_decryption_time": 0.0,
            "max_absolute_error": None,
            "mean_absolute_error": None,
            "ckks_num_chunks": 0,
            "ckks_chunk_size": ckks_chunk_size,
        }
        if privacy == "selected_layer_ckks":
            aggregated_state, ckks_info = mixed_plaintext_ckks_fedavg(
                client_state_dicts=client_states,
                client_weights=client_sample_counts,
                selected_keys=selected_ckks_keys,
                context=ckks_context,
                chunk_size=ckks_chunk_size,
            )
            ckks_num_chunks = int(ckks_info["ckks_num_chunks"])
            ckks_encryption_time_total += float(ckks_info["ckks_encryption_time"])
            ckks_aggregation_time_total += float(ckks_info["ckks_aggregation_time"])
            ckks_decryption_time_total += float(ckks_info["ckks_decryption_time"])
        else:
            aggregated_state = aggregate_state_dicts(client_states, client_sample_counts)
        global_model.load_state_dict(aggregated_state)
        del client_states, aggregated_state

        val_metrics = evaluate(
            model=global_model,
            loader=val_loader,
            criterion=eval_criterion,
            device=device,
            amp_enabled=amp_enabled,
            desc=f"Round {round_index} val",
            max_batches=max_val_batches,
        )
        round_time = time.perf_counter() - round_start
        gpu_text = gpu_memory_text()
        print(
            f"Round {round_index}/{global_rounds} Summary | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy'] * 100:.1f}% | "
            f"time={round_time:.1f}s | "
            f"gpu={gpu_text}"
        )
        if privacy == "selected_layer_ckks":
            print(
                f"Round {round_index}/{global_rounds} CKKS | "
                f"enc={float(ckks_info['ckks_encryption_time']):.4f}s | "
                f"agg={float(ckks_info['ckks_aggregation_time']):.4f}s | "
                f"dec={float(ckks_info['ckks_decryption_time']):.4f}s | "
                f"chunks={ckks_info['ckks_num_chunks']}x{ckks_info['ckks_chunk_size']} | "
                f"max_err={ckks_info['max_absolute_error']:.3e} | "
                f"mean_err={ckks_info['mean_absolute_error']:.3e}"
            )

        round_rows.append(
            {
                "round": round_index,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "learning_rate": current_learning_rate,
                "ckks_encryption_time": ckks_info["ckks_encryption_time"],
                "ckks_aggregation_time": ckks_info["ckks_aggregation_time"],
                "ckks_decryption_time": ckks_info["ckks_decryption_time"],
                "ckks_chunk_size": ckks_info["ckks_chunk_size"],
                "ckks_num_chunks": ckks_info["ckks_num_chunks"],
                "ckks_max_absolute_error": ckks_info["max_absolute_error"],
                "ckks_mean_absolute_error": ckks_info["mean_absolute_error"],
                "round_time_sec": round_time,
                "gpu_memory_peak_gb": (
                    torch.cuda.max_memory_allocated() / (1024 ** 3)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_round = round_index
            save_best_checkpoint(
                model=global_model,
                config=config,
                path=paths["best_checkpoint_path"],
                round_index=round_index,
                best_val_accuracy=best_val_accuracy,
            )

        save_last_checkpoint(
            model=global_model,
            config=config,
            path=paths["last_checkpoint_path"],
            round_index=round_index,
            best_val_accuracy=best_val_accuracy,
            best_round=best_round,
        )

    total_training_time = time.perf_counter() - training_start
    test_metrics: dict[str, Any] | None = None

    if dry_run:
        print("FedAvg dry-run mode: full test evaluation skipped.")
    else:
        checkpoint = load_checkpoint(paths["best_checkpoint_path"], device)
        global_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best global checkpoint for test evaluation: {paths['best_checkpoint_path']}")
        test_metrics = evaluate(
            model=global_model,
            loader=test_loader,
            criterion=eval_criterion,
            device=device,
            amp_enabled=amp_enabled,
            desc="FedAvg test",
        )
        class_names = class_names_from_csv(config["test_csv"])
        save_test_reports(
            test_metrics=test_metrics,
            class_names=class_names,
            classification_report_path=paths["classification_report_path"],
            confusion_matrix_path=paths["confusion_matrix_path"],
        )

    save_csv(round_rows, paths["round_log_path"])
    save_csv(client_rows, paths["client_log_path"])

    test_accuracy = test_metrics["accuracy"] if test_metrics else None
    test_macro_f1 = test_metrics["macro_f1"] if test_metrics else None
    test_weighted_f1 = test_metrics["weighted_f1"] if test_metrics else None
    metrics_payload = {
        "experiment_id": experiment_id,
        "base_experiment_id": config.get("base_experiment_id"),
        "output_suffix": config.get("output_suffix"),
        "dry_run": dry_run,
        "privacy": privacy,
        "global_rounds": int(config.get("global_rounds", config.get("epochs", 5))),
        "actual_global_rounds": global_rounds,
        "local_epochs": local_epochs,
        "num_clients": num_clients,
        "client_split_dir": config["client_split_dir"],
        "client_sample_counts": client_sample_counts,
        "total_client_samples": total_client_samples,
        "best_round": best_round,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "total_training_time_seconds": total_training_time,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "amp_enabled": amp_enabled,
        "model_name": config["model_name"],
        "num_classes": num_classes,
        "model_parameter_count": parameter_count,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "aug_policy": aug_policy,
        "label_smoothing": label_smoothing,
        "scheduler": scheduler_name,
        "seed": int(config["seed"]),
        "selected_ckks_keys": selected_ckks_keys,
        "selected_ckks_num_parameters": selected_ckks_num_parameters,
        "ckks_poly_modulus_degree": ckks_poly_modulus_degree if privacy == "selected_layer_ckks" else None,
        "ckks_coeff_mod_bit_sizes": ckks_coeff_mod_bit_sizes if privacy == "selected_layer_ckks" else None,
        "ckks_global_scale": ckks_global_scale if privacy == "selected_layer_ckks" else None,
        "ckks_global_scale_exponent": (
            ckks_global_scale_exponent if privacy == "selected_layer_ckks" else None
        ),
        "ckks_chunk_size": ckks_chunk_size if privacy == "selected_layer_ckks" else None,
        "ckks_num_chunks": ckks_num_chunks if privacy == "selected_layer_ckks" else None,
        "ckks_encryption_time_total": ckks_encryption_time_total,
        "ckks_aggregation_time_total": ckks_aggregation_time_total,
        "ckks_decryption_time_total": ckks_decryption_time_total,
        "round_log_path": str(paths["round_log_path"]),
        "client_log_path": str(paths["client_log_path"]),
        "checkpoint_path": str(paths["best_checkpoint_path"]),
        "best_checkpoint_path": str(paths["best_checkpoint_path"]),
        "last_checkpoint_path": str(paths["last_checkpoint_path"]),
        "classification_report_path": str(paths["classification_report_path"]),
        "confusion_matrix_path": str(paths["confusion_matrix_path"]),
    }
    save_json(metrics_payload, paths["metrics_path"])

    print("\nFedAvg run completed.")
    print(f"best_round: {best_round}")
    print(f"best_val_accuracy: {best_val_accuracy:.4f}")
    print(f"test_accuracy: {test_accuracy:.4f}" if test_accuracy is not None else "test_accuracy: n/a")
    print(f"test_macro_f1: {test_macro_f1:.4f}" if test_macro_f1 is not None else "test_macro_f1: n/a")
    print(
        f"test_weighted_f1: {test_weighted_f1:.4f}"
        if test_weighted_f1 is not None
        else "test_weighted_f1: n/a"
    )
    print(f"metrics_path: {paths['metrics_path']}")
    print(f"round_log_path: {paths['round_log_path']}")
    print(f"client_log_path: {paths['client_log_path']}")
    print(f"checkpoint_path: {paths['best_checkpoint_path']}")
    print(f"last_checkpoint_path: {paths['last_checkpoint_path']}")
    if privacy == "selected_layer_ckks":
        print(f"selected_ckks_keys: {selected_ckks_keys}")
        print(f"selected_ckks_num_parameters: {selected_ckks_num_parameters}")
        print(f"ckks_chunk_size: {ckks_chunk_size}")
        print(f"ckks_num_chunks: {ckks_num_chunks}")
        print(f"ckks_encryption_time_total: {ckks_encryption_time_total:.4f}s")
        print(f"ckks_aggregation_time_total: {ckks_aggregation_time_total:.4f}s")
        print(f"ckks_decryption_time_total: {ckks_decryption_time_total:.4f}s")

    return metrics_payload


def cli_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment_id": args.experiment_id,
        "experiment_name": args.experiment_id,
        "dataset": args.dataset,
        "client_split_dir": args.client_split_dir,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "output_dir": args.output_dir,
        "metrics_path": args.metrics_path,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "privacy": args.privacy,
        "num_clients": args.num_clients,
        "global_rounds": args.global_rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": args.device,
        "resume_from": args.resume_from,
        "aug_policy": args.aug_policy,
        "label_smoothing": args.label_smoothing,
        "scheduler": args.scheduler,
        "selected_ckks_keys": args.selected_ckks_keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FedAvg ViT-Base experiment.")
    parser.add_argument("--experiment_id", default="fedavg_cli")
    parser.add_argument("--dataset", default="UCMerced Land Use")
    parser.add_argument("--client_split_dir", default="data/splits/clients_iid")
    parser.add_argument("--val_csv", default="data/splits/val.csv")
    parser.add_argument("--test_csv", default="data/splits/test.csv")
    parser.add_argument("--output_dir", default="experiments/fedavg_iid/fedavg_cli")
    parser.add_argument("--metrics_path", default="results/metrics/fedavg_cli_metrics.json")
    parser.add_argument("--model_name", default="vit_base_patch16_224")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--privacy", choices=["none", "selected_layer_ckks"], default="none")
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--global_rounds", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="Optional device override, for example cuda or cpu.")
    parser.add_argument("--resume_from", help="Optional FedAvg checkpoint to resume from.")
    parser.add_argument(
        "--aug_policy",
        choices=sorted(SUPPORTED_AUG_POLICIES),
        default="basic",
        help="Training augmentation policy for local client loaders.",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument(
        "--selected_ckks_keys",
        nargs="+",
        help="Optional explicit CKKS tensor keys, for example head.weight head.bias norm.weight norm.bias.",
    )
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
