from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from utils import resolve_path


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def format_size(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 2):.2f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a PyTorch checkpoint file.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to .pt checkpoint.")
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path)
    keys = sorted(checkpoint.keys())
    best_val_accuracy = checkpoint.get("best_val_accuracy")
    if best_val_accuracy is None and checkpoint.get("val_metrics") is not None:
        best_val_accuracy = checkpoint["val_metrics"].get("accuracy")

    print("Checkpoint Inspection")
    print("=====================")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"file_size: {format_size(checkpoint_path.stat().st_size)}")
    print(f"saved_epoch: {checkpoint.get('epoch', 'not available')}")
    print(f"best_val_accuracy: {best_val_accuracy if best_val_accuracy is not None else 'not available'}")
    print(f"best_epoch: {checkpoint.get('best_epoch', 'not available')}")
    print("keys:")
    for key in keys:
        print(f"- {key}")


if __name__ == "__main__":
    main()
