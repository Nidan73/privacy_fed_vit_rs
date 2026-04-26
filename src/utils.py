from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from config import IMAGE_EXTENSIONS, PROJECT_ROOT


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def project_relative(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def scan_image_paths(data_dir: str | Path) -> list[Path]:
    root = resolve_path(data_dir)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
