from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from utils import resolve_path


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class RemoteSensingCSVDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        transform: transforms.Compose | None = None,
        train: bool = False,
    ) -> None:
        self.csv_path = resolve_path(csv_path)
        self.data = pd.read_csv(self.csv_path)

        required_columns = {"image_path", "label", "class_name"}
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

        self.transform = transform or (get_train_transforms() if train else get_eval_transforms())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.data.iloc[index]
        image_path = resolve_path(row["image_path"])
        label = int(row["label"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        return image_tensor, label
