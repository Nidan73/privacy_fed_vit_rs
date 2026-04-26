from __future__ import annotations

import timm
import torch.nn as nn


SUPPORTED_MODEL_NAMES = {"vit_base_patch16_224"}


def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            f"Allowed models for the frozen setup: {sorted(SUPPORTED_MODEL_NAMES)}"
        )

    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def get_vit_base(num_classes: int) -> nn.Module:
    """Create a ViT-Base model for image classification.

    This only defines the model. Training is intentionally not started here.
    """
    return get_model("vit_base_patch16_224", num_classes=num_classes, pretrained=True)
