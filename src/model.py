from __future__ import annotations

import timm
import torch.nn as nn


def get_vit_base(num_classes: int) -> nn.Module:
    """Create a ViT-Base model for image classification.

    This only defines the model. Training is intentionally not started here.
    """
    return timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes,
    )
