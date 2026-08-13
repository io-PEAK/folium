"""Model definitions and freeze/unfreeze helpers (see AGENT.md -> ml/model.py).

Primary backbone is MobileNetV2 pretrained on ImageNet; EfficientNet-B0 is
available as the comparison baseline the paper needs (Sprint 4).

Sprint 1 (Stage 1): base is frozen, only the new classification head trains.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

# classifier[1].in_features for each supported backbone (both use a
# Sequential(Dropout, Linear) tail, so the pattern below is uniform).
MODEL_FEATURE_DIMS = {
    "mobilenet_v2": 1280,
    "efficientnet_b0": 1280,
}


def _pretrained_weights(backbone: str):
    return {
        "mobilenet_v2": models.MobileNet_V2_Weights.IMAGENET1K_V1,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    }[backbone]


def build_model(
    num_classes: int,
    backbone: str = "mobilenet_v2",
    pretrained: bool = True,
    freeze: bool = True,
) -> nn.Module:
    """Build a CNN classifier for ``num_classes``.

    Args:
        num_classes: number of output classes (38 for PlantVillage).
        backbone: model name in ``MODEL_FEATURE_DIMS``.
        pretrained: load ImageNet weights.
        freeze: keep the backbone frozen (Stage 1); unfreeze later with
            ``unfreeze_last_blocks`` for Stage 2 fine-tuning.
    """
    if backbone not in MODEL_FEATURE_DIMS:
        raise ValueError(f"Unsupported backbone '{backbone}'; choose from {sorted(MODEL_FEATURE_DIMS)}")

    model = getattr(models, backbone)(weights=_pretrained_weights(backbone) if pretrained else None)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    in_features = MODEL_FEATURE_DIMS[backbone]
    model.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_features, num_classes))
    return model


def unfreeze_last_blocks(model: nn.Module, n_blocks: int) -> None:
    """Re-enable gradients on the last ``n_blocks`` *parameter-bearing* backbone modules.

    Stage 2 fine-tuning trains only these blocks (at a much lower LR) plus the
    head, leaving early layers intact to avoid catastrophic forgetting.

    We filter to modules that actually own parameters: torchvision's MobileNetV2
    ``features`` ends with a 1x1 conv, ReLU6, AdaptiveAvgPool2d and Flatten, so a
    naive ``features[-n:]`` could unfreeze pooling/flatten (zero trainable params)
    and silently do nothing.
    """
    if n_blocks <= 0:
        return
    trainable_modules = [m for m in model.features if len(list(m.parameters())) > 0]
    for module in trainable_modules[-n_blocks:]:
        for param in module.parameters():
            param.requires_grad = True


def trainable_parameters(model: nn.Module):
    """Iterate parameters with ``requires_grad=True`` (the optimizer's job)."""
    return (p for p in model.parameters() if p.requires_grad)
