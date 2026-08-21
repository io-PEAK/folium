"""Model definitions and freeze/unfreeze helpers (see AGENT.md -> ml/model.py).

Primary backbone is MobileNetV2 pretrained on ImageNet; EfficientNet-B0 is
available as the comparison baseline the paper needs (Sprint 4). Sprint 7 adds
ResNet-50 (and keeps EfficientNet-B0) to push the field-photo ceiling toward
0.60+ F1, since MobileNetV2's best field score so far is `both`'s 0.5578.

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
    "resnet50": 2048,
}


def _pretrained_weights(backbone: str):
    return {
        "mobilenet_v2": models.MobileNet_V2_Weights.IMAGENET1K_V1,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
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
    head = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_features, num_classes))
    if "resnet" in backbone:
        model.fc = head
    else:
        model.classifier = head
    return model


def head_module(model: nn.Module) -> nn.Module:
    """Return the classification module whose parameters train as the 'head'.

    torchvision ResNets call it ``fc``; MobileNet/EfficientNet call it
    ``classifier``. Both are the final Sequential(Dropout, Linear) tail.
    """
    if "resnet" in type(model).__name__.lower():
        return model.fc
    return model.classifier


def _feature_modules(model: nn.Module) -> list[nn.Module]:
    """The backbone's sequential feature modules (for unfreeze-by-position).

    MobileNetV2 / EfficientNet expose ``features``; ResNets expose
    ``conv1..layer4`` as named children.
    """
    name = type(model).__name__.lower()
    if name in ("mobilenetv2", "efficientnet"):
        return list(model.features)
    if "resnet" in name:
        return [model.conv1, model.bn1, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4]
    raise ValueError(f"Unsupported backbone type '{name}'")


def unfreeze_last_blocks(model: nn.Module, n_blocks: int) -> None:
    """Re-enable gradients on the last ``n_blocks`` *parameter-bearing* backbone modules.

    Stage 2 fine-tuning trains only these blocks (at a much lower LR) plus the
    head, leaving early layers intact to avoid catastrophic forgetting.

    We filter to modules that actually own parameters: torchvision's MobileNetV2
    ``features`` ends with a 1x1 conv, ReLU6, AdaptiveAvgPool2d and Flatten, so a
    naive ``features[-n:]`` could unfreeze pooling/flatten (zero trainable params)
    and silently do nothing. Same filtering applies to ResNet (skips ``maxpool``).
    """
    if n_blocks <= 0:
        return
    trainable_modules = [m for m in _feature_modules(model) if len(list(m.parameters())) > 0]
    for module in trainable_modules[-n_blocks:]:
        for param in module.parameters():
            param.requires_grad = True


def trainable_parameters(model: nn.Module):
    """Iterate parameters with ``requires_grad=True`` (the optimizer's job)."""
    return (p for p in model.parameters() if p.requires_grad)


class DualHeadModel(nn.Module):
    """Backbone with two independent classification heads (Sprint 8/9).

    Solves the lab-vs-field catastrophic forgetting: each head trains on its
    own domain and never sees the other.  A lightweight domain classifier
    routes each image to the correct head at inference time.

    When ``separate_backbones=True`` (Sprint 9), each head gets its OWN
    backbone — this eliminates the shared-backbone interference that capped
    the field head at 0.41 in Sprint 8 (Sprint 8 proved the bottleneck is the
    backbone, not the head). The domain classifier runs on the lab backbone's
    features (more stable, trained on more data).

    Architecture (separate_backbones=True)::

        backbone_lab   (PV-trained)    -> head_lab
        backbone_field (PD-adapted)    -> head_field
        domain_classifier (on backbone_lab features) -> lab=0 / field=1
    """

    def __init__(self, backbone: nn.Module, in_features: int, num_classes: int, backbone_name: str, separate_backbones: bool = False):
        super().__init__()
        self.backbone_name = backbone_name
        self.in_features = in_features
        self.num_classes = num_classes
        self.separate_backbones = separate_backbones

        if separate_backbones:
            self.backbone_lab = backbone
            # backbone_field is attached by build_dual_head_model after __init__
        else:
            self.backbone = backbone

        self.head_lab = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_features, num_classes))
        self.head_field = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_features, num_classes))
        self.domain_classifier = nn.Linear(in_features, 2)

    def _features(self, x: torch.Tensor, which: str) -> torch.Tensor:
        """Run the appropriate backbone, return flat feature vector."""
        if self.separate_backbones:
            bb = self.backbone_lab if which == "lab" else self.backbone_field
        else:
            bb = self.backbone
        x = bb(x)
        return torch.flatten(x, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the lab backbone (used by the domain classifier)."""
        return self._features(x, "lab")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (lab_logits, field_logits)."""
        if self.separate_backbones:
            lab_feat = self._features(x, "lab")
            field_feat = self._features(x, "field")
            return self.head_lab(lab_feat), self.head_field(field_feat)
        feat = self._features(x, "shared")
        return self.head_lab(feat), self.head_field(feat)

    @torch.no_grad()
    def predict_dual(self, x: torch.Tensor) -> torch.Tensor:
        """Run both heads, return the argmax from the higher-confidence head per sample."""
        lab_out, field_out = self.forward(x)
        lab_probs = torch.softmax(lab_out, dim=1)
        field_probs = torch.softmax(field_out, dim=1)
        lab_conf = lab_probs.max(dim=1).values
        field_conf = field_probs.max(dim=1).values

        use_lab = lab_conf >= field_conf
        lab_preds = lab_out.argmax(dim=1)
        field_preds = field_out.argmax(dim=1)
        return torch.where(use_lab, lab_preds, field_preds)

    @torch.no_grad()
    def predict_routed(self, x: torch.Tensor) -> torch.Tensor:
        """Domain-classifier routing: backbone -> domain clf -> correct head."""
        if self.separate_backbones:
            lab_feat = self._features(x, "lab")
            domain_pred = self.domain_classifier(lab_feat).argmax(dim=1)
            field_feat = self._features(x, "field")
            lab_out = self.head_lab(lab_feat)
            field_out = self.head_field(field_feat)
            return torch.where(domain_pred == 0, lab_out.argmax(dim=1), field_out.argmax(dim=1))
        feat = self.extract_features(x)
        domain_pred = self.domain_classifier(feat).argmax(dim=1)
        lab_out = self.head_lab(feat)
        field_out = self.head_field(feat)
        return torch.where(domain_pred == 0, lab_out.argmax(dim=1), field_out.argmax(dim=1))

    @torch.no_grad()
    def predict_head(self, x: torch.Tensor, which: str) -> torch.Tensor:
        """Run only the named head, return argmax predictions."""
        feat = self._features(x, which)
        out = self.head_lab(feat) if which == "lab" else self.head_field(feat)
        return out.argmax(dim=1)

    def get_head(self, which: str) -> nn.Module:
        """Return ``head_lab``, ``head_field``, or ``domain_classifier``."""
        if which == "lab":
            return self.head_lab
        if which == "field":
            return self.head_field
        if which == "domain":
            return self.domain_classifier
        raise ValueError(f"which must be 'lab', 'field', or 'domain', got '{which}'")

    def freeze_all_except(self, which: str) -> None:
        """Freeze everything except the named component."""
        for param in self.parameters():
            param.requires_grad = False
        if which == "domain":
            for param in self.domain_classifier.parameters():
                param.requires_grad = True
        elif which == "lab":
            if self.separate_backbones:
                for param in self.backbone_lab.parameters():
                    param.requires_grad = True
            for param in self.head_lab.parameters():
                param.requires_grad = True
        elif which == "field":
            if self.separate_backbones:
                for param in self.backbone_field.parameters():
                    param.requires_grad = True
            for param in self.head_field.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"which must be 'lab', 'field', or 'domain', got '{which}'")


def build_dual_head_model(
    num_classes: int,
    backbone: str = "resnet50",
    pretrained: bool = True,
    separate_backbones: bool = False,
) -> DualHeadModel:
    """Build a ``DualHeadModel``.

    With ``separate_backbones=False`` (Sprint 8): one frozen backbone shared by
    both heads. With ``separate_backbones=True`` (Sprint 9): two independent
    backbones, each trained on its own domain — eliminates shared-backbone
    interference so the field head can reach the PlantDoc ceiling (~0.66).
    """
    if backbone not in MODEL_FEATURE_DIMS:
        raise ValueError(f"Unsupported backbone '{backbone}'; choose from {sorted(MODEL_FEATURE_DIMS)}")

    in_features = MODEL_FEATURE_DIMS[backbone]

    def make_feature_backbone() -> nn.Module:
        raw = getattr(models, backbone)(weights=_pretrained_weights(backbone) if pretrained else None)
        for param in raw.parameters():
            param.requires_grad = False
        if "resnet" in backbone:
            return nn.Sequential(
                raw.conv1, raw.bn1, raw.relu, raw.maxpool,
                raw.layer1, raw.layer2, raw.layer3, raw.layer4,
                raw.avgpool,
            )
        return raw.features

    if separate_backbones:
        backbone_lab = make_feature_backbone()
        backbone_field = make_feature_backbone()
        model = DualHeadModel(backbone_lab, in_features, num_classes, backbone, separate_backbones=True)
        model.backbone_field = backbone_field
        return model

    feature_backbone = make_feature_backbone()
    return DualHeadModel(feature_backbone, in_features, num_classes, backbone, separate_backbones=False)
