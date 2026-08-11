"""Dataset loading and preprocessing (see AGENT.md -> ml/data_loading.py).

- `augment=False`: Sprint 1 / Sprint 3-baseline path. No random transforms —
  this exact class is the ablation baseline.
- `augment=True`: Sprint 3 path. Albumentations pipeline (flips, rotation,
  brightness/contrast, blur, perspective, JPEG compression artifacts) applied
  to the TRAINING split only. Val/test are NEVER augmented — evaluation must
  be deterministic.

Data is expected in the layout produced by scripts/organize_datasets.py:

    <data-dir>/<dataset>/{train,val,test}/<class>/*.jpg
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def _augment_transform(image_size: int) -> transforms.Compose:
    """Albumentations pipeline -> a [0,1] CHW float tensor (then torch Normalize)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Perspective(scale=(0.03, 0.08), keep_size=True, p=0.3),
            A.ImageCompression(compression_type="jpeg", quality_range=(60, 95), p=0.3),
            A.Resize(image_size, image_size),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ]
    )


class _AlbumentationsAdapter:
    """Wraps an Albumentations pipeline so torchvision.ImageFolder can use it.

    ImageFolder hands us a PIL image; albumentations wants HWC numpy; the
    pipeline's ToTensorV2 hands back a CHW float tensor in [0, 1].
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, pil_image):
        return self.pipeline(image=np.array(pil_image))["image"]


def make_transforms(image_size: int = IMAGE_SIZE, augment: bool = False) -> transforms.Compose:
    """Preprocessing shared by every split and by predict.py.

    Augmented (training) path resizes+crops inside albumentations; the plain
    path matches MobileNetV2's ImageNet convention (resize 256, center-crop).
    """
    if augment:
        return transforms.Compose(
            [
                _AlbumentationsAdapter(_augment_transform(image_size)),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _worker_init_fn(worker_id: int) -> None:
    """Deterministic per-worker seeding."""
    torch.manual_seed(42 + worker_id)


def build_loaders(
    data_dir: str,
    dataset: str = "plantvillage",
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    pin_memory: bool = True,
    augment: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train/val/test DataLoaders (ImageFolder) for one dataset.

    Returns (train_loader, val_loader, test_loader, class_names). class_names
    maps model output index -> class folder name; it is what gets saved in
    checkpoints so evaluate/predict can label predictions. Augmentation is
    applied to the training loader only.
    """
    root = Path(data_dir) / dataset

    train_ds = datasets.ImageFolder(root / "train", transform=make_transforms(augment=augment))
    val_ds = datasets.ImageFolder(root / "val", transform=make_transforms())
    test_ds = datasets.ImageFolder(root / "test", transform=make_transforms())

    generator = torch.Generator()
    generator.manual_seed(seed)

    common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _worker_init_fn,
    }
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator, **common
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader, train_ds.classes
