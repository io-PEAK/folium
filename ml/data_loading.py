"""Dataset loading and preprocessing (see AGENT.md -> ml/data_loading.py).

Sprint 1 uses PlantVillage only, with NO augmentation — the exact class is
the ablation baseline. Augmentation lands in Sprint 3.

Data is expected in the layout produced by scripts/organize_datasets.py:

    <data-dir>/<dataset>/{train,val,test}/<class>/*.jpg
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def make_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Preprocessing shared by every split and by predict.py.

    Resize the shorter side to 256 then center-crop 224, matching MobileNetV2's
    ImageNet input convention. No random transforms yet.
    """
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
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train/val/test DataLoaders (ImageFolder) for one dataset.

    Returns (train_loader, val_loader, test_loader, class_names). class_names
    maps model output index -> class folder name; it is what gets saved in
    checkpoints so evaluate/predict can label predictions.
    """
    root = Path(data_dir) / dataset
    transform = make_transforms()

    train_ds = datasets.ImageFolder(root / "train", transform=transform)
    val_ds = datasets.ImageFolder(root / "val", transform=transform)
    test_ds = datasets.ImageFolder(root / "test", transform=transform)

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
