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

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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


class _MappedImageFolder:
    """ImageFolder whose folder names are translated to PlantVillage class indices.

    Used for cross-dataset evaluation / fine-tuning: PlantDoc folders are renamed
    to their aligned PlantVillage class via ``class_map.json``, so a 38-class
    PlantVillage model can be trained and scored on PlantDoc images. Samples whose
    folder is not in the map are dropped.
    """

    def __init__(self, root: Path, class_map: dict, pv_names: list[str], transform=None):
        inner = datasets.ImageFolder(str(root))
        pv_index = {name: i for i, name in enumerate(pv_names)}
        self.samples = [
            (path, pv_index[class_map[inner.classes[cls]]])
            for (path, cls) in inner.samples
            if inner.classes[cls] in class_map and class_map[inner.classes[cls]] in pv_index
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _load_class_map(data_dir: Path) -> dict:
    path = data_dir / "class_map.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/organize_datasets.py (it writes class_map.json)."
        )
    return json.loads(path.read_text())["plantdoc_to_plantvillage"]


def build_loaders(
    data_dir: str,
    dataset: str = "plantvillage",
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    pin_memory: bool = True,
    augment: bool = False,
    map_to_pv: bool = False,
    mix_with: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train/val/test DataLoaders (ImageFolder) for one dataset.

    Returns (train_loader, val_loader, test_loader, class_names). class_names
    maps model output index -> class folder name; it is what gets saved in
    checkpoints so evaluate/predict can label predictions. Augmentation is
    applied to the training loader only.

    ``map_to_pv=True`` (Sprint 4): read PlantDoc under ``dataset`` but translate
    its class folders to PlantVillage classes via class_map.json, and return the
    full sorted PlantVillage class list as ``class_names`` (so the 38-class model
    head stays compatible). Only the `plantvillage` folder layout can provide the
    authoritative class list, so this requires the PlantVillage data to be
    organized too.

    ``mix_with="plantdoc"`` (Sprint 5): train on PlantVillage (``dataset``) AND
    PlantDoc together — ``torch.utils.data.ConcatDataset`` of the two train sets
    (and the two val sets), both already in the shared 38-class PlantVillage label
    space. The fix for the Sprint 4 catastrophic forgetting: every epoch sees both
    domains, so the head keeps lab knowledge while learning field photos.
    """
    root = Path(data_dir) / dataset

    if mix_with is not None:
        if dataset != "plantvillage" or mix_with != "plantdoc":
            raise ValueError(
                f"mix_with only supports dataset='plantvillage' + mix_with='plantdoc', "
                f"got '{dataset}' + '{mix_with}'"
            )
        pv_root = Path(data_dir) / "plantvillage"
        pd_root = Path(data_dir) / mix_with
        class_map = _load_class_map(Path(data_dir))
        pv_names = datasets.ImageFolder(str(pv_root / "train")).classes
        make_mapped = lambda split, aug: _MappedImageFolder(
            pd_root / split,
            class_map,
            pv_names,
            transform=make_transforms(augment=aug),
        )
        train_ds = torch.utils.data.ConcatDataset(
            [
                datasets.ImageFolder(pv_root / "train", transform=make_transforms(augment=augment)),
                make_mapped("train", augment),
            ]
        )
        val_ds = torch.utils.data.ConcatDataset(
            [
                datasets.ImageFolder(pv_root / "val", transform=make_transforms()),
                make_mapped("val", False),
            ]
        )
        test_ds = datasets.ImageFolder(pv_root / "test", transform=make_transforms())
        class_names = pv_names
    elif map_to_pv:
        if dataset != "plantdoc":
            raise ValueError(f"map_to_pv only makes sense for dataset='plantdoc', got '{dataset}'")
        class_map = _load_class_map(Path(data_dir))
        pv_names = datasets.ImageFolder(str(Path(data_dir) / "plantvillage" / "train")).classes
        make_ds = lambda split: _MappedImageFolder(
            root / split,
            class_map,
            pv_names,
            transform=make_transforms(augment=augment and split == "train"),
        )
        train_ds, val_ds, test_ds = make_ds("train"), make_ds("val"), make_ds("test")
        class_names = pv_names
    else:
        train_ds = datasets.ImageFolder(root / "train", transform=make_transforms(augment=augment))
        val_ds = datasets.ImageFolder(root / "val", transform=make_transforms())
        test_ds = datasets.ImageFolder(root / "test", transform=make_transforms())
        class_names = train_ds.classes

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

    return train_loader, val_loader, test_loader, class_names
