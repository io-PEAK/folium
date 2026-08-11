#!/usr/bin/env python3
"""Organize raw datasets into the <split>/<class>/ image layout used everywhere.

Builds (see AGENT.md):

    <data-dir>/plantvillage/{train,val,test}/<class>/image.jpg
    <data-dir>/plantdoc/{train,val,test}/<class>/image.jpg

- PlantVillage: stratified 80/10/10 train/val/test split, fixed seed.
- PlantDoc: keeps the dataset's shipped test set, carves `--val-fraction` of
  the training images out as validation.

Also writes <data-dir>/class_map.json: the PlantDoc -> PlantVillage class
mapping for the classes present in both datasets. This aligned class set is
what the ablation study and cross-dataset evaluation use later.

Run after scripts/download_datasets.py:

    python scripts/organize_datasets.py --data-dir data

Raw images are read from --raw-dir (default: --data-dir). On Colab, raw lives
on Google Drive while the organized splits are written locally so the heavy
copies don't hit the flaky Drive FUSE mount:

    python scripts/organize_datasets.py \
        --raw-dir /content/drive/MyDrive/folium/data \
        --data-dir /content/folium_data
"""
import argparse
import json
import random
import shutil
from pathlib import Path

DEFAULT_SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# PlantDoc folder name -> PlantVillage class label (as present in the
# mohanty/PlantVillage "color" config). Derives from the class lists of the
# two upstream repos. Any PlantDoc class not in this dict is reported as
# unmatched when the script runs.
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_yellow_leaf_curl_virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Two-spotted_spider_mite",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}


def _class_dirs(raw_root: Path) -> list[Path]:
    if not raw_root.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_root}. Run scripts/download_datasets.py first."
        )
    return sorted(p for p in raw_root.iterdir() if p.is_dir())


def _images_in(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _split_files(files: list[Path], fractions: list[float], seed: int) -> list[list[Path]]:
    """Deterministically split a shuffled list into len(fractions) buckets."""
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    buckets, start = [], 0
    for frac in fractions:
        n = int(len(shuffled) * frac)
        buckets.append(shuffled[start:start + n])
        start += n
    buckets[-1].extend(shuffled[start:])  # remainder goes to the last bucket
    return buckets


def _copy(files: list[Path], dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dst_dir / src.name)
    return len(files)


def organize_plantvillage(data_dir: Path, raw_dir: Path, seed: int, val_frac: float, test_frac: float) -> dict:
    raw = raw_dir / "plantvillage" / "raw"
    print(f"[plantvillage] organizing {raw} ...")
    summary = {}
    total = 0
    for class_dir in _class_dirs(raw):
        files = _images_in(class_dir)
        if not files:
            print(f"[plantvillage]  WARN: no images in {class_dir.name}; skipping")
            continue
        train, val, test = _split_files(files, [1 - val_frac - test_frac, val_frac, test_frac], seed)
        for split, bucket in zip(("train", "val", "test"), (train, val, test)):
            n = _copy(bucket, data_dir / "plantvillage" / split / class_dir.name)
            summary.setdefault(split, {})[class_dir.name] = n
            total += n
    for split in ("train", "val", "test"):
        count = sum(summary.get(split, {}).values())
        print(f"[plantvillage]  {split}: {count} images across {len(summary.get(split, {}))} classes")
    print(f"[plantvillage] total {total} images")
    return summary


def organize_plantdoc(data_dir: Path, raw_dir: Path, seed: int, val_frac: float) -> dict:
    raw = raw_dir / "plantdoc" / "raw"
    print(f"[plantdoc] organizing {raw} ...")
    summary = {}

    for class_dir in _class_dirs(raw / "train"):
        files = _images_in(class_dir)
        train, val = _split_files(files, [1 - val_frac, val_frac], seed)
        for split, bucket in zip(("train", "val"), (train, val)):
            n = _copy(bucket, data_dir / "plantdoc" / split / class_dir.name)
            summary.setdefault(split, {})[class_dir.name] = n

    for class_dir in _class_dirs(raw / "test"):
        n = _copy(_images_in(class_dir), data_dir / "plantdoc" / "test" / class_dir.name)
        summary.setdefault("test", {})[class_dir.name] = n

    for split in ("train", "val", "test"):
        count = sum(summary.get(split, {}).values())
        print(f"[plantdoc]  {split}: {count} images across {len(summary.get(split, {}))} classes")
    return summary


def write_class_map(data_dir: Path) -> None:
    doc_root = data_dir / "plantdoc" / "train"
    mapped, unmatched = {}, {}
    if doc_root.exists():
        for class_dir in _class_dirs(doc_root):
            name = class_dir.name
            if name in PLANTDOC_TO_PLANTVILLAGE:
                mapped[name] = PLANTDOC_TO_PLANTVILLAGE[name]
            else:
                unmatched[name] = None
    else:
        # Fall back to the mapping table so the file is written even before PlantDoc is organized.
        mapped = dict(PLANTDOC_TO_PLANTVILLAGE)

    out_path = data_dir / "class_map.json"
    out_path.write_text(
        json.dumps(
            {
                "description": "PlantDoc class name -> PlantVillage class label, for the classes present in both datasets. Used for the ablation study and cross-dataset evaluation (see AGENT.md / final_brief_and_plan.md).",
                "plantdoc_to_plantvillage": dict(sorted(mapped.items())),
                "unmatched_plantdoc_classes": dict(sorted(unmatched.items())),
            },
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )
    print(f"[class_map] {len(mapped)} aligned classes, {len(unmatched)} unmatched -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="root directory where train/val/test and class_map.json are written (default: data)")
    parser.add_argument("--raw-dir", type=Path, default=None, help="directory holding <dataset>/raw to read images from (default: --data-dir; point this at the Drive data dir when splits are written locally)")
    parser.add_argument("--datasets", choices=("plantvillage", "plantdoc", "both"), default="both", help="which dataset(s) to organize (default: both)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed for reproducible splits (default: 42)")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="validation fraction of the training data (default: 0.1)")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="held-out test fraction for PlantVillage (default: 0.1)")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    raw_dir = (args.raw_dir or args.data_dir).resolve()

    if args.datasets in ("plantvillage", "both"):
        organize_plantvillage(data_dir, raw_dir, args.seed, args.val_fraction, args.test_fraction)
    if args.datasets in ("plantdoc", "both"):
        organize_plantdoc(data_dir, raw_dir, args.seed, args.val_fraction)

    write_class_map(data_dir)


if __name__ == "__main__":
    main()
