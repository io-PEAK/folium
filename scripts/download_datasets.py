#!/usr/bin/env python3
"""Download PlantVillage and PlantDoc datasets into <data-dir>/<name>/raw.

Runs identically on Google Colab and a local machine — no hardcoded paths,
everything is configurable via CLI args (see AGENT.md conventions).

    python scripts/download_datasets.py --dataset all --data-dir data

Datasets:
  - PlantVillage  Hugging Face "mohanty/PlantVillage" (default config = color images)
  - PlantDoc      GitHub "pratikkayal/PlantDoc-Dataset" (git clone)

If a dataset already exists it is skipped (use --force to re-download).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATASETS = ("plantvillage", "plantdoc")

PLANTVILLAGE_HF_ID = "mohanty/PlantVillage"
PLANTDOC_REPO_URL = "https://github.com/pratikkayal/PlantDoc-Dataset.git"


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


def _check_case_sensitive() -> bool:
    """Return True if the filesystem distinguishes filename case.

    The PlantDoc repo contains 6 pairs of files whose names differ only in
    case (e.g. CAR1.jpg / car1.jpg). Cloning on a case-insensitive filesystem
    (macOS APFS/HFS+, Windows NTFS) silently drops one file per pair, so we
    warn before cloning there.
    """
    probe_a = Path(".folium_case_probe_a")
    probe_b = Path(".folium_case_probe_B")
    try:
        probe_a.write_bytes(b"a")
        probe_b.write_bytes(b"b")
        return probe_a.read_bytes() == b"a" and probe_b.read_bytes() == b"b"
    except OSError:
        return True
    finally:
        probe_a.unlink(missing_ok=True)
        probe_b.unlink(missing_ok=True)


def download_plantvillage(data_dir: Path, sample_limit: int | None, force: bool) -> int:
    out = data_dir / "plantvillage" / "raw"
    existing = _count_images(out)
    if existing and not force:
        print(f"[plantvillage] {existing} images already present at {out}; skipping (use --force to re-download)")
        return existing

    try:
        from datasets import load_dataset
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}\n  pip install datasets  (see requirements.txt)")

    print(f"[plantvillage] downloading {PLANTVILLAGE_HF_ID} via Hugging Face ...")
    print("[plantvillage] first download is data.zip (~2 GB), extracted once into the HF cache; this can take a few minutes")
    try:
        # The repo is a custom loading script (plant_village.py) that only defines a
        # "default" config (= color images) and needs trust_remote_code=True. It also
        # ships its own leaf-grouped train/test splits, so use split="all" to get
        # every image and let organize_datasets.py do our 80/10/10 split.
        dataset = load_dataset(PLANTVILLAGE_HF_ID, split="all", trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - report the real cause to the user
        sys.exit(f"[plantvillage] failed to load dataset: {exc}")

    label_col = "label" if "label" in dataset.column_names else dataset.column_names[-1]
    id_col = next((c for c in ("image_path", "image_id", "filename") if c in dataset.column_names), None)

    count, failed = 0, 0
    for i, row in enumerate(dataset):
        if sample_limit is not None and i >= sample_limit:
            break
        label = str(row[label_col]).replace("/", "_")
        class_dir = out / label
        class_dir.mkdir(parents=True, exist_ok=True)
        if id_col:
            name = f"{Path(str(row[id_col])).stem}.jpg"
        else:
            name = f"{label}_{i:06d}.jpg"
        try:
            image = row["image"]
            if isinstance(image, str):
                from PIL import Image
                image = Image.open(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(class_dir / name, "JPEG", quality=95)
            count += 1
        except Exception as exc:  # noqa: BLE001 - keep going on corrupt images
            failed += 1
            print(f"[plantvillage]  WARN: failed to save row {i} ({label}): {exc}")

    print(f"[plantvillage] saved {count} images to {out}" + (f" ({failed} skipped)" if failed else ""))
    return count


def download_plantdoc(data_dir: Path, force: bool) -> int:
    out = data_dir / "plantdoc" / "raw"
    existing = _count_images(out)
    if existing and not force:
        print(f"[plantdoc] {existing} images already present at {out}; skipping (use --force to re-clone)")
        return existing

    if not _check_case_sensitive():
        print(
            "[plantdoc] WARNING: this filesystem is case-insensitive; cloning may lose 6 image files.\n"
            "[plantdoc]          For the complete 2,578-image dataset, run this on Colab/Linux.",
            file=sys.stderr,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[plantdoc] cloning {PLANTDOC_REPO_URL} -> {out}")
    subprocess.run(["git", "clone", "--depth", "1", PLANTDOC_REPO_URL, str(out)], check=True)
    n = _count_images(out)
    print(f"[plantdoc] cloned {n} images to {out}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all", help="which dataset to download (default: all)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="root directory for datasets (default: data)")
    parser.add_argument("--force", action="store_true", help="re-download/re-clone even if data exists")
    parser.add_argument("--sample-limit", type=int, default=None, help="download only the first N PlantVillage images (dev/testing; re-run without it and with --force for the full set)")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("plantvillage", "all"):
        download_plantvillage(data_dir, args.sample_limit, args.force)
    if args.dataset in ("plantdoc", "all"):
        download_plantdoc(data_dir, args.force)

    print(f"\nDone. Datasets under {data_dir}/:")

    def _summarize(name: str) -> None:
        base = data_dir / name / "raw"
        total = _count_images(base)
        classes = len([p for p in base.iterdir() if p.is_dir()]) if base.exists() else 0
        print(f"  {name}: {total} images, {classes} class dirs  ({base})")

    if args.dataset in ("plantvillage", "all"):
        _summarize("plantvillage")
    if args.dataset in ("plantdoc", "all"):
        _summarize("plantdoc")


if __name__ == "__main__":
    main()
