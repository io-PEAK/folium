#!/usr/bin/env python3
"""Download PlantVillage and PlantDoc datasets into <data-dir>/<name>/raw.

Runs identically on Google Colab and a local machine — no hardcoded paths,
everything is configurable via CLI args (see AGENT.md conventions).

    python scripts/download_datasets.py --dataset all --data-dir data

Datasets:
  - PlantVillage  GitHub "spMohanty/PlantVillage-Dataset" — sparse checkout of
                  "raw/color" only (~1 GB), so no grayscale/segmented data and
                  no full-repo clone. Uses the same color images as the Hugging
                  Face "mohanty/PlantVillage" mirror.
  - PlantDoc      GitHub "pratikkayal/PlantDoc-Dataset" (git clone)

The heavy PlantVillage checkout is done in --work-dir (session scratch on
Colab) and the final raw/color class folders are moved under --data-dir.
If a dataset already exists it is skipped (use --force to re-download).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATASETS = ("plantvillage", "plantdoc")

PLANTVILLAGE_GITHUB_REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
PLANTVILLAGE_SUBDIR = "raw/color"
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


def download_plantvillage(data_dir: Path, work_dir: Path, force: bool) -> int:
    out = data_dir / "plantvillage" / "raw"
    existing = _count_images(out)
    if existing and not force:
        print(f"[plantvillage] {existing} images already present at {out}; skipping (use --force to re-download)")
        return existing

    work_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = work_dir / "plantvillage_repo"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    print(f"[plantvillage] sparse-checking out '{PLANTVILLAGE_SUBDIR}' from {PLANTVILLAGE_GITHUB_REPO} ...")
    print("[plantvillage] this downloads only the ~1 GB of color images; a few minutes on Colab")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", PLANTVILLAGE_GITHUB_REPO, str(clone_dir)],
            check=True,
        )
        subprocess.run(["git", "-C", str(clone_dir), "sparse-checkout", "set", PLANTVILLAGE_SUBDIR], check=True)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(clone_dir, ignore_errors=True)
        sys.exit(f"[plantvillage] git checkout failed: {exc}")

    src = clone_dir / "raw" / "color"
    if not src.is_dir():
        shutil.rmtree(clone_dir, ignore_errors=True)
        sys.exit(f"[plantvillage] expected '{src}' not found after checkout; aborting")

    out.mkdir(parents=True, exist_ok=True)
    moved = 0
    for class_dir in sorted(src.iterdir()):
        if not class_dir.is_dir():
            continue
        dest = out / class_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(class_dir), str(dest))
        moved += 1
    shutil.rmtree(clone_dir, ignore_errors=True)

    count = _count_images(out)
    print(f"[plantvillage] moved {moved} class folders, {count} images -> {out}")
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
    parser.add_argument("--work-dir", type=Path, default=None, help="scratch directory for the ~1 GB PlantVillage checkout (default: --data-dir; on Colab use /content/folium_cache so Drive isn't cluttered)")
    parser.add_argument("--force", action="store_true", help="re-download/re-clone even if data exists")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    work_dir = (args.work_dir or args.data_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("plantvillage", "all"):
        download_plantvillage(data_dir, work_dir, args.force)
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
