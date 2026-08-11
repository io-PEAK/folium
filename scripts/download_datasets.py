#!/usr/bin/env python3
"""Download PlantVillage and PlantDoc datasets into <data-dir>/<name>/raw.

Runs identically on Google Colab and a local machine - no hardcoded paths,
everything is configurable via CLI args (see AGENT.md conventions).

    python scripts/download_datasets.py --dataset all \
        --data-dir /content/folium_raw \
        --work-dir /content/folium_cache \
        --upload-dir /content/drive/MyDrive/folium/data

Datasets:
  - PlantDoc      GitHub "pratikkayal/PlantDoc-Dataset" (small, ~2.6k images;
                  downloaded FIRST so a mid-run disconnect never strands it).
  - PlantVillage  GitHub "spMohanty/PlantVillage-Dataset" - sparse checkout of
                  "raw/color" only (~1 GB), no grayscale/segmented data.

Durable storage on Google Drive (why a zip):
  Colab writes to Drive through a FUSE mount. Creating ~54,000 individual
  image files on Drive in one run exceeds Google's per-day file-request quota
  (that is the "Drive quota exceeded" error, not free space), and the FUSE
  cache can also serve reads that were never committed to the backend. So:
    - all downloads/checkouts happen on LOCAL disk (--data-dir / --work-dir),
    - the raw data is shipped to Drive as ONE archive per dataset
      (<upload-dir>/<name>_raw.zip) with a manifest recording its sha256 and
      per-class counts, and
    - each session "hydrates" the local raw dir by unzipping that single file
      (drive operations: 2 file writes once, 2 file reads per session).
  Always re-verify in a FRESH session before trusting the data - see the
  durability-gate cell in notebooks/sprint0_data_exploration.ipynb.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DOWNLOAD_ORDER = ("plantdoc", "plantvillage")  # small, fast dataset first
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

PLANTVILLAGE_GITHUB_REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
PLANTVILLAGE_SUBDIR = "raw/color"
PLANTDOC_REPO_URL = "https://github.com/pratikkayal/PlantDoc-Dataset.git"

# Per-class image counts measured from the repo trees with `git ls-tree`
# (spMohanty/PlantVillage-Dataset @ master: raw/color; pratikkayal/PlantDoc-Dataset
# @ master: train/ + test/). Used to verify a download and to resume a partial
# one (only classes that differ are fetched). If the upstream repo changes,
# regenerate these with:
#   git ls-tree -r --name-only HEAD raw/color | cut -d/ -f3 | sort | uniq -c
PLANTVILLAGE_EXPECTED = {
    "Apple___Apple_scab": 630,
    "Apple___Black_rot": 621,
    "Apple___Cedar_apple_rust": 275,
    "Apple___healthy": 1645,
    "Blueberry___healthy": 1502,
    "Cherry_(including_sour)___healthy": 854,
    "Cherry_(including_sour)___Powdery_mildew": 1052,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 513,
    "Corn_(maize)___Common_rust_": 1192,
    "Corn_(maize)___healthy": 1162,
    "Corn_(maize)___Northern_Leaf_Blight": 985,
    "Grape___Black_rot": 1180,
    "Grape___Esca_(Black_Measles)": 1383,
    "Grape___healthy": 423,
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 1076,
    "Orange___Haunglongbing_(Citrus_greening)": 5507,
    "Peach___Bacterial_spot": 2297,
    "Peach___healthy": 360,
    "Pepper,_bell___Bacterial_spot": 997,
    "Pepper,_bell___healthy": 1478,
    "Potato___Early_blight": 1000,
    "Potato___healthy": 152,
    "Potato___Late_blight": 1000,
    "Raspberry___healthy": 371,
    "Soybean___healthy": 5090,
    "Squash___Powdery_mildew": 1835,
    "Strawberry___healthy": 456,
    "Strawberry___Leaf_scorch": 1109,
    "Tomato___Bacterial_spot": 2127,
    "Tomato___Early_blight": 1000,
    "Tomato___healthy": 1591,
    "Tomato___Late_blight": 1909,
    "Tomato___Leaf_Mold": 952,
    "Tomato___Septoria_leaf_spot": 1771,
    "Tomato___Spider_mites Two-spotted_spider_mite": 1676,
    "Tomato___Target_Spot": 1404,
    "Tomato___Tomato_mosaic_virus": 373,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 5357,
}

PLANTDOC_EXPECTED = {
    "train": {
        "Apple leaf": 82,
        "Apple rust leaf": 79,
        "Apple Scab Leaf": 83,
        "Bell_pepper leaf": 53,
        "Bell_pepper leaf spot": 62,
        "Blueberry leaf": 106,
        "Cherry leaf": 47,
        "Corn Gray leaf spot": 64,
        "Corn leaf blight": 180,
        "Corn rust leaf": 106,
        "grape leaf": 57,
        "grape leaf black rot": 56,
        "Peach leaf": 103,
        "Potato leaf early blight": 109,
        "Potato leaf late blight": 97,
        "Raspberry leaf": 112,
        "Soyabean leaf": 57,
        "Squash Powdery mildew leaf": 124,
        "Strawberry leaf": 88,
        "Tomato Early blight leaf": 79,
        "Tomato leaf": 55,
        "Tomato leaf bacterial spot": 101,
        "Tomato leaf late blight": 101,
        "Tomato leaf mosaic virus": 44,
        "Tomato leaf yellow virus": 70,
        "Tomato mold leaf": 85,
        "Tomato Septoria leaf spot": 140,
        "Tomato two spotted spider mites leaf": 2,
    },
    "test": {
        "Apple leaf": 9,
        "Apple rust leaf": 10,
        "Apple Scab Leaf": 10,
        "Bell_pepper leaf": 8,
        "Bell_pepper leaf spot": 9,
        "Blueberry leaf": 11,
        "Cherry leaf": 10,
        "Corn Gray leaf spot": 4,
        "Corn leaf blight": 12,
        "Corn rust leaf": 10,
        "grape leaf": 12,
        "grape leaf black rot": 8,
        "Peach leaf": 9,
        "Potato leaf early blight": 8,
        "Potato leaf late blight": 8,
        "Raspberry leaf": 7,
        "Soyabean leaf": 8,
        "Squash Powdery mildew leaf": 6,
        "Strawberry leaf": 8,
        "Tomato Early blight leaf": 9,
        "Tomato leaf": 8,
        "Tomato leaf bacterial spot": 9,
        "Tomato leaf late blight": 10,
        "Tomato leaf mosaic virus": 10,
        "Tomato leaf yellow virus": 6,
        "Tomato mold leaf": 6,
        "Tomato Septoria leaf spot": 11,
    },
}


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _class_counts(root: Path) -> dict:
    if not root.exists():
        return {}
    return {d.name: _count_images(d) for d in sorted(root.iterdir()) if d.is_dir()}


def _plantdoc_counts(raw_root: Path) -> dict:
    return {
        "train": _class_counts(raw_root / "train"),
        "test": _class_counts(raw_root / "test"),
    }


def _sum_counts(classes: dict) -> int:
    total = 0
    for v in classes.values():
        total += sum(v.values()) if isinstance(v, dict) else v
    return total


def _manifest_path(data_dir: Path, dataset: str) -> Path:
    return data_dir / dataset / "manifest.json"


def _write_manifest(data_dir: Path, dataset: str, expected: dict, counts: dict) -> None:
    payload = {
        "dataset": dataset,
        "expected_total": _sum_counts(expected),
        "classes": counts,
        "total": _sum_counts(counts),
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = _manifest_path(data_dir, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[{dataset}] manifest written -> {path}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _counts_for(name: str, raw_dir: Path) -> dict:
    return _class_counts(raw_dir) if name == "plantvillage" else _plantdoc_counts(raw_dir)


def _zip_paths(upload_dir: Path, name: str) -> tuple:
    return upload_dir / f"{name}_raw.zip", upload_dir / f"{name}_raw.manifest.json"


def _zip_verified(upload_dir: Path, name: str):
    """Return the upload manifest if <upload-dir>/<name>_raw.zip is intact (sha256 match)."""
    if upload_dir is None:
        return None
    zip_path, manifest_path = _zip_paths(upload_dir, name)
    if not (zip_path.exists() and manifest_path.exists()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("zip_sha256") != _sha256(zip_path):
        return None
    if manifest.get("zip_bytes") != zip_path.stat().st_size:
        return None
    return manifest


def _upload_zip(upload_dir: Path, name: str, raw_dir: Path, expected: dict) -> dict:
    """Zip <raw_dir> and copy the single archive + manifest to <upload_dir>."""
    data_root = raw_dir.parent.parent
    zip_path = data_root / f"{name}_raw.zip"
    print(f"[{name}] creating archive {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(raw_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(raw_dir).as_posix())

    sha = _sha256(zip_path)
    dest_zip, dest_manifest = _zip_paths(upload_dir, name)
    upload_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] uploading single archive ({zip_path.stat().st_size / 1e6:.0f} MB) to Drive ...")
    try:
        shutil.copy2(zip_path, dest_zip)
    except OSError as exc:
        print(f"[{name}] WARNING: upload to Drive failed ({exc}); data is local only for now", file=sys.stderr)
        return {}

    manifest = {
        "dataset": name,
        "expected_total": _sum_counts(expected),
        "classes": _counts_for(name, raw_dir),
        "total": _count_images(raw_dir),
        "zip_sha256": sha,
        "zip_bytes": dest_zip.stat().st_size,
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    dest_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[{name}] uploaded {dest_zip} + manifest (sha256 {sha[:12]}...)")
    return manifest


def hydrate_dataset(data_dir: Path, upload_dir: Path, name: str, expected: dict) -> None:
    """Unzip <upload-dir>/<name>_raw.zip into <data-dir>/<name>/raw if needed.

    Skips when the local raw dir already matches the expected counts. Raises if
    there is no verified archive on Drive.
    """
    raw = data_dir / name / "raw"
    if raw.exists() and _counts_for(name, raw) == expected:
        print(f"[{name}] local raw already hydrated at {raw}; skipping")
        return

    manifest = _zip_verified(upload_dir, name)
    if manifest is None:
        raise RuntimeError(
            f"[{name}] no verified {name}_raw.zip on Drive - run download_datasets.py --upload-dir first"
        )

    zip_path = upload_dir / f"{name}_raw.zip"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True)
    print(f"[{name}] unzipping {zip_path} -> {raw} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw)

    counts = _counts_for(name, raw)
    if counts != expected:
        raise RuntimeError(f"[{name}] hydrated counts differ from expected")
    print(f"[{name}] hydrated {_count_images(raw)} images -> {raw}")


def _check_case_sensitive() -> bool:
    """Return True if the filesystem distinguishes filename case.

    The PlantDoc repo contains 6 pairs of files whose names differ only in
    case (e.g. CAR1.jpg / car1.jpg). Cloning on a case-insensitive filesystem
    (macOS APFS/HFS+, Windows NTFS) silently drops one file per pair, so the
    clone will not match PLANTDOC_EXPECTED there. Colab/Linux keeps all files.
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


def download_plantdoc(data_dir: Path, work_dir: Path, force: bool, upload_dir: Path | None) -> int:
    out = data_dir / "plantdoc" / "raw"
    counts = _plantdoc_counts(out)
    remote = _zip_verified(upload_dir, "plantdoc")

    if not force and remote is not None:
        print(f"[plantdoc] verified archive already on Drive ({remote['total']} images); skipping download")
        return remote["total"]
    if not force and counts == PLANTDOC_EXPECTED:
        print(f"[plantdoc] local raw already verified ({_count_images(out)} images); skipping download")
        if upload_dir is not None:
            _upload_zip(upload_dir, "plantdoc", out, PLANTDOC_EXPECTED)
        return _count_images(out)

    if not _check_case_sensitive():
        print(
            "[plantdoc] WARNING: this filesystem is case-insensitive; the clone will not match\n"
            "[plantdoc]          PLANTDOC_EXPECTED (6 files lost). Run on Colab/Linux for the full set.",
            file=sys.stderr,
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = work_dir / "plantdoc_repo"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    print(f"[plantdoc] cloning {PLANTDOC_REPO_URL} -> {clone_dir}, then moving into data dir")
    try:
        subprocess.run(["git", "clone", "--depth", "1", PLANTDOC_REPO_URL, str(clone_dir)], check=True)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(clone_dir, ignore_errors=True)
        sys.exit(f"[plantdoc] git clone failed: {exc}")

    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(clone_dir), str(out))

    counts = _plantdoc_counts(out)
    _write_manifest(data_dir, "plantdoc", PLANTDOC_EXPECTED, counts)
    if counts != PLANTDOC_EXPECTED:
        print("[plantdoc] WARNING: counts differ from expected; re-run with --force if this persists")
    if upload_dir is not None:
        _upload_zip(upload_dir, "plantdoc", out, PLANTDOC_EXPECTED)
    print(f"[plantdoc] {_count_images(out)} images -> {out}")
    return _count_images(out)


def download_plantvillage(data_dir: Path, work_dir: Path, force: bool, upload_dir: Path | None) -> int:
    out = data_dir / "plantvillage" / "raw"
    on_disk = _class_counts(out)
    remote = _zip_verified(upload_dir, "plantvillage")

    if not force and remote is not None:
        print(f"[plantvillage] verified archive already on Drive ({remote['total']} images); skipping download")
        return remote["total"]
    if not force and on_disk == PLANTVILLAGE_EXPECTED:
        print(f"[plantvillage] local raw already verified ({_sum_counts(on_disk)} images); skipping download")
        if upload_dir is not None:
            _upload_zip(upload_dir, "plantvillage", out, PLANTVILLAGE_EXPECTED)
        return _sum_counts(on_disk)

    needed = [c for c, want in PLANTVILLAGE_EXPECTED.items() if on_disk.get(c, 0) != want]
    print(f"[plantvillage] {len(on_disk)} classes on disk; {len(needed)} missing/incomplete:")
    for c in needed:
        print(f"[plantvillage]    {c}: {on_disk.get(c, 0)} != {PLANTVILLAGE_EXPECTED[c]}")

    work_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = work_dir / "plantvillage_repo"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    print(f"[plantvillage] sparse-checking out '{PLANTVILLAGE_SUBDIR}' from {PLANTVILLAGE_GITHUB_REPO} ...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", PLANTVILLAGE_GITHUB_REPO, str(clone_dir)],
            check=True,
        )
        paths = [PLANTVILLAGE_SUBDIR] if len(needed) == len(PLANTVILLAGE_EXPECTED) else [f"{PLANTVILLAGE_SUBDIR}/{c}" for c in needed]
        subprocess.run(["git", "-C", str(clone_dir), "sparse-checkout", "set", *paths], check=True)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(clone_dir, ignore_errors=True)
        sys.exit(f"[plantvillage] git checkout failed: {exc}")

    src = clone_dir / "raw" / "color"
    if not src.is_dir():
        shutil.rmtree(clone_dir, ignore_errors=True)
        sys.exit(f"[plantvillage] expected '{src}' not found after checkout; aborting")

    out.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for cls in needed:
        cls_src = src / cls
        if not cls_src.is_dir():
            print(f"[plantvillage]  WARN: class '{cls}' missing after checkout; skipping")
            continue
        dest = out / cls
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(cls_src), str(dest))
        fetched += 1
    shutil.rmtree(clone_dir, ignore_errors=True)

    final = _class_counts(out)
    _write_manifest(data_dir, "plantvillage", PLANTVILLAGE_EXPECTED, final)
    if final != PLANTVILLAGE_EXPECTED:
        print("[plantvillage] WARNING: final counts still differ from expected - the download was partial")
    if upload_dir is not None:
        _upload_zip(upload_dir, "plantvillage", out, PLANTVILLAGE_EXPECTED)
    total = _sum_counts(final)
    print(f"[plantvillage] fetched {fetched} classes, {total} images -> {out}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("plantdoc", "plantvillage", "all"), default="all", help="which dataset to download (default: all, small dataset first)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="LOCAL root directory for raw datasets (default: data; on Colab use /content/folium_raw)")
    parser.add_argument("--work-dir", type=Path, default=None, help="scratch directory for clones/checkouts (default: --data-dir; on Colab use /content/folium_cache)")
    parser.add_argument("--upload-dir", type=Path, default=None, help="Drive directory to store one <name>_raw.zip per dataset (2 files total, avoids Drive file-request quotas); omit to keep data local only")
    parser.add_argument("--force", action="store_true", help="re-download/re-clone even if data is verified")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    work_dir = (args.work_dir or args.data_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = args.upload_dir.resolve() if args.upload_dir else None
    if upload_dir is not None:
        upload_dir.mkdir(parents=True, exist_ok=True)

    to_run = DOWNLOAD_ORDER if args.dataset == "all" else (args.dataset,)
    for ds in to_run:
        if ds == "plantdoc":
            download_plantdoc(data_dir, work_dir, args.force, upload_dir)
        else:
            download_plantvillage(data_dir, work_dir, args.force, upload_dir)

    print(f"\nDone. Raw datasets under {data_dir}/:")

    def _summarize(name: str) -> None:
        base = data_dir / name / "raw"
        total = _count_images(base)
        if upload_dir is not None:
            m = _zip_verified(upload_dir, name)
            drive = f", Drive archive {'verified (' + str(m['total']) + ' images)' if m else 'MISSING'}"
        else:
            drive = ""
        print(f"  {name}: {total} images, {len([p for p in base.iterdir() if p.is_dir()]) if base.exists() else 0} class dirs{drive}  ({base})")

    for ds in to_run:
        _summarize(ds)


if __name__ == "__main__":
    main()
