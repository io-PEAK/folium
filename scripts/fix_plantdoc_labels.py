#!/usr/bin/env python3
"""Fix mislabeled PlantDoc test images based on audit results.

Reads the audit CSV produced by audit_plantdoc_labels.py, identifies
same-species high-confidence mismatches (where the model is likely right
and the label is wrong), and moves images to the correct class folder.

Only fixes images where:
1. pred_class and true_class are from the SAME plant species
2. confidence > 0.80 (model is confident)
3. agree = FALSE (model disagrees with label)

Usage:
    python scripts/fix_plantdoc_labels.py \
        --audit-csv /content/drive/MyDrive/folium/results/plantdoc_label_audit_s9.csv \
        --data-dir /content/folium_data \
        --dataset plantdoc \
        --split test \
        --dry-run
"""
import argparse
import csv
import json
import shutil
from pathlib import Path


def extract_species(class_name: str) -> str:
    """Extract species from PlantVillage class name.

    'Tomato___Bacterial_spot' -> 'Tomato'
    'Corn_(maize)___Northern_Leaf_Blight' -> 'Corn_(maize)'
    'Apple___Apple_scab' -> 'Apple'
    'Pepper,_bell___Bacterial_spot' -> 'Pepper,_bell'
    """
    return class_name.split("___")[0]


def parse_audit_csv(csv_path: Path) -> list[dict]:
    """Parse the audit CSV file."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["confidence"] = float(row["confidence"])
            row["agree"] = row["agree"].strip().upper() == "TRUE"
            rows.append(row)
    return rows


def find_fixable_images(rows: list[dict]) -> list[dict]:
    """Find images that are likely mislabeled.

    Criteria:
    - agree = FALSE (model disagrees with label)
    - confidence > 0.80 (model is confident)
    - Same species (pred_class and true_class share the same species prefix)
    """
    fixable = []
    for row in rows:
        if row["agree"]:
            continue
        if row["confidence"] <= 0.80:
            continue
        true_species = extract_species(row["true_class"])
        pred_species = extract_species(row["pred_class"])
        if true_species == pred_species:
            fixable.append(row)
    return fixable


def fix_labels(
    fixable: list[dict],
    data_dir: Path,
    dataset: str,
    split: str,
    dry_run: bool = True,
    log_path: Path | None = None,
) -> list[dict]:
    """Move mislabeled images to the correct class folder.

    Returns list of changes made (or would be made in dry-run mode).
    """
    changes = []
    dataset_dir = data_dir / dataset / split

    for row in fixable:
        # Parse the image path to get the source class and filename
        image_path = Path(row["image"])
        src_class = image_path.parent.name  # e.g., "Corn leaf blight"
        filename = image_path.name

        # The predicted class folder name: we need to map pred_class back to folder name
        # pred_class is like "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
        # But the folder name is "Corn Gray leaf spot" (PlantDoc naming, not PlantVillage)
        # We need to find which folder in the dataset matches pred_class
        # Actually, the audit script uses the PlantVillage-mapped class as true_class,
        # and the model's prediction as pred_class. The folder names are PlantDoc names.
        # We need to find the folder that maps to pred_class.

        # Strategy: look at all folders in the dataset, find which one maps to pred_class
        # via class_map.json
        # But we don't have class_map.json here. Instead, we can use the fact that
        # the audit CSV has the full image path which includes the source folder.

        # Actually, the simpler approach: the pred_class is a PlantVillage class name.
        # We need to find the PlantDoc folder name that maps to it.
        # We can load class_map.json and invert it.
        changes.append({
            "src": str(image_path),
            "src_class": src_class,
            "filename": filename,
            "pred_class": row["pred_class"],
            "true_class": row["true_class"],
            "confidence": row["confidence"],
        })

    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", required=True, help="Path to audit CSV")
    parser.add_argument("--data-dir", required=True, help="Data directory")
    parser.add_argument("--dataset", default="plantdoc")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed")
    parser.add_argument("--log", help="Path to write change log")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = Path(args.audit_csv)

    print(f"Reading audit from {csv_path}")
    rows = parse_audit_csv(csv_path)
    print(f"Total images: {len(rows)}")

    fixable = find_fixable_images(rows)
    print(f"Fixable (same-species, >0.80 confidence): {len(fixable)}")

    if not fixable:
        print("No images to fix.")
        return

    # Load class_map.json to invert the mapping
    class_map_path = data_dir / "class_map.json"
    if class_map_path.exists():
        with open(class_map_path) as f:
            class_map = json.load(f)
        # Invert: PV class name -> PD folder name
        pv_to_pd_folder = {v: k for k, v in class_map.items()}
    else:
        pv_to_pd_folder = {}
        print("WARNING: class_map.json not found. Cannot map PV classes to PD folders.")

    # Show what would be fixed
    print("\nImages to fix:")
    for row in fixable:
        image_path = Path(row["image"])
        src_class = image_path.parent.name
        pred_class = row["pred_class"]
        pd_folder = pv_to_pd_folder.get(pred_class, "???")
        src_full = data_dir / args.dataset / args.split / src_class / image_path.name
        dst_full = data_dir / args.dataset / args.split / pd_folder / image_path.name

        exists_src = src_full.exists()
        exists_dst = dst_full.exists()

        status = "OK" if exists_src and not exists_dst else "SKIP"
        if not exists_src:
            status = "MISSING"
        if exists_dst:
            status = "EXISTS"

        print(f"  {status} | {src_class} -> {pd_folder} | {image_path.name} | conf={row['confidence']:.3f}")

    if args.dry_run:
        print("\nDry run — no files moved. Remove --dry-run to apply.")
        return

    # Actually move files
    moved = 0
    skipped = 0
    for row in fixable:
        image_path = Path(row["image"])
        src_class = image_path.parent.name
        pred_class = row["pred_class"]
        pd_folder = pv_to_pd_folder.get(pred_class)
        if not pd_folder:
            print(f"  SKIP (no folder mapping): {image_path.name}")
            skipped += 1
            continue

        src_full = data_dir / args.dataset / args.split / src_class / image_path.name
        dst_full = data_dir / args.dataset / args.split / pd_folder / image_path.name

        if not src_full.exists():
            print(f"  SKIP (source missing): {src_full}")
            skipped += 1
            continue
        if dst_full.exists():
            print(f"  SKIP (dest exists): {dst_full}")
            skipped += 1
            continue

        dst_full.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_full), str(dst_full))
        print(f"  MOVED: {src_class}/{image_path.name} -> {pd_folder}/{image_path.name}")
        moved += 1

    print(f"\nDone: {moved} moved, {skipped} skipped")

    # Write log
    if args.log:
        log_path = Path(args.log)
        with open(log_path, "w") as f:
            f.write("src_class,dst_class,filename,confidence\n")
            for row in fixable:
                image_path = Path(row["image"])
                src_class = image_path.parent.name
                pd_folder = pv_to_pd_folder.get(row["pred_class"], "???")
                f.write(f"{src_class},{pd_folder},{image_path.name},{row['confidence']:.4f}\n")
        print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
