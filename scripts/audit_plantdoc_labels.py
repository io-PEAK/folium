#!/usr/bin/env python3
"""Audit PlantDoc labels against a field-strong model (see AGENT.md).

Sprint 7 lever: past ~0.60 field F1, the cheapest win is better labels, not more
training. PlantDoc is crowd-sourced and known to contain mislabeled / ambiguous
images; those cap every model. This runs a checkpoint over a PlantDoc split and
writes each image with its mapped PlantVillage class, predicted class, and
confidence, sorted least-confident first, so suspected mislabels are easy to
eyeball and relabel or drop.

The field-strong checkpoints (`both` / `both_resnet50` / `both_efficientnet`)
are the usual inputs because they are committed to the field domain - their
disagreements with the ground-truth label are the ones worth looking at.

Run from the repo root (matches the other CLI docs):

    python scripts/audit_plantdoc_labels.py \\
        --checkpoint checkpoints/best_plantdoc_stage2_aug.pt \\
        --data-dir data --split train \\
        --out results/plantdoc_label_audit.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from ml.data_loading import _MappedImageFolder, _load_class_map, make_transforms
from ml.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score every PlantDoc image with a checkpoint; flag likely mislabels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="field-strong model checkpoint (.pt)")
    parser.add_argument("--data-dir", type=Path, required=True, help="root holding plantvillage/ and plantdoc/ splits")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"), help="PlantDoc split to audit")
    parser.add_argument("--out", type=Path, default=Path("results/plantdoc_label_audit.csv"),
                        help="CSV output (sorted least-confident first)")
    parser.add_argument("--top", type=int, default=25, help="how many most-suspicious images to print")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="'auto' | 'cuda' | 'cpu'")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_names = ckpt["class_names"]
    model = build_model(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    class_map = _load_class_map(args.data_dir)
    pv_names = datasets.ImageFolder(str(args.data_dir / "plantvillage" / "train")).classes
    mapped = _MappedImageFolder(
        args.data_dir / "plantdoc" / args.split,
        class_map,
        pv_names,
        transform=make_transforms(),
    )
    loader = DataLoader(mapped, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    preds, probs = [], []
    for images, _ in loader:
        images = images.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            sm = torch.softmax(model(images), dim=1)
        preds.extend(sm.argmax(dim=1).cpu().tolist())
        probs.extend(sm.max(dim=1).values.cpu().tolist())

    rows = []
    for (path, true_label), pred, prob in zip(mapped.samples, preds, probs):
        rows.append({
            "image": str(path),
            "plantdoc_class": Path(path).parent.name,
            "true_class": class_names[true_label],
            "pred_class": class_names[pred],
            "confidence": prob,
            "agree": pred == true_label,
        })
    rows.sort(key=lambda r: r["confidence"])

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    n_disagree = sum(1 for r in rows if not r["agree"])
    print(f"Audited {n} PlantDoc/{args.split} images with {ckpt['model_kwargs'].get('backbone', '?')}")
    print(f"  model disagrees with the label on {n_disagree} ({n_disagree / n:.1%}); "
          f"mean confidence on agreements {sum(r['confidence'] for r in rows if r['agree']) / max(n - n_disagree, 1):.3f}")
    print(f"\nTop {min(args.top, n)} most-suspicious (lowest confidence) -> {out_path}")
    for r in rows[: min(args.top, n)]:
        flag = "OK" if r["agree"] else "SUSPECT"
        print(f"  [{flag}] {r['confidence']:.3f}  {r['plantdoc_class']} (map-> {r['true_class']})  "
              f"predicted {r['pred_class']}")


if __name__ == "__main__":
    main()
