"""Evaluation on a held-out split + ablation logging (see AGENT.md -> ml/evaluate.py).

Loads a checkpoint, runs inference on one split, prints accuracy /
precision / recall / F1 (macro) plus per-class detail, writes a confusion
matrix PNG, and APPENDS one row to the ablation CSV that the paper's Results
section is written from. Never fabricate: every number here comes from an
actual logged run.

Run from the repo root:

    python -m ml.evaluate --checkpoint <best_ckpt.pt> --data-dir data \
        --results results/ablation_results.csv --confusion-path results/confusion_matrix.png \
        --variant baseline_pv_only_no_aug
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from .data_loading import build_loaders
from .model import build_model

ABLATION_COLUMNS = [
    "variant",
    "dataset",
    "split",
    "backbone",
    "epochs",
    "classes",
    "test_images",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "timestamp",
    "checkpoint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a held-out split and log to the ablation CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="saved model checkpoint (.pt)")
    parser.add_argument("--data-dir", type=Path, required=True, help="root holding <dataset>/{train,val,test}")
    parser.add_argument("--dataset", default="plantvillage", help="dataset subfolder under --data-dir")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="'auto' | 'cuda' | 'cpu'")
    parser.add_argument("--results", type=Path, default=Path("results/ablation_results.csv"), help="ablation CSV to append to")
    parser.add_argument("--variant", default="baseline_pv_only_no_aug", help="experiment name written in the CSV row")
    parser.add_argument("--confusion-path", type=Path, default=Path("results/confusion_matrix.png"), help="confusion matrix output PNG")
    return parser.parse_args()


@torch.no_grad()
def predict_all(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(images)
        all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.tolist())
    return np.asarray(all_preds), np.asarray(all_labels)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Confusion matrix -> {path.resolve()}")


def log_ablation_row(
    path: Path,
    variant: str,
    dataset: str,
    split: str,
    checkpoint: Path,
    ckpt: dict,
    metrics: dict,
    n_test: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [[
            variant,
            f"{dataset}_{split}",
            split,
            ckpt["model_kwargs"].get("backbone", "?"),
            ckpt.get("epoch", "?"),
            len(ckpt.get("class_names", [])),
            n_test,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            str(checkpoint),
        ]],
        columns=ABLATION_COLUMNS,
    )
    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, row], ignore_index=True)
    else:
        df = row
    df.to_csv(path, index=False)
    print(f"Appended ablation row -> {path.resolve()}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    train_loader, val_loader, test_loader, _ = build_loaders(
        args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    class_names = ckpt["class_names"]
    preds, labels = predict_all(model, loader, device)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)

    print(f"\nResults on {args.dataset}/{args.split} ({len(labels)} images, {len(class_names)} classes):")
    print(f"  accuracy  {acc:.4f}")
    print(f"  precision {precision:.4f}")
    print(f"  recall    {recall:.4f}")
    print(f"  f1        {f1:.4f}")

    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    plot_confusion_matrix(cm, class_names, args.confusion_path)

    log_ablation_row(
        args.results,
        args.variant,
        args.dataset,
        args.split,
        args.checkpoint,
        ckpt,
        {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1},
        len(labels),
    )


if __name__ == "__main__":
    main()
