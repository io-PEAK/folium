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
from .model import build_model, build_dual_head_model

CM_ANNOT_THRESHOLD_PERCENT = 2.0  # only annotate percent cells at least this significant

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
    parser.add_argument("--map-to-pv", action="store_true",
                        help="dataset='plantdoc': translate PlantDoc classes to the PlantVillage label "
                             "space via class_map.json so a 38-class model can be scored on PlantDoc "
                             "photos (the Sprint 4 cross-dataset gap measurement)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="'auto' | 'cuda' | 'cpu'")
    parser.add_argument("--results", type=Path, default=Path("results/ablation_results.csv"), help="ablation CSV to append to")
    parser.add_argument("--variant", default="baseline_pv_only_no_aug", help="experiment name written in the CSV row")
    parser.add_argument("--confusion-path", type=Path, default=None,
                        help="confusion matrix output PNG (default: <results-dir>/cm_<variant>.png, "
                             "so each sprint/variant keeps its own matrix)")
    parser.add_argument("--cm-style", choices=("raw", "percent"), default="percent",
                        help="confusion matrix rendering: 'percent' (row-normalized, annotated; "
                             "small errors stay visible despite class imbalance) or 'raw' (counts)")
    parser.add_argument("--tta", action="store_true",
                        help="test-time augmentation: average softmax over the image and its "
                             "horizontal flip. Bought ~1-3 F1 points on the small field set; keep "
                             "OFF for apples-to-apples ablation comparisons unless the whole table "
                             "is re-run with it")
    parser.add_argument("--dual-head", action="store_true",
                        help="Sprint 8: load a DualHeadModel checkpoint")
    parser.add_argument("--separate-backbones", action="store_true",
                        help="Sprint 9: DualHeadModel trained with separate backbones "
                             "(backbone_lab + backbone_field). Inferred from checkpoint if omitted.")
    parser.add_argument("--predict-mode", default="routed", choices=("routed", "dual"),
                        help="Sprint 8: 'routed' uses domain classifier to route to the "
                             "correct head (default); 'dual' uses confidence-race fallback")
    parser.add_argument("--eval-head", default=None, choices=("lab", "field"),
                        help="Sprint 8: evaluate only one head (lab or field) instead of "
                             "routing. Use this to measure each head's true score.")
    args = parser.parse_args()
    if args.confusion_path is None:
        stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in args.variant)
        args.confusion_path = args.results.parent / f"cm_{stem}.png"
    return args


@torch.no_grad()
def predict_all(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, tta: bool = False, dual_head: bool = False, eval_head: str | None = None, predict_mode: str = "routed"):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            if eval_head is not None:
                preds = model.predict_head(images, eval_head)
            elif dual_head and predict_mode == "routed":
                preds = model.predict_routed(images)
            elif dual_head and predict_mode == "dual":
                preds = model.predict_dual(images)
            else:
                probs = torch.softmax(model(images), dim=1)
                if tta:
                    probs = (probs + torch.softmax(model(torch.flip(images, dims=[3])), dim=1)) / 2
                preds = probs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
    return np.asarray(all_preds), np.asarray(all_labels)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    path: Path,
    style: str = "percent",
) -> None:
    """Render the confusion matrix to PNG.

    style="percent" (default): each actual-class row is normalized so a cell
    shows what % of that class was predicted as the column. This makes small
    error rates visible and classes comparable despite test-set imbalance.
    Cells >= CM_ANNOT_THRESHOLD_PERCENT are annotated (diagonal shows each
    class's recall; only meaningful confusions show off-diagonal).

    style="raw": raw image counts (original behavior, unannotated).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if style == "percent":
        row_sums = cm.sum(axis=1, keepdims=True)
        matrix = np.divide(
            cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
        ) * 100
        vmin, vmax, label = 0, 100, "% of actual class (row)"
    else:
        matrix = cm.astype(float)
        vmin = vmax = None
        label = "images"

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(matrix, cmap="Blues", vmin=vmin, vmax=vmax)
    if style == "percent":
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if v < CM_ANNOT_THRESHOLD_PERCENT:
                    continue
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 50 else "black")

    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(im, ax=ax, fraction=0.046, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Confusion matrix -> {path.resolve()} (style={style})")


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
        existing = pd.read_csv(path).drop_duplicates()
        # Skip re-logging the same run: key on the identity columns (not the
        # full row, whose timestamp differs every run). Re-running evaluate on
        # the same checkpoint/variant must not bloat the source-of-truth table.
        identity = ["variant", "dataset", "split", "checkpoint"]
        new_id = row[identity].astype(str).iloc[0]
        if (existing[identity].astype(str) == new_id).all(axis=1).any():
            print(f"Skipped duplicate ablation row (already logged) -> {path.resolve()}")
            return
        df = pd.concat([existing, row], ignore_index=True)
    else:
        df = row
    df.to_csv(path, index=False)
    print(f"Appended ablation row -> {path.resolve()}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if args.eval_head and not args.dual_head:
        raise ValueError("--eval-head requires --dual-head")

    if args.dual_head:
        separate = args.separate_backbones or ckpt["model_kwargs"].get("separate_backbones", False)
        model = build_dual_head_model(
            num_classes=ckpt["model_kwargs"]["num_classes"],
            backbone=ckpt["model_kwargs"]["backbone"],
            separate_backbones=separate,
        )
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"Note: missing keys (expected for older checkpoints): {missing}", flush=True)
        print(f"DualHeadModel loaded: {ckpt['model_kwargs']['backbone']} "
              f"(separate_backbones={separate})", flush=True)
    else:
        model = build_model(**ckpt["model_kwargs"])
        model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    train_loader, val_loader, test_loader, _ = build_loaders(
        args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        map_to_pv=args.map_to_pv,
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    class_names = ckpt["class_names"]
    if args.eval_head:
        print(f"Evaluating only head_{args.eval_head}", flush=True)
    elif args.dual_head:
        print(f"Dual-head predict_mode={args.predict_mode}", flush=True)
    preds, labels = predict_all(model, loader, device, tta=args.tta, dual_head=args.dual_head, eval_head=args.eval_head, predict_mode=args.predict_mode)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)

    print(f"\nResults on {args.dataset}/{args.split} ({len(labels)} images, {len(class_names)} classes):")
    print(f"  accuracy  {acc:.4f}")
    print(f"  precision {precision:.4f}")
    print(f"  recall    {recall:.4f}")
    print(f"  f1        {f1:.4f}")

    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    plot_confusion_matrix(cm, class_names, args.confusion_path, style=args.cm_style)

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
