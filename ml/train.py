"""Training (see AGENT.md -> ml/train.py).

Stage 1 (Sprint 1/3): freezes the pretrained backbone and trains only the new
classification head, optionally with augmentation. Stage 2 (Sprint 4): warm-starts
from a Stage 1 checkpoint (--init-from), unfreezes the last backbone blocks
(--unfreeze-blocks) and fine-tunes at a low --lr with the head at --head-lr,
optionally on PlantDoc mapped into the PlantVillage label space (--map-to-pv).

Uses Adam + mixed precision on CUDA, saves a checkpoint every epoch plus a
best-on-validation copy so a dropped Colab session never loses everything.

Run from the repo root so the package imports resolve:

    python -m ml.train --data-dir data \
        --checkpoint-dir /content/drive/MyDrive/folium/checkpoints \
        --epochs 5
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .data_loading import build_loaders
from .model import build_model, unfreeze_last_blocks

DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 (head-only) training on PlantVillage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="root holding <dataset>/{train,val,test}")
    parser.add_argument("--dataset", default="plantvillage", help="dataset subfolder under --data-dir")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="training/validation batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (backbone params; head uses --head-lr if set)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="where checkpoints are saved")
    parser.add_argument("--tag", default="stage1", help="artifact tag used in checkpoint filenames, e.g. stage1 (baseline) or stage1_aug")
    parser.add_argument("--augment", action="store_true", help="apply the Sprint 3 albumentations pipeline to the training split")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to resume training from")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="load only the model weights from this checkpoint as a warm start (Stage 2 "
                             "fine-tuning from a Stage 1 model); starts fresh epochs/optimizer")
    parser.add_argument("--unfreeze-blocks", type=int, default=0,
                        help="Stage 2: re-enable gradients on the last N parameter-bearing backbone "
                             "modules (train them at --lr alongside the head)")
    parser.add_argument("--head-lr", type=float, default=None,
                        help="learning rate for the classification head (defaults to --lr; give it a "
                             "higher value when the backbone is being fine-tuned at low --lr)")
    parser.add_argument("--map-to-pv", action="store_true",
                        help="dataset='plantdoc': read PlantDoc but translate its classes to the "
                             "PlantVillage label space via class_map.json (Sprint 4 cross-dataset use)")
    parser.add_argument("--device", default="auto", help="'auto' | 'cuda' | 'cpu'")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _cuda_autocast():
    """Autocast context manager for mixed precision (no-op on CPU)."""
    return torch.amp.autocast("cuda", enabled=torch.cuda.is_available())


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with _cuda_autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with _cuda_autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    val_acc: float,
    class_names: list[str],
    model_kwargs: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kwargs": model_kwargs,
            "class_names": class_names,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = _resolve_device(args.device)
    if device.type == "cuda":
        print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("CPU only (mixed precision disabled)", flush=True)

    train_loader, val_loader, _, class_names = build_loaders(
        args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=args.augment,
        map_to_pv=args.map_to_pv,
    )
    model_kwargs = {"num_classes": len(class_names), "backbone": "mobilenet_v2", "pretrained": True, "freeze": True}
    model = build_model(**model_kwargs).to(device)

    if args.init_from is not None:
        init_ckpt = torch.load(args.init_from, map_location="cpu")
        if init_ckpt["class_names"] != class_names:
            raise ValueError(
                f"init checkpoint classes ({len(init_ckpt['class_names'])}) differ from "
                f"current dataset classes ({len(class_names)})"
            )
        model.load_state_dict(init_ckpt["state_dict"])
        print(f"Warm-started weights from {args.init_from}", flush=True)

    if args.unfreeze_blocks > 0:
        unfreeze_last_blocks(model, args.unfreeze_blocks)
        print(f"Unfroze last {args.unfreeze_blocks} parameter-bearing backbone modules", flush=True)

    criterion = nn.CrossEntropyLoss()
    head_params = list(model.classifier.parameters())
    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and all(id(p) != id(h) for h in head_params)
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": args.head_lr or args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Trainable modules: backbone {len(backbone_params)} params, head {len(head_params)} params "
          f"({n_trainable} tensors total)", flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch, best_val_acc = 1, 0.0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"Resumed from {args.resume} at epoch {ckpt['epoch']} (best val acc {best_val_acc:.4f})")

    mode = "fine-tuning" if args.unfreeze_blocks > 0 else "head-only"
    print(f"Training {mode} on {len(class_names)} classes, {args.epochs} epochs, "
          f"lr={args.lr}, head_lr={args.head_lr or args.lr}, "
          f"augmentation={'ON' if args.augment else 'OFF'}, tag={args.tag}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"epoch {epoch:02d}/{args.epochs}: train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}", flush=True)

        save_checkpoint(
            args.checkpoint_dir / f"{args.dataset}_{args.tag}_epoch{epoch:02d}.pt",
            model, optimizer, scaler, epoch, val_acc, class_names, model_kwargs,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                args.checkpoint_dir / f"best_{args.dataset}_{args.tag}.pt",
                model, optimizer, scaler, epoch, val_acc, class_names, model_kwargs,
            )

    print(f"Done. Best val acc {best_val_acc:.4f}; checkpoints in {args.checkpoint_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
