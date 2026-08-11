"""Stage 1 training on PlantVillage (see AGENT.md -> ml/train.py, Sprint 1).

Freezes the pretrained backbone and trains only the new classification head,
no augmentation yet. Uses Adam + mixed precision on CUDA, saves a checkpoint
every epoch plus a best-on-validation copy so a dropped Colab session never
loses everything.

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
from .model import build_model, trainable_parameters

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
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (head only)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--image-size", type=int, default=224, help="center-crop size fed to the model")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="where checkpoints are saved")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to resume training from")
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
    )
    model_kwargs = {"num_classes": len(class_names), "backbone": "mobilenet_v2", "pretrained": True, "freeze": True}
    model = build_model(**model_kwargs).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
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

    print(f"Training head-only on {len(class_names)} classes, {args.epochs} epochs, lr={args.lr}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"epoch {epoch:02d}/{args.epochs}: train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}", flush=True)

        save_checkpoint(
            args.checkpoint_dir / f"{args.dataset}_stage1_epoch{epoch:02d}.pt",
            model, optimizer, scaler, epoch, val_acc, class_names, model_kwargs,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                args.checkpoint_dir / f"best_{args.dataset}_stage1.pt",
                model, optimizer, scaler, epoch, val_acc, class_names, model_kwargs,
            )

    print(f"Done. Best val acc {best_val_acc:.4f}; checkpoints in {args.checkpoint_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
