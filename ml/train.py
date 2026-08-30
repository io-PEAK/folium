"""Training (see AGENT.md -> ml/train.py).

Stage 1 (Sprint 1/3): freezes the pretrained backbone and trains only the new
classification head, optionally with augmentation. Stage 2 (Sprint 4): warm-starts
from a Stage 1 checkpoint (--init-from), unfreezes the last backbone blocks
(--unfreeze-blocks) and fine-tunes at a low --lr with the head at --head-lr,
optionally on PlantDoc mapped into the PlantVillage label space (--map-to-pv).
Sprint 5: --mix-with plantdoc trains PlantVillage + PlantDoc together in every
epoch (concatenated loaders), so the model keeps both domains instead of
forgetting PlantVillage. Sprint 6: --plantdoc-repeat N repeats the PlantDoc
train set inside the mixed loader to give field photos a bigger epoch share.
Sprint 7 (the backbone sweep): --backbone resnet50/efficientnet_b0 runs the same
Sprint 4 + fine-tune-then-mix recipes on stronger feature extractors to push the
field-photo F1 past MobileNetV2's 0.5578 ceiling toward 0.60+, keeping a single
self-contained model (v13 warm-starts from the field-strong Stage-2 checkpoint
via --init-from, then --mix-with plantdoc re-learns the lab).

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

from .data_loading import build_loaders, build_domain_loaders
from .model import MODEL_FEATURE_DIMS, build_model, build_dual_head_model, DualHeadModel, head_module, unfreeze_last_blocks

DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 (head-only) training on PlantVillage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="root holding <dataset>/{train,val,test}")
    parser.add_argument("--dataset", default="plantvillage", help="dataset subfolder under --data-dir")
    parser.add_argument("--backbone", default="mobilenet_v2", choices=sorted(MODEL_FEATURE_DIMS),
                        help="feature extractor; Sprint 7 sweep uses resnet50/efficientnet_b0 to push "
                             "the field-photo F1 ceiling past MobileNetV2's 0.5578")
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
    parser.add_argument("--mix-with", default=None,
                        help="Sprint 5: train --dataset together with this second dataset in every "
                             "epoch (mix_with='plantdoc' = PlantVillage + PlantDoc concatenated into "
                             "one loader; the fix for catastrophic forgetting). Requires "
                             "dataset='plantvillage'.")
    parser.add_argument("--plantdoc-repeat", type=int, default=1,
                        help="Sprint 6: repeat the PlantDoc train set N times inside the mixed "
                             "loader so field photos get a bigger share of every epoch (PlantDoc is "
                             "naturally ~5 percent; e.g. 8 -> ~28 percent). Only used with --mix-with plantdoc.")
    parser.add_argument("--class-balanced", action="store_true",
                        help="Sprint 12: sample the training set class-balanced (WeightedRandomSampler "
                             "with 1/class-frequency weights) instead of uniform shuffle. Calibrates "
                             "each constituent (lab + field) separately, so minority classes in the "
                             "imbalanced PlantDoc-in-mixed set get a fair share every epoch.")
    parser.add_argument("--dual-head", action="store_true",
                        help="Sprint 8: use DualHeadModel with two independent classification "
                             "heads (head_lab + head_field) instead of a single shared head. "
                             "Eliminates catastrophic forgetting by keeping each domain separate.")
    parser.add_argument("--separate-backbones", action="store_true",
                        help="Sprint 9: give each head its OWN backbone (backbone_lab + "
                             "backbone_field) instead of one shared backbone. Eliminates the "
                             "shared-backbone interference that capped the field head at 0.41.")
    parser.add_argument("--train-head", default=None, choices=("lab", "field", "domain"),
                        help="With --dual-head: which component to train "
                             "(lab=PlantVillage head, field=PlantDoc head, "
                             "domain=domain classifier on both datasets).")
    parser.add_argument("--backbone-lr", type=float, default=0.0,
                        help="Sprint 8: learning rate for backbone params during dual-head training. "
                             "0=frozen (default); 1e-5 to unfreeze backbone at low LR for domain adaptation.")
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
    active_head: str | None = None,
) -> tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with _cuda_autocast():
            raw_out = model(images)
            if active_head is not None:
                outputs = raw_out[0] if active_head == "lab" else raw_out[1]
            else:
                outputs = raw_out
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
    active_head: str | None = None,
) -> tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with _cuda_autocast():
            raw_out = model(images)
            if active_head is not None:
                outputs = raw_out[0] if active_head == "lab" else raw_out[1]
            else:
                outputs = raw_out
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


def train_domain_classifier(args: argparse.Namespace) -> None:
    """Train the domain classifier on PlantVillage (lab) + PlantDoc (field)."""
    device = _resolve_device(args.device)
    if device.type == "cuda":
        print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("CPU only (mixed precision disabled)", flush=True)

    train_loader, val_loader = build_domain_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=args.augment,
    )

    from torchvision import datasets as tv_datasets
    pv_class_names = tv_datasets.ImageFolder(str(Path(args.data_dir) / "plantvillage" / "train")).classes

    model = build_dual_head_model(
        num_classes=len(pv_class_names),
        backbone=args.backbone,
        pretrained=True,
        separate_backbones=args.separate_backbones,
    ).to(device)

    if args.init_from is not None:
        init_ckpt = torch.load(args.init_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(init_ckpt["state_dict"], strict=False)
        if missing:
            print(f"Loaded with missing keys (expected for domain training): {missing}", flush=True)
        print(f"Warm-started from {args.init_from}", flush=True)

    model.freeze_all_except("domain")
    print("Frozen backbone + head_lab + head_field; training domain_classifier only", flush=True)

    model_kwargs = {"num_classes": len(pv_class_names), "backbone": args.backbone, "pretrained": True,
                     "freeze": True, "separate_backbones": args.separate_backbones}

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable} (domain classifier only)", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.domain_classifier.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    n_lab = len(train_loader.dataset.lab_dataset)
    n_field = len(train_loader.dataset.field_dataset)
    print(f"Training domain classifier on {n_lab} lab + {n_field} field images, "
          f"{args.epochs} epochs, lr={args.lr}", flush=True)

    start_epoch, best_val_acc = 1, 0.0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, domain_labels in tqdm(train_loader, desc="train-domain", leave=False):
            images, domain_labels = images.to(device), domain_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _cuda_autocast():
                features = model.extract_features(images)
                domain_logits = model.domain_classifier(features)
                loss = criterion(domain_logits, domain_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            correct += (domain_logits.argmax(dim=1) == domain_labels).sum().item()
            total += images.size(0)
        train_loss, train_acc = running_loss / total, correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, domain_labels in val_loader:
                images, domain_labels = images.to(device), domain_labels.to(device)
                with _cuda_autocast():
                    features = model.extract_features(images)
                    domain_logits = model.domain_classifier(features)
                    loss = criterion(domain_logits, domain_labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (domain_logits.argmax(dim=1) == domain_labels).sum().item()
                val_total += images.size(0)
        val_loss, val_acc = val_loss / val_total, val_correct / val_total

        print(f"epoch {epoch:02d}/{args.epochs}: train loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f}", flush=True)

        save_checkpoint(
            args.checkpoint_dir / f"domain_{args.tag}_epoch{epoch:02d}.pt",
            model, optimizer, scaler, epoch, val_acc, pv_class_names, model_kwargs,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                args.checkpoint_dir / f"best_domain_{args.tag}.pt",
                model, optimizer, scaler, epoch, val_acc, pv_class_names, model_kwargs,
            )

    print(f"Done. Best domain val acc {best_val_acc:.4f}", flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.dual_head and args.train_head == "domain":
        train_domain_classifier(args)
        return
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
        mix_with=args.mix_with,
        plantdoc_repeat=args.plantdoc_repeat,
        sample_balanced=args.class_balanced,
    )
    model_kwargs = {"num_classes": len(class_names), "backbone": args.backbone, "pretrained": True,
                     "freeze": True}

    if args.dual_head:
        model = build_dual_head_model(
            num_classes=len(class_names),
            backbone=args.backbone,
            pretrained=True,
            separate_backbones=args.separate_backbones,
        ).to(device)
        if args.train_head is not None:
            model.freeze_all_except(args.train_head)
            print(f"DualHead: training head_{args.train_head}, other head frozen", flush=True)
            if args.backbone_lr > 0 and args.train_head != "domain":
                target = model.backbone_field if getattr(model, "separate_backbones", False) else model.backbone
                for p in target.parameters():
                    p.requires_grad = True
                print(f"DualHead: backbone UNFROZEN for domain adaptation (backbone_lr={args.backbone_lr})", flush=True)
        dual_head_mode = True
    else:
        model = build_model(**model_kwargs).to(device)
        dual_head_mode = False

    if args.init_from is not None:
        init_ckpt = torch.load(args.init_from, map_location="cpu")
        if init_ckpt["class_names"] != class_names:
            raise ValueError(
                f"init checkpoint classes ({len(init_ckpt['class_names'])}) differ from "
                f"current dataset classes ({len(class_names)})"
            )
        model.load_state_dict(init_ckpt["state_dict"], strict=False)
        print(f"Warm-started weights from {args.init_from}", flush=True)
        if dual_head_mode and args.train_head is not None:
            if args.train_head == "field":
                model.head_field.load_state_dict(model.head_lab.state_dict())
                if getattr(model, "separate_backbones", False):
                    model.backbone_field.load_state_dict(model.backbone_lab.state_dict())
                    print("Copied backbone_lab+head_lab -> backbone_field+head_field (warm start)", flush=True)
                else:
                    print("Copied head_lab -> head_field (warm-start field head from lab head)", flush=True)
            model.freeze_all_except(args.train_head)
            print(f"DualHead: re-froze after init, training head_{args.train_head}", flush=True)
            if args.backbone_lr > 0 and args.train_head != "domain":
                target = model.backbone_field if getattr(model, "separate_backbones", False) else model.backbone
                for p in target.parameters():
                    p.requires_grad = True
                print(f"DualHead: backbone UNFROZEN for domain adaptation (backbone_lr={args.backbone_lr})", flush=True)

    if not dual_head_mode and args.unfreeze_blocks > 0:
        unfreeze_last_blocks(model, args.unfreeze_blocks)
        print(f"Unfroze last {args.unfreeze_blocks} parameter-bearing backbone modules", flush=True)

    criterion = nn.CrossEntropyLoss()
    if dual_head_mode:
        active_head = model.get_head(args.train_head or "lab")
        head_params = list(active_head.parameters())
        backbone_params = [
            p for p in model.parameters()
            if p.requires_grad and all(id(p) != id(h) for h in head_params)
        ]
    else:
        head_params = list(head_module(model).parameters())
        backbone_params = [
            p for p in model.parameters()
            if p.requires_grad and all(id(p) != id(h) for h in head_params)
        ]
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": args.backbone_lr if args.backbone_lr > 0 else args.lr},
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

    mode = "dual-head" if dual_head_mode else ("fine-tuning" if args.unfreeze_blocks > 0 else "head-only")
    if dual_head_mode and args.train_head:
        mode = f"dual-head({args.train_head})"
    dataset_label = f"{args.dataset}+{args.mix_with}" if args.mix_with else args.dataset
    if args.mix_with and args.plantdoc_repeat > 1:
        dataset_label = f"{dataset_label} x{args.plantdoc_repeat}"
    print(f"Training {mode} on {dataset_label} ({len(train_loader.dataset)} train images, "
          f"{len(class_names)} classes), {args.epochs} epochs, "
          f"backbone={args.backbone}, lr={args.lr}, head_lr={args.head_lr or args.lr}, "
          f"backbone_lr={args.backbone_lr}, "
          f"augmentation={'ON' if args.augment else 'OFF'}, "
          f"class_balanced={'ON' if args.class_balanced else 'OFF'}, tag={args.tag}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, active_head=args.train_head)
        val_loss, val_acc = validate(model, val_loader, criterion, device, active_head=args.train_head)
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
