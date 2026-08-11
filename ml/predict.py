"""Single-image prediction (Sprint 1 'done when' check).

Loads a checkpoint, classifies one image, prints top-k classes with
confidence. Run from the repo root:

    python -m ml.predict --checkpoint checkpoints/best_plantvillage_stage1.pt \\
        --image data/plantvillage/test/Tomato___healthy/xxx.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .data_loading import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from .model import build_model

DEFAULT_TOP_K = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify a single image with a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="saved model checkpoint (.pt)")
    parser.add_argument("--image", type=Path, required=True, help="path to an image file")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K, help="print the top-k classes")
    parser.add_argument("--device", default="auto", help="'auto' | 'cuda' | 'cpu'")
    return parser.parse_args()


def preprocess(image_path: Path) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    image = preprocess(args.image).to(device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(image)
    probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idx = probs.topk(min(args.topk, len(ckpt["class_names"])))
    class_names = ckpt["class_names"]
    print(f"image: {args.image}")
    for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
        print(f"  {class_names[idx]:<50s} {prob * 100:5.1f}%")


if __name__ == "__main__":
    main()
