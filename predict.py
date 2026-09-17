"""
Run inference on foreign images using the trained models/model.pt.

Examples:
    python predict.py
    python predict.py samples/cat.jpg samples/dog.jpg samples/dog2.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.pt"
SAMPLES_DIR = ROOT / "samples"

LABELS_HE = {"cat": "חתול", "dog": "כלב"}


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )
    return model


def load_checkpoint() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Run python train.py first."
        )
    return torch.load(MODEL_PATH, map_location="cpu", weights_only=False)


def predict_image(
    image_path: Path, checkpoint: dict, model: nn.Module
) -> tuple[str, float, dict[str, float]]:
    classes: list[str] = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 224)
    mean = checkpoint.get("mean", (0.485, 0.456, 0.406))
    std = checkpoint.get("std", (0.229, 0.224, 0.225))

    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = tf(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    best_idx = int(probs.argmax().item())
    label = classes[best_idx]
    confidence = float(probs[best_idx].item())
    all_probs = {cls: float(probs[i].item()) for i, cls in enumerate(classes)}
    return label, confidence, all_probs


def default_samples() -> list[Path]:
    files = sorted(
        p
        for p in SAMPLES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if len(files) < 3:
        raise FileNotFoundError(
            f"Need at least 3 foreign images in {SAMPLES_DIR}"
        )
    return files


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or default_samples()
    checkpoint = load_checkpoint()
    print(f"Loaded {MODEL_PATH}")
    print(f"Val accuracy when saved: {checkpoint.get('val_accuracy', 0):.2%}")
    print("-" * 56)

    classes: list[str] = checkpoint["classes"]
    model = build_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for path in paths:
        if not path.exists():
            print(f"{path}: FILE NOT FOUND")
            continue
        label, confidence, all_probs = predict_image(path, checkpoint, model)
        he = LABELS_HE.get(label, label)
        extras = "  ".join(f"{k}={v:.1%}" for k, v in all_probs.items())
        print(f"{path.name:20}  ->  {label} ({he})  {confidence:.1%}    [{extras}]")


if __name__ == "__main__":
    main()
