"""
Train a cat-vs-dog classifier and save models/model.pt.

Dataset: Kaggle Dogs vs Cats (filtered educational subset)
https://www.kaggle.com/c/dogs-vs-cats
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "cats_and_dogs_filtered"
MODEL_PATH = ROOT / "models" / "model.pt"

# Hyperparameters — change these when calibrating on a branch
EPOCHS = 4
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
IMAGE_SIZE = 224
UNFREEZE_LAST_BLOCKS = 1

CLASSES = ["cat", "dog"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "validation"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found under {DATA_DIR}. "
            "Expected train/ and validation/ folders with cats/ and dogs/."
        )

    train_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    if train_ds.classes != CLASSES:
        print(f"Warning: folder classes {train_ds.classes} != {CLASSES}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


def build_model(num_classes: int = 2) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False

    if UNFREEZE_LAST_BLOCKS > 0:
        for block in model.features[-UNFREEZE_LAST_BLOCKS:]:
            for param in block.parameters():
                param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def train() -> None:
    device = get_device()
    print(f"Device: {device}", flush=True)
    print(f"Epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LEARNING_RATE}", flush=True)

    train_loader, val_loader = build_dataloaders()
    print(f"Train images: {len(train_loader.dataset)}", flush=True)
    print(f"Val images:   {len(val_loader.dataset)}", flush=True)

    model = build_model(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    best_acc = 0.0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"Starting epoch {epoch}/{EPOCHS}...", flush=True)
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        train_loss = running_loss / max(seen, 1)
        val_acc = evaluate(model, val_loader, device)
        marker = ""
        if val_acc >= best_acc:
            best_acc = val_acc
            marker = "  <- saved"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "image_size": IMAGE_SIZE,
                    "mean": IMAGENET_MEAN,
                    "std": IMAGENET_STD,
                    "val_accuracy": best_acc,
                    "epochs_trained": epoch,
                },
                MODEL_PATH,
            )
        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"loss={train_loss:.4f}  val_acc={val_acc:.2%}{marker}",
            flush=True,
        )

    print("-" * 48, flush=True)
    print(f"Best validation accuracy: {best_acc:.2%}", flush=True)
    print(f"Saved: {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    train()
