from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.pt"
LABELS_HE = {"cat": "חתול", "dog": "כלב"}

@asynccontextmanager
async def lifespan(_: FastAPI):
    load_bundle()
    yield


app = FastAPI(title="pawID", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained weights: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = build_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
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
    return {
        "model": model,
        "classes": classes,
        "transform": tf,
        "val_accuracy": checkpoint.get("val_accuracy"),
    }


@app.get("/health")
def health() -> dict:
    bundle = load_bundle()
    return {
        "status": "ok",
        "classes": bundle["classes"],
        "val_accuracy": bundle["val_accuracy"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    content_type = (file.content_type or "").lower()
    if content_type and not (
        content_type.startswith("image/") or content_type == "application/octet-stream"
    ):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    raw = await file.read()
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read image.") from exc

    bundle = load_bundle()
    tensor = bundle["transform"](image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(bundle["model"](tensor), dim=1)[0]

    classes = bundle["classes"]
    best_idx = int(probs.argmax().item())
    label = classes[best_idx]
    return {
        "label": label,
        "label_he": LABELS_HE.get(label, label),
        "confidence": round(float(probs[best_idx].item()), 4),
        "probs": {
            cls: round(float(probs[i].item()), 4) for i, cls in enumerate(classes)
        },
    }
