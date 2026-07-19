"""
serving/main.py

FastAPI inference endpoint for the final aggregated (or centralized)
global model checkpoint.

Endpoints:
  POST /predict     -> accepts a 28x28 grayscale image, returns prediction + confidence
  GET  /health       -> liveness check
  GET  /model-info    -> version / training-round / accuracy metadata

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000

The model path is configurable via the MODEL_PATH env var so a new
checkpoint can be swapped in (via a mounted volume) without rebuilding
the Docker image.
"""

import os
from typing import List

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "model_store/global_model.pt")
DEVICE = torch.device("cpu")  # inference-only container, keep it simple/portable


# --- Model definition (kept in sync with fl/model.py; duplicated here so the
#     serving image has zero dependency on the rest of the repo) -------------
class FLModel(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(16 * 7 * 7, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PredictRequest(BaseModel):
    # Flattened 28x28 grayscale pixel values, normalized 0-1
    pixels: List[float] = Field(..., min_length=784, max_length=784)


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    class_probabilities: List[float]


app = FastAPI(title="Federated Learning — Global Model Serving API")

_model = None
_checkpoint_meta = {}


@app.on_event("startup")
def load_model():
    global _model, _checkpoint_meta
    if not os.path.exists(MODEL_PATH):
        # Don't crash the container — /health and /model-info should still
        # report a clear "not loaded" status instead of a 500 loop.
        print(f"[startup] WARNING: no checkpoint found at {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = FLModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _model = model
    _checkpoint_meta = {
        "training_round": checkpoint.get("round"),
        "test_accuracy": checkpoint.get("test_accuracy"),
    }
    print(f"[startup] Loaded model from {MODEL_PATH}: {_checkpoint_meta}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model-info")
def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {
        "model_path": MODEL_PATH,
        "training_round": _checkpoint_meta.get("training_round"),
        "test_accuracy": _checkpoint_meta.get("test_accuracy"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check /health")

    x = torch.tensor(request.pixels, dtype=torch.float32).view(1, 1, 28, 28)
    with torch.no_grad():
        logits = _model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
        pred = int(torch.argmax(probs).item())
        confidence = float(probs[pred].item())

    return PredictResponse(
        prediction=pred,
        confidence=confidence,
        class_probabilities=[float(p) for p in probs],
    )
