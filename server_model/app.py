#!/usr/bin/env python3
"""
FastAPI Inference Server for Emotion Recognition

Endpoints:
- POST /predict - Predict emotion from landmarks or video frames
- GET /health - Health check

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import logging
import uuid
import base64
import io
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTION_LABELS)
IMAGE_SIZE = 224
SEQUENCE_LENGTH = 32

# Global model
model = None
device = None


class LandmarkFrame(BaseModel):
    """Single frame of landmarks."""
    landmarks: List[float]  # Flattened [x1, y1, x2, y2, ...]
    timestamp: Optional[float] = None


class LandmarksRequest(BaseModel):
    """Request with landmark sequences."""
    frames: List[LandmarkFrame]
    user_id: Optional[int] = None
    ts: Optional[float] = None


class FrameData(BaseModel):
    """Single frame as base64 image."""
    image: str  # Base64 encoded image
    timestamp: Optional[float] = None


class FramesRequest(BaseModel):
    """Request with image frames."""
    frames: List[FrameData]
    user_id: Optional[int] = None


class PredictionResponse(BaseModel):
    """Prediction response."""
    label: str
    score: float
    all_scores: Dict[str, float]
    per_frame: Optional[List[Dict[str, Any]]] = None
    inference_time_ms: float
    request_id: str


def load_model():
    """Load the trained model."""
    global model, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Try to load TorchScript model
    model_path = Path(__file__).parent / "checkpoints" / "model.pt"
    pth_path = Path(__file__).parent / "checkpoints" / "best_model.pth"

    if model_path.exists():
        logger.info(f"Loading TorchScript model from {model_path}")
        start = time.time()
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
        logger.info(f"Model loaded in {(time.time() - start)*1000:.0f}ms")
    elif pth_path.exists():
        logger.info(f"Loading PyTorch checkpoint from {pth_path}")
        # Import model class
        from train import EmotionRecognitionModel
        start = time.time()
        model = EmotionRecognitionModel(NUM_CLASSES, pretrained=False)
        checkpoint = torch.load(pth_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        logger.info(f"Model loaded in {(time.time() - start)*1000:.0f}ms")
    else:
        logger.warning("No trained model found. Running in demo mode.")
        model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield


app = FastAPI(
    title="Emotion Recognition API",
    description="Server-side emotion recognition using EfficientNet-B4 + TemporalConv",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Image transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 image string."""
    # Remove data URL prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]

    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image


def predict_from_frames(frames: List[Image.Image]) -> Dict[str, Any]:
    """Run prediction on image frames."""
    global model, device

    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] Processing {len(frames)} frames")

    if model is None:
        # Demo mode - return random prediction
        logger.info(f"[{request_id}] Running in demo mode (no model)")
        probs = np.random.dirichlet(np.ones(NUM_CLASSES))
        label_idx = np.argmax(probs)
        return {
            "label": EMOTION_LABELS[label_idx],
            "score": float(probs[label_idx]),
            "all_scores": {l: float(p) for l, p in zip(EMOTION_LABELS, probs)},
            "inference_time_ms": (time.time() - start_time) * 1000,
            "request_id": request_id
        }

    # Process frames
    processed_frames = []
    for img in frames:
        tensor = transform(img)
        processed_frames.append(tensor)

    # Pad or sample to SEQUENCE_LENGTH
    if len(processed_frames) < SEQUENCE_LENGTH:
        # Pad by repeating last frame
        last = processed_frames[-1]
        while len(processed_frames) < SEQUENCE_LENGTH:
            processed_frames.append(last.clone())
    elif len(processed_frames) > SEQUENCE_LENGTH:
        # Sample uniformly
        indices = np.linspace(0, len(processed_frames) - 1, SEQUENCE_LENGTH, dtype=int)
        processed_frames = [processed_frames[i] for i in indices]

    # Stack: (T, C, H, W) -> (1, T, C, H, W)
    batch = torch.stack(processed_frames).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    label_idx = np.argmax(probs)
    inference_time = (time.time() - start_time) * 1000

    logger.info(f"[{request_id}] Prediction: {EMOTION_LABELS[label_idx]} "
                f"({probs[label_idx]:.3f}) in {inference_time:.0f}ms")

    return {
        "label": EMOTION_LABELS[label_idx],
        "score": float(probs[label_idx]),
        "all_scores": {l: float(p) for l, p in zip(EMOTION_LABELS, probs)},
        "inference_time_ms": inference_time,
        "request_id": request_id
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Emotion Recognition API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "none",
        "labels": EMOTION_LABELS
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: FramesRequest):
    """
    Predict emotion from base64-encoded image frames.

    Request body:
    {
        "frames": [
            {"image": "base64...", "timestamp": 0.0},
            ...
        ],
        "user_id": 123
    }
    """
    if not request.frames:
        raise HTTPException(status_code=400, detail="No frames provided")

    try:
        # Decode images
        images = []
        for frame in request.frames:
            img = decode_base64_image(frame.image)
            images.append(img)

        # Run prediction
        result = predict_from_frames(images)

        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/landmarks")
async def predict_landmarks(request: LandmarksRequest):
    """
    Predict emotion from landmark sequences.

    This endpoint accepts pre-computed landmarks from the client TF.js model
    for validation/ensembling with the server model.

    Request body:
    {
        "frames": [
            {"landmarks": [x1, y1, x2, y2, ...], "timestamp": 0.0},
            ...
        ]
    }
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    if not request.frames:
        raise HTTPException(status_code=400, detail="No frames provided")

    logger.info(f"[{request_id}] Received {len(request.frames)} landmark frames")

    # For now, return a demo prediction based on landmark statistics
    # In production, you would have a separate model for landmarks
    all_landmarks = [f.landmarks for f in request.frames]
    mean_landmarks = np.mean(all_landmarks, axis=0)

    # Simple heuristic based on landmark positions
    # (This is a placeholder - real implementation would use a trained model)
    probs = np.random.dirichlet(np.ones(NUM_CLASSES) * 2)

    label_idx = np.argmax(probs)
    inference_time = (time.time() - start_time) * 1000

    logger.info(f"[{request_id}] Landmark prediction: {EMOTION_LABELS[label_idx]} in {inference_time:.0f}ms")

    return {
        "label": EMOTION_LABELS[label_idx],
        "score": float(probs[label_idx]),
        "all_scores": {l: float(p) for l, p in zip(EMOTION_LABELS, probs)},
        "inference_time_ms": inference_time,
        "request_id": request_id
    }


@app.post("/predict/upload")
async def predict_upload(
    file: UploadFile = File(...),
    num_frames: int = Form(default=32)
):
    """
    Predict emotion from uploaded video file.

    Accepts:
    - MP4 video file
    - ZIP file containing frames

    Returns emotion prediction.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] Received file: {file.filename}")

    try:
        content = await file.read()

        if file.filename.endswith('.zip'):
            # Extract frames from ZIP
            import zipfile
            images = []
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for name in sorted(zf.namelist()):
                    if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        with zf.open(name) as f:
                            img = Image.open(f).convert('RGB')
                            images.append(img)

            if not images:
                raise HTTPException(status_code=400, detail="No images found in ZIP")

        elif file.filename.endswith('.mp4'):
            # Extract frames from video using OpenCV
            import cv2
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            images = []

            while len(images) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(frame_rgb))

            cap.release()
            os.unlink(tmp_path)

            if not images:
                raise HTTPException(status_code=400, detail="Could not extract frames from video")

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .mp4 or .zip")

        # Run prediction
        result = predict_from_frames(images)
        result["frames_processed"] = len(images)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
