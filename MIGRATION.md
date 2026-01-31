# Two-Tier Emotion Detection System

## Overview

This implementation provides a sophisticated two-tier emotion detection system:

1. **Client-side (TensorFlow.js + MediaPipe)**: Fast, real-time detection at ~15 FPS
2. **Server-side (PyTorch EfficientNet-B4 + TemporalConv)**: High-accuracy detection for validation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                       │
├─────────────────────────────────────────────────────────────────┤
│  CameraEmotionTFJS.tsx                                          │
│  ├── MediaPipe FaceMesh (468 landmarks)                         │
│  ├── Feature extraction (1872 features/frame)                   │
│  ├── TensorFlow.js Conv1D model                                 │
│  └── Real-time overlay with emotion icons                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ (Optional validation)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SERVER (FastAPI + PyTorch)                    │
├─────────────────────────────────────────────────────────────────┤
│  app.py                                                         │
│  ├── EfficientNet-B4 backbone                                   │
│  ├── TemporalConv1D head (3 layers)                             │
│  ├── Mixed precision inference                                  │
│  └── TorchScript/ONNX export                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

**Frontend:**
```bash
cd frontend
npm install @tensorflow/tfjs @mediapipe/face_mesh
npm install
```

**Server (in a virtual environment):**
```bash
cd server_model
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Client Model Training (optional):**
```bash
pip install tensorflow tensorflowjs scikit-learn
```

### 2. Train Models (or use demo mode)

**Train client landmark model:**
```bash
# Generate synthetic data and train
python train_landmark_model.py --synthetic --epochs 50

# Convert to TensorFlow.js
./convert_to_tfjs.sh
```

**Train server PyTorch model:**
```bash
cd server_model
python train.py --synthetic --epochs 60
```

### 3. Start Services

**Frontend:**
```bash
cd frontend
npm run dev
# Opens at http://localhost:3000
```

**Server (in separate terminal):**
```bash
cd server_model
uvicorn app:app --host 0.0.0.0 --port 8000
# API at http://localhost:8000
```

### 4. Run Smoke Tests

```bash
# Client test
./tests/smoke_client.sh

# Server test (with server running)
./tests/smoke_server.sh
```

## Expected Console Logs

When running the client correctly, you should see:

```
[Camera] Request started
[Camera] Requesting camera permission...
[Camera] Permission granted
[Camera] Video playing
[MediaPipe] Ready
[Window inference] label=happy, conf=0.847
[Window inference] label=happy, conf=0.892
...
```

## File Structure

```
majorproj/
├── frontend/
│   ├── src/
│   │   └── components/
│   │       └── CameraEmotionTFJS.tsx    # Main TF.js component
│   ├── public/
│   │   ├── emotions/                     # Emotion SVG icons
│   │   │   ├── happy.svg
│   │   │   ├── sad.svg
│   │   │   ├── angry.svg
│   │   │   ├── neutral.svg
│   │   │   ├── surprise.svg
│   │   │   ├── fear.svg
│   │   │   └── disgust.svg
│   │   └── models/
│   │       └── tfjs_landmark_model/      # TF.js model files
│   │           └── model.json
│   └── package.json                      # Updated with TF.js deps
│
├── server_model/
│   ├── train.py                          # PyTorch training pipeline
│   ├── app.py                            # FastAPI inference server
│   ├── eval.py                           # Evaluation script
│   ├── requirements.txt                  # Server dependencies
│   ├── Dockerfile                        # CPU Docker build
│   ├── Dockerfile.gpu                    # GPU Docker build
│   └── checkpoints/                      # Trained models
│       ├── best_model.pth
│       ├── model.pt                      # TorchScript
│       └── model.onnx                    # ONNX export
│
├── train_landmark_model.py               # TF.js model training
├── convert_to_tfjs.sh                    # TF.js conversion
├── export_to_onnx.sh                     # ONNX export
│
├── tests/
│   ├── smoke_client.sh                   # Client smoke test
│   └── smoke_server.sh                   # Server smoke test
│
└── MIGRATION.md                          # This file
```

## API Endpoints

### Server API (http://localhost:8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Predict from base64 frames |
| `/predict/landmarks` | POST | Predict from landmark data |
| `/predict/upload` | POST | Predict from video file |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"frames": [{"image": "data:image/png;base64,..."}]}'
```

**Response:**
```json
{
  "label": "happy",
  "score": 0.92,
  "all_scores": {"happy": 0.92, "neutral": 0.05, ...},
  "inference_time_ms": 145.2,
  "request_id": "abc123"
}
```

## Model Architecture

### Client Model (TF.js)
- **Input**: (T=20, F=1872) - 20 frames × 1872 features
- **Architecture**:
  - Conv1D(64, kernel=3, relu)
  - Conv1D(128, kernel=3, relu)
  - GlobalAveragePooling1D
  - Dense(128, relu) + Dropout(0.3)
  - Dense(7, softmax)
- **Size**: < 6 MB (quantized)

### Server Model (PyTorch)
- **Input**: (B, T=32, C=3, H=224, W=224) - Video frames
- **Architecture**:
  - EfficientNet-B4 backbone (pretrained)
  - Linear projection to 512
  - 3× TemporalConv1D(512, kernel=3)
  - Global pooling + Dense classifier
- **Size**: ~75 MB

## Expected Performance

| Model | Dataset | Accuracy | F1 (weighted) |
|-------|---------|----------|---------------|
| Client (landmarks) | FER2013 | 60-75% | 0.58-0.72 |
| Server (EfficientNet) | AffectNet | 75-90% | 0.73-0.88 |

## Recommended Datasets

1. **AffectNet**: ~400k images, 8 emotions
2. **RAF-DB**: ~30k images, 7 emotions
3. **FER2013**: ~35k images, 7 emotions

### Preprocessing

```bash
# Expected directory structure after preprocessing:
data/affectnet/
├── angry/
│   ├── video1/
│   │   ├── frame001.jpg
│   │   └── ...
│   └── ...
├── happy/
└── ...
```

## Docker Deployment

**CPU:**
```bash
cd server_model
docker build -t emotion-server .
docker run -p 8000:8000 emotion-server
```

**GPU:**
```bash
docker build -f Dockerfile.gpu -t emotion-server-gpu .
docker run --gpus all -p 8000:8000 emotion-server-gpu
```

## Troubleshooting

### Camera not working
1. Check HTTPS or localhost (required for getUserMedia)
2. Allow camera permission in browser
3. Check console for errors

### Model not loading
1. Ensure model files exist in `public/models/tfjs_landmark_model/`
2. Check browser console for network errors
3. Run in demo mode without model files

### Server connection issues
1. Verify server is running: `curl http://localhost:8000/health`
2. Check CORS settings
3. Verify firewall allows port 8000

## License

MIT License - See LICENSE file for details.
