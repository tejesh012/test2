#!/bin/bash
#
# Export PyTorch model to ONNX format
#
# Usage:
#   ./export_to_onnx.sh [model_path] [output_path]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${1:-$SCRIPT_DIR/server_model/checkpoints/best_model.pth}"
OUTPUT_PATH="${2:-$SCRIPT_DIR/server_model/checkpoints/model.onnx}"

echo "=============================================="
echo "ONNX Model Export"
echo "=============================================="
echo ""
echo "Input:  $MODEL_PATH"
echo "Output: $OUTPUT_PATH"
echo ""

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    echo ""
    echo "Please train the model first:"
    echo "  cd server_model && python train.py --synthetic"
    exit 1
fi

# Run export script
python3 << 'PYTHON_SCRIPT'
import sys
import torch
sys.path.insert(0, 'server_model')

from train import EmotionRecognitionModel, NUM_CLASSES, SEQUENCE_LENGTH, IMAGE_SIZE

model_path = sys.argv[1] if len(sys.argv) > 1 else "server_model/checkpoints/best_model.pth"
output_path = sys.argv[2] if len(sys.argv) > 2 else "server_model/checkpoints/model.onnx"

print("Loading model...")
device = torch.device('cpu')
model = EmotionRecognitionModel(NUM_CLASSES, pretrained=False)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Creating dummy input...")
dummy_input = torch.randn(1, SEQUENCE_LENGTH, 3, IMAGE_SIZE, IMAGE_SIZE)

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=['video_frames'],
    output_names=['emotion_probs'],
    dynamic_axes={
        'video_frames': {0: 'batch_size'},
        'emotion_probs': {0: 'batch_size'}
    },
    opset_version=14,
    do_constant_folding=True
)

print(f"✓ Exported to {output_path}")

# Verify
import onnx
model_onnx = onnx.load(output_path)
onnx.checker.check_model(model_onnx)
print("✓ ONNX model verified")

# Show size
import os
size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"✓ Model size: {size_mb:.2f} MB")
PYTHON_SCRIPT

echo ""
echo "Export complete!"
echo ""
echo "To run inference with ONNX Runtime:"
echo "  import onnxruntime as ort"
echo "  session = ort.InferenceSession('$OUTPUT_PATH')"
echo "  outputs = session.run(None, {'video_frames': input_array})"
