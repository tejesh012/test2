#!/bin/bash
#
# Convert TensorFlow SavedModel to TensorFlow.js format
#
# Usage:
#   ./convert_to_tfjs.sh [--quantize]
#
# Requirements:
#   pip install tensorflowjs
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models/landmark_model/saved_model"
OUTPUT_DIR="${SCRIPT_DIR}/frontend/public/models/tfjs_landmark_model"
QUANTIZE=${1:-""}

echo "=============================================="
echo "TensorFlow.js Model Converter"
echo "=============================================="
echo ""

# Check if saved model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: SavedModel not found at $MODEL_DIR"
    echo "Please run train_landmark_model.py first:"
    echo ""
    echo "  python train_landmark_model.py --synthetic"
    echo ""
    exit 1
fi

# Check tensorflowjs is installed
if ! python -c "import tensorflowjs" 2>/dev/null; then
    echo "Installing tensorflowjs..."
    pip install tensorflowjs
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Input:  $MODEL_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Build conversion command
CMD="tensorflowjs_converter"
CMD="$CMD --input_format=tf_saved_model"
CMD="$CMD --output_format=tfjs_graph_model"

if [ "$QUANTIZE" == "--quantize" ]; then
    echo "Quantization: ENABLED (uint8)"
    CMD="$CMD --quantize_uint8"
elif [ "$QUANTIZE" == "--quantize_float16" ]; then
    echo "Quantization: ENABLED (float16)"
    CMD="$CMD --quantize_float16"
else
    echo "Quantization: DISABLED"
fi

CMD="$CMD $MODEL_DIR $OUTPUT_DIR"

echo ""
echo "Running: $CMD"
echo ""

eval $CMD

# Check output size
echo ""
echo "Conversion complete!"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"
echo ""

# Calculate total size
TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
echo "Total model size: $TOTAL_SIZE"
echo ""

# Check if under target
SIZE_MB=$(du -sm "$OUTPUT_DIR" | cut -f1)
if [ "$SIZE_MB" -lt 6 ]; then
    echo "✓ Model is under 6MB target"
else
    echo "⚠ Model exceeds 6MB target. Consider using --quantize flag."
fi

echo ""
echo "Next steps:"
echo "1. Start frontend: cd frontend && npm run dev"
echo "2. Navigate to dashboard"
echo "3. Model will load from /models/tfjs_landmark_model/model.json"
echo ""
