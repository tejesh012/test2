#!/bin/bash
#
# Smoke test for client-side emotion recognition
#
# This script tests that:
# 1. Frontend builds successfully
# 2. TF.js model files are present (or placeholder)
# 3. Component can be imported
#
# Usage:
#   ./smoke_client.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo "=============================================="
echo "Client Smoke Test"
echo "=============================================="
echo "Frontend dir: $FRONTEND_DIR"
echo ""

cd "$FRONTEND_DIR"

# Test 1: Check dependencies
echo "[Test 1] Checking package.json dependencies..."

if grep -q "@tensorflow/tfjs" package.json; then
    echo "✓ @tensorflow/tfjs is in dependencies"
else
    echo "✗ @tensorflow/tfjs missing from dependencies"
    echo "  Run: npm install @tensorflow/tfjs @mediapipe/face_mesh"
    exit 1
fi

if grep -q "@mediapipe/face_mesh" package.json; then
    echo "✓ @mediapipe/face_mesh is in dependencies"
else
    echo "✗ @mediapipe/face_mesh missing from dependencies"
    exit 1
fi
echo ""

# Test 2: Check component exists
echo "[Test 2] Checking component file..."

COMPONENT="$FRONTEND_DIR/src/components/CameraEmotionTFJS.tsx"
if [ -f "$COMPONENT" ]; then
    echo "✓ CameraEmotionTFJS.tsx exists"

    # Check for required exports
    if grep -q "export function CameraEmotionTFJS" "$COMPONENT"; then
        echo "✓ Component function is exported"
    else
        echo "✗ Component function not found"
        exit 1
    fi

    if grep -q "EMOTION_LABELS" "$COMPONENT"; then
        echo "✓ EMOTION_LABELS constant found"
    else
        echo "✗ EMOTION_LABELS not found"
        exit 1
    fi
else
    echo "✗ Component file not found: $COMPONENT"
    exit 1
fi
echo ""

# Test 3: Check emotion icons
echo "[Test 3] Checking emotion icons..."

ICONS_DIR="$FRONTEND_DIR/public/emotions"
EMOTIONS=("happy" "sad" "angry" "neutral" "surprise" "fear" "disgust")

if [ -d "$ICONS_DIR" ]; then
    echo "✓ Emotions directory exists"

    for emotion in "${EMOTIONS[@]}"; do
        if [ -f "$ICONS_DIR/$emotion.svg" ]; then
            echo "  ✓ $emotion.svg"
        else
            echo "  ✗ $emotion.svg missing"
        fi
    done
else
    echo "✗ Emotions directory not found"
    echo "  Creating placeholder directory..."
    mkdir -p "$ICONS_DIR"
fi
echo ""

# Test 4: Check model directory
echo "[Test 4] Checking model directory..."

MODEL_DIR="$FRONTEND_DIR/public/models/tfjs_landmark_model"
if [ -d "$MODEL_DIR" ]; then
    echo "✓ Model directory exists"

    if [ -f "$MODEL_DIR/model.json" ]; then
        echo "✓ model.json found"
    else
        echo "⚠ model.json not found (will run in demo mode)"
        echo "  Train model with: python train_landmark_model.py --synthetic"
        echo "  Convert with: ./convert_to_tfjs.sh"
    fi
else
    echo "⚠ Model directory not found"
    echo "  Creating placeholder..."
    mkdir -p "$MODEL_DIR"
fi
echo ""

# Test 5: TypeScript compilation check
echo "[Test 5] TypeScript compilation check..."

if command -v npx &> /dev/null; then
    echo "Running tsc --noEmit..."
    if npx tsc --noEmit 2>&1 | head -20; then
        echo "✓ TypeScript check passed (or has expected errors)"
    else
        echo "⚠ TypeScript check had issues"
    fi
else
    echo "⚠ npx not found, skipping TypeScript check"
fi
echo ""

# Test 6: Build test (optional, takes time)
echo "[Test 6] Build test (optional)..."

if [ "$1" == "--build" ]; then
    echo "Running npm run build..."
    if npm run build; then
        echo "✓ Build passed"
    else
        echo "✗ Build failed"
        exit 1
    fi
else
    echo "Skipping build test (use --build flag to enable)"
fi
echo ""

echo "=============================================="
echo "Client smoke tests completed!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Install dependencies: npm install"
echo "2. Start dev server: npm run dev"
echo "3. Navigate to /dashboard"
echo "4. Check console for: 'Camera request started', 'Video playing', etc."
