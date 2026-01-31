#!/bin/bash
#
# Smoke test for server-side emotion recognition API
#
# Usage:
#   ./smoke_server.sh [server_url]
#

set -e

SERVER_URL="${1:-http://localhost:8000}"

echo "=============================================="
echo "Server Smoke Test"
echo "=============================================="
echo "Server URL: $SERVER_URL"
echo ""

# Test 1: Health check
echo "[Test 1] Health check..."
HEALTH=$(curl -s "$SERVER_URL/health")
echo "Response: $HEALTH"

if echo "$HEALTH" | grep -q '"status":"healthy"'; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed"
    exit 1
fi
echo ""

# Test 2: Root endpoint
echo "[Test 2] Root endpoint..."
ROOT=$(curl -s "$SERVER_URL/")
echo "Response: $ROOT"

if echo "$ROOT" | grep -q '"status":"running"'; then
    echo "✓ Root endpoint passed"
else
    echo "✗ Root endpoint failed"
    exit 1
fi
echo ""

# Test 3: Predict with dummy data
echo "[Test 3] Predict endpoint with sample data..."

# Create a simple base64 encoded 1x1 red pixel PNG
# This is a minimal valid image for testing
SAMPLE_IMAGE="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

# Create JSON payload with 5 frames (minimum for demo)
PAYLOAD=$(cat <<EOF
{
  "frames": [
    {"image": "data:image/png;base64,$SAMPLE_IMAGE", "timestamp": 0.0},
    {"image": "data:image/png;base64,$SAMPLE_IMAGE", "timestamp": 0.1},
    {"image": "data:image/png;base64,$SAMPLE_IMAGE", "timestamp": 0.2},
    {"image": "data:image/png;base64,$SAMPLE_IMAGE", "timestamp": 0.3},
    {"image": "data:image/png;base64,$SAMPLE_IMAGE", "timestamp": 0.4}
  ],
  "user_id": 1
}
EOF
)

# Send request
START_TIME=$(date +%s%N)
PREDICT=$(curl -s -X POST "$SERVER_URL/predict" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")
END_TIME=$(date +%s%N)

ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))

echo "Response: $PREDICT"
echo "Request time: ${ELAPSED_MS}ms"

# Check response has required fields
if echo "$PREDICT" | grep -q '"label"'; then
    echo "✓ Response has 'label' field"
else
    echo "✗ Response missing 'label' field"
    exit 1
fi

if echo "$PREDICT" | grep -q '"score"'; then
    echo "✓ Response has 'score' field"
else
    echo "✗ Response missing 'score' field"
    exit 1
fi

echo ""

# Test 4: Landmarks endpoint
echo "[Test 4] Landmarks predict endpoint..."

LANDMARKS_PAYLOAD=$(cat <<EOF
{
  "frames": [
    {"landmarks": [0.1, 0.2, 0.3, 0.4, 0.5], "timestamp": 0.0},
    {"landmarks": [0.1, 0.2, 0.3, 0.4, 0.5], "timestamp": 0.1}
  ]
}
EOF
)

LANDMARKS_RESULT=$(curl -s -X POST "$SERVER_URL/predict/landmarks" \
    -H "Content-Type: application/json" \
    -d "$LANDMARKS_PAYLOAD")

echo "Response: $LANDMARKS_RESULT"

if echo "$LANDMARKS_RESULT" | grep -q '"label"'; then
    echo "✓ Landmarks endpoint passed"
else
    echo "✗ Landmarks endpoint failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "All smoke tests passed!"
echo "=============================================="
