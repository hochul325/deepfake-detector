#!/bin/bash
set -e

echo "=== Deepfake Detector Deployment ==="

VENV="/root/bitmind-subnet/.venv"
PYTHON="$VENV/bin/python"
APP_DIR="/root/deepfake-detector"
PORT=8080

# Install ffmpeg for audio format support
echo "Installing ffmpeg..."
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
echo "ffmpeg installed."

# Install Python dependencies into existing venv
echo "Installing Python dependencies..."
$PYTHON -m pip install -q fastapi uvicorn python-multipart opencv-python-headless
echo "Dependencies installed."

# Set up app directory with symlinks to model dirs
mkdir -p "$APP_DIR/templates"

# Symlink model dirs if not already present
[ -L "$APP_DIR/image_detector" ] || ln -sf /root/image_detector_gas "$APP_DIR/image_detector"
[ -L "$APP_DIR/audio_detector" ] || ln -sf /root/audio_detector_gas2 "$APP_DIR/audio_detector"
[ -L "$APP_DIR/video_detector" ] || ln -sf /root/video_detector_gas "$APP_DIR/video_detector"

# Kill any existing uvicorn process
echo "Checking for existing server..."
pkill -f "uvicorn app:app" 2>/dev/null || true
sleep 1

# Start server
echo "Starting server on port $PORT..."
cd "$APP_DIR"
nohup $PYTHON -m uvicorn app:app --host 0.0.0.0 --port $PORT > /tmp/deepfake-detector.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start (models loading)..."
for i in $(seq 1 60); do
    if curl -s http://localhost:$PORT/api/health > /dev/null 2>&1; then
        echo ""
        echo "=== Server is running! ==="
        echo "Internal: http://localhost:$PORT"
        curl -s http://localhost:$PORT/api/health | python3 -m json.tool
        exit 0
    fi
    printf "."
    sleep 2
done

echo ""
echo "ERROR: Server did not start within 120 seconds."
echo "Check logs: tail -f /tmp/deepfake-detector.log"
exit 1
