#!/bin/bash
set -e

echo "=== Deepfake Detector Deployment ==="

# Install ffmpeg for audio format support
echo "Installing ffmpeg..."
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
echo "ffmpeg installed."

# Install Python dependencies into existing venv
echo "Installing Python dependencies..."
pip install -q fastapi uvicorn[standard] python-multipart opencv-python-headless
echo "Dependencies installed."

# Sync repo
REPO_DIR="/root/deepfake-detector"
if [ -d "$REPO_DIR/.git" ]; then
    echo "Updating existing repo..."
    cd "$REPO_DIR"
    git pull
else
    echo "Cloning repo..."
    cd /root
    git clone https://github.com/hochul325/deepfake-detector.git
    cd "$REPO_DIR"
fi

# Pull LFS files if needed
if command -v git-lfs &> /dev/null; then
    git lfs pull
fi

# Kill any existing uvicorn process on port 8000
echo "Checking for existing server..."
pkill -f "uvicorn app:app" 2>/dev/null || true
sleep 1

# Start server
echo "Starting server on port 8000..."
cd "$REPO_DIR"
nohup uvicorn app:app --host 0.0.0.0 --port 8000 > /tmp/deepfake-detector.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo ""
        echo "=== Server is running! ==="
        echo "URL: http://$(hostname -I | awk '{print $1}'):8000"
        curl -s http://localhost:8000/api/health | python3 -m json.tool
        exit 0
    fi
    printf "."
    sleep 2
done

echo ""
echo "ERROR: Server did not start within 120 seconds."
echo "Check logs: tail -f /tmp/deepfake-detector.log"
exit 1
