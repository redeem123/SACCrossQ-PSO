#!/bin/bash
# run_docker.sh — Build & run the Olympe container on macOS
# Usage: bash run_docker.sh [command]
# Examples:
#   bash run_docker.sh python connect_test.py
#   bash run_docker.sh python stream_gps.py --duration 120
#   bash run_docker.sh bash    # interactive shell

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMAGE="anafi-olympe"

# Build if needed
if [[ "$(docker images -q $IMAGE 2>/dev/null)" == "" ]]; then
    echo "[INFO] Building Docker image (first run — takes ~3 min)..."
    docker build -t $IMAGE "$SCRIPT_DIR/docker/"
fi

# Run with host networking so it can reach 192.168.53.1 over USB-RNDIS
# Mount the entire validation/anafi folder as /workspace
docker run --rm \
    --network host \
    -v "$SCRIPT_DIR:/workspace" \
    -w /workspace \
    $IMAGE \
    "${@:-bash}"
