#!/usr/bin/env bash
# Auto-detect CUDA and launch the appropriate Docker profile
#
# Usage:
#   ./run.sh          # auto-detect
#   ./run.sh cpu      # force CPU
#   ./run.sh gpu      # force GPU
#   ./run.sh build    # auto-detect + force rebuild

set -e

FORCE=${1:-""}
REBUILD=""
if [ "$FORCE" = "build" ]; then
    REBUILD="--build"
    FORCE=""
fi

# Detect CUDA
if [ -z "$FORCE" ]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        echo "NVIDIA GPU detected: ${GPU_NAME:-unknown}"
        PROFILE="gpu"
    else
        echo "No NVIDIA GPU detected — using CPU build."
        PROFILE="cpu"
    fi
else
    PROFILE="$FORCE"
    echo "Profile forced: $PROFILE"
fi

echo "Starting wood-defect:${PROFILE} on http://localhost:8000"
echo ""

docker compose --profile "$PROFILE" up $REBUILD
