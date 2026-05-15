#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "  ============================================================"
echo "    Ceramic Defect Detector  |  Auto-installer"
echo "  ============================================================"
echo ""

# ── 1. Check Python ────────────────────────────────────────────
echo -e "  ${YELLOW}[1/4]${NC} Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo -e "  ${RED}[ERROR]${NC} python3 not found. Install Python 3.11+ from https://python.org"
    exit 1
fi
PYVER=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "  ${GREEN}[OK]${NC} Python $PYVER found."
echo ""

# ── 2. Create virtual environment ──────────────────────────────
echo -e "  ${YELLOW}[2/4]${NC} Creating virtual environment (.venv)..."
if [ -d ".venv" ]; then
    echo -e "  [SKIP] .venv already exists."
else
    python3 -m venv .venv
    echo -e "  ${GREEN}[OK]${NC} Virtual environment created."
fi
echo ""

# ── 3. Install dependencies ─────────────────────────────────────
echo -e "  ${YELLOW}[3/4]${NC} Installing dependencies (this may take a few minutes)..."
source .venv/bin/activate

pip install --upgrade pip --quiet

# Detect NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    echo -e "  ${GREEN}[GPU]${NC} NVIDIA GPU detected — installing PyTorch with CUDA 12.1..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
else
    echo -e "  ${YELLOW}[CPU]${NC} No NVIDIA GPU — installing CPU-only PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
fi

pip install -r requirements.txt --quiet
echo -e "  ${GREEN}[OK]${NC} All dependencies installed."
echo ""

# ── 4. Model weights ────────────────────────────────────────────
echo -e "  ${YELLOW}[4/4]${NC} Checking model weights..."
WEIGHTS_DIR="scratch_detector/outputs/runs/scratch_yolo/weights"
mkdir -p "$WEIGHTS_DIR"

if [ ! -f "$WEIGHTS_DIR/best.pt" ]; then
    echo -e "  ${YELLOW}[INFO]${NC} Model weights not found."
    echo "         Download best.pt from the GitHub Releases page and place it at:"
    echo "         $WEIGHTS_DIR/best.pt"
    echo ""
    echo "         Or train your own model:"
    echo "           cd scratch_detector && python src/train.py"
else
    echo -e "  ${GREEN}[OK]${NC} Model weights found."
fi

echo ""
echo "  ============================================================"
echo "    Installation complete!"
echo "  ============================================================"
echo ""
echo "  To start the web application:"
echo ""
echo "    1. Activate the environment:"
echo "         source .venv/bin/activate"
echo ""
echo "    2. Start the server:"
echo "         cd scratch_detector"
echo "         uvicorn web.app:app --reload --port 8000"
echo ""
echo "    3. Open your browser at:"
echo "         http://localhost:8000"
echo ""
