#!/usr/bin/env bash
# Mirror Box — one-command environment setup
# Usage: bash setup.sh
#
# Creates venv, installs all packages with all extras, ready to run.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

echo "Setting up Mirror Box environment..."

# Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip --quiet

# Install both packages with all extras
echo "Installing aiware base library..."
pip install -e "$REPO_DIR/packages/aiware" --quiet

echo "Installing mirror-box with all extras (dashboard, cuda, dev)..."
pip install -e "$REPO_DIR/packages/mirror_box[dashboard,cuda,dev]" --quiet

# Install web chat dependencies
echo "Installing web UI dependencies (fastapi, uvicorn with websocket support)..."
pip install -e "$REPO_DIR/packages/mirror_box[web]" --quiet

# Install root dev dependencies
echo "Installing workspace dev tools..."
pip install -e "$REPO_DIR[dev]" --quiet

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Run web UI:  python -m mirror_box.scripts.web_chat --gpu"
echo "Run demo:    mirror-box-demo --gpu --dashboard"
