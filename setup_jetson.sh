#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "[1/5] Update apt index"
sudo apt-get update

echo "[2/5] Install base system deps"
sudo apt-get install -y python3-venv python3-pip git wget curl ca-certificates build-essential default-jre

echo "[3/5] Ensure cuSPARSELt is installed (for Jetson PyTorch wheel)"
if ! ldconfig -p | grep -qi cusparselt; then
  TMP_SCRIPT="/tmp/install_cusparselt.sh"
  wget -qO "$TMP_SCRIPT" https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cusparselt.sh
  chmod +x "$TMP_SCRIPT"
  export CUDA_VERSION=12.6
  sudo -E bash "$TMP_SCRIPT"
fi
sudo ldconfig

echo "[4/5] Create/activate venv"
if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "[5/5] Install Python dependencies"
pip install --no-cache-dir -r requirements.txt

echo "Done. Activate env with: source venv/bin/activate"
echo "Then run: python imgcap.py"
