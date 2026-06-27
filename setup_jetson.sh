#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "[1/6] Update apt index"
sudo apt-get update

echo "[2/6] Install base system deps"
sudo apt-get install -y python3-venv python3-pip git wget curl ca-certificates build-essential default-jre

echo "[3/6] Ensure cuSPARSELt is installed and discoverable"
if ! ldconfig -p | grep -qi 'libcusparseLt\.so'; then
  TMP_SCRIPT="/tmp/install_cusparselt.sh"
  wget -qO "$TMP_SCRIPT" https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cusparselt.sh
  chmod +x "$TMP_SCRIPT"
  export CUDA_VERSION=12.6

  # Script requires a version arg. Try known versions until one succeeds.
  CANDIDATES=("0.8.1.1" "0.8.0.4" "0.7.1.0" "0.6.2.3")
  INSTALLED=0
  for ver in "${CANDIDATES[@]}"; do
    echo "Trying cuSPARSELt version: $ver"
    rm -rf tmp_cusparselt /tmp/tmp_cusparselt || true
    if sudo -E bash "$TMP_SCRIPT" "$ver"; then
      INSTALLED=1
      echo "Installed cuSPARSELt version: $ver"
      break
    fi
  done

  if [ "$INSTALLED" -ne 1 ]; then
    echo "ERROR: Could not install cuSPARSELt from candidate versions."
    exit 1
  fi
fi

LIB_PATHS=$(sudo find /usr /usr/local -name 'libcusparseLt.so*' 2>/dev/null || true)
if [ -z "$LIB_PATHS" ]; then
  echo "ERROR: libcusparseLt not found on system after install attempt."
  echo "Jetson release:"
  cat /etc/nv_tegra_release || true
  exit 1
fi

LIB_FILE=$(echo "$LIB_PATHS" | head -n1)
LIB_DIR="$(dirname "$LIB_FILE")"
echo "Found cuSPARSELt in: $LIB_DIR"

echo "$LIB_DIR" | sudo tee /etc/ld.so.conf.d/cusparselt.conf >/dev/null

if [ -f "$LIB_DIR/libcusparseLt.so.0" ]; then
  echo "libcusparseLt.so.0 already present"
else
  ALT_SO=$(ls "$LIB_DIR"/libcusparseLt.so.* 2>/dev/null | sort -V | tail -n1 || true)
  if [ -n "${ALT_SO:-}" ]; then
    sudo ln -sf "$(basename "$ALT_SO")" "$LIB_DIR/libcusparseLt.so.0"
    echo "Created symlink: $LIB_DIR/libcusparseLt.so.0 -> $(basename "$ALT_SO")"
  fi
fi

sudo ldconfig
ldconfig -p | grep -i cusparselt || { echo "ERROR: ldconfig cannot find cuSPARSELt"; exit 1; }

echo "[4/6] Create/activate venv"
if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "[5/6] Install Python dependencies"
pip install --no-cache-dir -r requirements.txt

echo "[6/6] Verify torch import"
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY

echo "Done. Activate env with: source venv/bin/activate"
echo "Then run: python imgcap.py"
