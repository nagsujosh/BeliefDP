#!/usr/bin/env bash
# Setup script for the BeliefDP conda environment.
# Usage: bash setup_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="beliefdp"

echo "=== Creating conda environment '${ENV_NAME}' ==="
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
conda env create -f "${SCRIPT_DIR}/environment.yml"

echo "=== Activating environment ==="
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== Installing project packages ==="
pip install -e "${SCRIPT_DIR}/external/r3m"
pip install -e "${SCRIPT_DIR}"

echo "=== Installing runtime dependency ==="
pip install dm-tree

echo "=== Verifying installations ==="
python -c "import torch; print(f'PyTorch ....... {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import h5py; print('h5py .......... OK')"
python -c "import cv2; print('OpenCV ........ OK')"
python -c "import r3m; print('R3M ........... OK')"

echo ""
echo "=== Setup complete! ==="
echo "Activate with:  conda activate ${ENV_NAME}"
