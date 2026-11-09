#!/bin/bash
# One-time env for the d1 API on a GPU host (runs ON the server). Everything lives under ~/d1-api; nothing outside it is touched.
# Versions pinned to what trained/evaluated the checkpoints (torch 2.11 cu128, transformers 5.21).
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd ~/d1-api
command -v gcc >/dev/null || { echo "gcc missing (Triton needs it)"; exit 1; }
[ -d venv ] || uv venv venv --python 3.11
uv pip install -q --python venv/bin/python "torch==2.11.0" torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python venv/bin/python "transformers==5.21.0" accelerate pillow numpy fastapi "uvicorn[standard]" pydantic safetensors huggingface_hub
venv/bin/python -c "import torch,transformers,fastapi;print('ENV OK',torch.__version__,transformers.__version__,torch.cuda.get_device_name(0))"
