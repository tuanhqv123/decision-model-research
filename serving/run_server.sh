#!/bin/bash
# Start the d1 API detached, bound to 127.0.0.1 only (reach it through an SSH tunnel). Args: [model_dir] [img_width] [mem_frac]
set -euo pipefail
cd ~/d1-api
MODEL=${1:-models/e2-final}; W=${2:-720}; FRAC=${3:-0.35}
[ -f .env ] && set -a && . ./.env && set +a
[ -f api.pid ] && kill -0 "$(cat api.pid)" 2>/dev/null && { echo "already running pid $(cat api.pid)"; exit 0; }
mkdir -p logs
D1_MODEL=$MODEL D1_IMG_W=$W D1_MEM_FRAC=$FRAC setsid nohup venv/bin/python -m uvicorn d1_api:app --host 127.0.0.1 --port 8700 --workers 1 \
  > logs/api.log 2>&1 < /dev/null &
echo $! > api.pid; echo "started pid $(cat api.pid) model=$MODEL img_w=$W mem_frac=$FRAC (log: ~/d1-api/logs/api.log)"
