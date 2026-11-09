#!/bin/bash
# Stop the d1 API and free its GPU memory. Add --purge to delete ~/d1-api completely (env, models, logs).
cd ~/d1-api 2>/dev/null || { echo "nothing to stop"; exit 0; }
[ -f api.pid ] && kill "$(cat api.pid)" 2>/dev/null && rm -f api.pid && echo "stopped"
if [ "${1:-}" = "--purge" ]; then cd ~ && rm -rf ~/d1-api && echo "purged ~/d1-api"; fi
