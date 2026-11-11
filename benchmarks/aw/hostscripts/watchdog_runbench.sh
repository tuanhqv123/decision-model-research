#!/usr/bin/env bash
# Kill a run_bench chunk that has written no episode file for STALE_MIN minutes (2026-11-11: with the emulator dead,
# a chunk wrote its last episode and then never exited; the chain waited 60+ min). Killing the container-side
# run_bench lets `docker exec` return, run_arm finish its round, and the chain's sweep re-queue the victims.
# Usage: watchdog_runbench.sh <run_dir> [STALE_MIN=25]
set -u
RUN=${1:?run dir}; STALE=${2:-25}
LOG=$HOME/aw/stage1_chain.log
while true; do
  sleep 120
  if sudo -n docker exec aw-emu pgrep -f "python3 run_bench.py" >/dev/null 2>&1; then
    newest=$(ls -t "$RUN"/episodes/*.jsonl 2>/dev/null | head -1)
    age=999
    [ -n "$newest" ] && age=$(( ( $(date +%s) - $(stat -c %Y "$newest") ) / 60 ))
    if [ "$age" -ge "$STALE" ]; then
      echo "$(date -u +%FT%TZ) WATCHDOG: no new episode for ${age} min -> stopping the stuck run_bench" >> "$LOG"
      sudo -n docker exec aw-emu pkill -f "python3 run_bench.py"
      sleep 60
    fi
  fi
done
