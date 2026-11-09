#!/usr/bin/env bash
# Backup then teardown for the 2026-11-06 AndroidWorld session.
#
# Order is not negotiable: copy and verify first, delete second. A teardown that runs before a
# verified copy destroys the only artifact of the day.
#
# Hosts are passed as environment variables so no alias or address lives in this file:
#   GPU_HOST_A   the KVM host (emulator + image + build tree)
#   GPU_HOST_B   the second 4090 (model server + checkpoint + eval set)
#
#   BACKUP=1 ./backup-and-cleanup-2026-11-06.sh     # copy + verify only
#   CLEAN=1  ./backup-and-cleanup-2026-11-06.sh     # copy + verify + delete
set -uo pipefail

: "${GPU_HOST_A:?set GPU_HOST_A}"
: "${GPU_HOST_B:?set GPU_HOST_B}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/backup-2026-11-06"
SSH="ssh -o BatchMode=yes -o ServerAliveInterval=30"

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "FAILED: $*"; exit 1; }

mkdir -p "$DEST"/{gate_runs,reports,logs,configs,offline}

# ---------------------------------------------------------------- 1. gather
log "collecting runs from the emulator container"
$SSH "$GPU_HOST_A" 'rm -rf /tmp/aw_runs_20261106 /tmp/aw_runs_20261106.tgz /tmp/aw_reports.tgz;
  docker cp aw-emu:/opt/awh/aw/runs /tmp/aw_runs_20261106 2>/dev/null;
  tar czf /tmp/aw_runs_20261106.tgz -C /tmp aw_runs_20261106 2>/dev/null;
  docker exec aw-emu tar czf /tmp/aw_reports.tgz -C /opt/awh --exclude=aw . 2>/dev/null;
  docker cp aw-emu:/tmp/aw_reports.tgz /tmp/aw_reports.tgz 2>/dev/null;
  ls /tmp/aw_runs_20261106.tgz /tmp/aw_reports.tgz' \
  || die "could not collect from host A"

log "pulling to $DEST"
scp -q "$GPU_HOST_A:/tmp/aw_runs_20261106.tgz" "$DEST/gate_runs/" || die "runs copy failed"
scp -q "$GPU_HOST_A:/tmp/aw_reports.tgz" "$DEST/reports/" || log "WARN: reports archive missing (non-fatal)"
rsync -az "$GPU_HOST_A:~/aw-serve/logs/" "$DEST/logs/hostA/" 2>/dev/null
rsync -az "$GPU_HOST_B:~/aw-work/logs/" "$DEST/logs/hostB/" 2>/dev/null
rsync -az "$GPU_HOST_B:~/aw-work/results/" "$DEST/offline/" 2>/dev/null
cp -R "$REPO/benchmarks/aw/configs" "$DEST/configs/" 2>/dev/null

# ---------------------------------------------------------------- 2. verify
log "verifying the copy"
( cd "$DEST/gate_runs" && tar tzf aw_runs_20261106.tgz | grep -c "episodes/.*\.jsonl" ) \
  || die "no episode files in the archive"
tar xzf "$DEST/gate_runs/aw_runs_20261106.tgz" -C "$DEST/gate_runs"
EPISODES=$(find "$DEST/gate_runs" -name '*.jsonl' | wc -l | tr -d ' ')
log "episode files on disk: $EPISODES"
[ "$EPISODES" -gt 0 ] || die "zero episodes backed up"
find "$DEST/gate_runs" -name 'index.db' -exec sqlite3 {} 'select count(*) from episodes' \; 2>/dev/null | tail -3
log "backup verified at $DEST"
[ "${CLEAN:-0}" = "1" ] || { log "BACKUP ONLY (set CLEAN=1 to remove)"; exit 0; }

# ---------------------------------------------------------------- 3. teardown
log "teardown: host A"
$SSH "$GPU_HOST_A" '
  pkill -f "[t]cp_proxy" 2>/dev/null
  docker rm -f aw-emu >/dev/null 2>&1
  docker rmi -f aw-emu-snap:20261106 >/dev/null 2>&1
  docker rmi -f android_world:latest >/dev/null 2>&1
  rm -rf ~/aw-build ~/aw-serve
  rm -rf ~/.cache/huggingface/modules/transformers_modules/ckpt_hyphen_E2_hyphen_final
  rm -rf /tmp/aw*.tgz /tmp/aw_runs_20261106* /tmp/gate_*.sh /tmp/ablate*.sh /tmp/aw_screen*.png
  echo "hostA: containers=$(docker ps -a --format "{{.Names}}" | grep -c "^aw-" || true) images=$(docker images --format "{{.Repository}}" | grep -cE "aw-emu|^android_world$" || true) dirs=$([ -e ~/aw-build ] && echo 1 || echo 0)$([ -e ~/aw-serve ] && echo 1 || echo 0)"
'

log "teardown: host B"
$SSH "$GPU_HOST_B" '
  pkill -f "[u]vicorn d1_api" 2>/dev/null
  pkill -f "[r]egen_pure" 2>/dev/null
  rm -rf ~/aw-work
  rm -rf ~/.cache/huggingface/modules/transformers_modules/ckpt_hyphen_E2_hyphen_final
  pkill -f "[u]vicorn d1_api" 2>/dev/null
  rm -f /tmp/serve_b.tgz /tmp/regen.tgz /tmp/get-pip.py /tmp/probe_api.py /tmp/probe_aw.py
  rm -f /tmp/aw_screen.png /tmp/e2-*.jsonl /tmp/probe.jsonl /tmp/regen*.log
  echo "hostB: aw-work=$([ -e ~/aw-work ] && echo 1 || echo 0)"
'

log "teardown: local tunnels"
for pid in $(pgrep -f "[s]sh -N.*18700"); do kill "$pid" 2>/dev/null; done
sleep 1
[ "$(pgrep -f "[s]sh -N.*18700" | wc -l | tr -d ' ')" = "0" ] || log "WARN: a tunnel survived the kill"
rm -f /tmp/aw*.tgz /tmp/gate_*.sh /tmp/ablate*.sh /tmp/probe_api.py /tmp/reasons.py /tmp/inspect_run.py

log "definition of done: check each line above shows 0 / empty"
