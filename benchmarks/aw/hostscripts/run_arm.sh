#!/bin/bash
# Run the 116-task AndroidWorld suite as resumable chunks.
#
# Design, and why each piece exists — every item below is a failure that actually happened on
# 2026-11-10, not a hypothetical:
#
#  * The a11y forwarder stops delivering protos after tens of minutes of emulator uptime.
#    `dumpsys accessibility` still says enabled+bound, /health still returns success, and
#    nothing in the HTTP surface reflects it; the only symptom is `Could not get a11y tree`
#    and episodes with n_steps=0. Recreating the container is the only reliable fix.
#  * `/tmp` inside the container is destroyed by that restart, so the harness, the config and
#    the per-task budget table are reinstalled every round, and RESULTS live on the host in a
#    bind-mounted directory.
#  * A freshly created container has no pyyaml. A chunk died on that AFTER paying the restart.
#  * A stale harness was silently used because tar's output went to /dev/null; the chunk then
#    ran the previous build while printing `harness_ok`.
#  * `--run-root` + `--run-id` nested the run directory two levels deep, so the recorder opened
#    a fresh index.db and re-ran completed episodes while the summary looked fine.
#
# Every one of those is now a cheap assertion in preflight.py. This script REFUSES to start a
# chunk if preflight fails, so a failure costs seconds instead of a restart.
set -u

CHUNK=${1:-12}
MAXROUNDS=${2:-12}
PORT=5051
HOSTHOME=$HOME
RUNID=${RUNID:?set RUNID}
RUNDIR=$HOSTHOME/awruns/$RUNID          # host path; mounted at /out
# The mount maps the HOST run directory straight onto /out, so /out IS the run root. Joining the
# run id again produced /out/aw116_episodic/ — a second, nested run. Episodes written there were
# invisible to resumption (`has_result` reads /out/index.db), so every chunk re-ran them or, once
# they existed in the nested index, reported `skipped: 5` while /out stayed empty. Half the
# results of 2026-11-10 were lost in that second directory.
CONTAINER_OUT=/out                       # /out IS the run root, not /out/<id>
HARNESS_TGZ=${HARNESS_TGZ:?set HARNESS_TGZ}
# Scored episodes that make the run directory complete: 116 for the suite, 20 for the Q10 subset.
# A 20-task run used to loop through empty rounds until MAXROUNDS because the target was fixed at 116.
TARGET=${TARGET:-116}

log() { echo "[$(date +%H:%M:%S)] $*"; }

install_harness() {
  sudo -n docker exec aw-emu bash -c 'rm -rf /tmp/awh; mkdir -p /tmp/awh'
  sudo -n docker exec -i aw-emu bash -c 'cat > /tmp/awh.tgz' < "$HARNESS_TGZ" || return 1
  sudo -n docker exec aw-emu bash -c 'tar xzf /tmp/awh.tgz -C /tmp/awh' 2>&1 \
    | grep -v "LIBARCHIVE\|Ignoring unknown" || true
  sudo -n docker exec -i aw-emu bash -c 'cat > /tmp/awh/aw/configs/aw116.yaml' < "${CFG:?set CFG}"
  sudo -n docker exec -i aw-emu bash -c 'cat > /tmp/probe.py' < "$HOSTHOME/aw/probe.py"
  sudo -n docker exec -i aw-emu bash -c 'cat > /tmp/preflight.py' < "$HOSTHOME/aw/preflight.py"

  # Per-task budget, the way upstream allocates it, regenerated because it lives in /tmp.
  sudo -n docker exec aw-emu python3 /tmp/awh/aw/dump_step_budgets.py /tmp/budgets.json 2>&1 \
    | head -1

  # A fresh container has no pyyaml and load_config needs it.
  sudo -n docker exec aw-emu bash -c \
    'python3 -c "import yaml" 2>/dev/null || uv pip install --system pyyaml 2>&1 | tail -1'
}

restart_emulator() {
  log "restart emulator (with /out mounted)"
  sudo -n docker rm -f aw-emu >/dev/null 2>&1 || true
  sudo -n docker run -d --name aw-emu \
    --device /dev/kvm --security-opt seccomp=unconfined \
    --cpus=8 --memory=24g --memory-swap=24g \
    --ulimit nofile=1048576:1048576 \
    -v "$RUNDIR":/out \
    -p 127.0.0.1:${PORT}:5000 \
    android_world:fixed >/dev/null 2>&1
  sleep 5
  sudo -n docker ps --format '{{.Names}}' | grep -q aw-emu || { log "START FAILED"; return 1; }
  for i in $(seq 1 60); do
    sleep 15
    if curl -s -m 8 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q success; then
      log "health ok after $((i*15))s"; return 0
    fi
    sudo -n docker ps --format '{{.Names}}' | grep -q aw-emu || { log "CONTAINER DIED"; return 1; }
  done
  log "HEALTH TIMEOUT"; return 1
}

ensure_container() {
  local need_restart=0
  sudo -n docker ps --format '{{.Names}}' | grep -q aw-emu || need_restart=1
  if [ "$need_restart" = 0 ]; then
    curl -s -m 8 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q success \
      || need_restart=1
  fi
  if [ "$need_restart" = 0 ]; then
    # The mount cannot be added to a running container, and without it results die on restart.
    sudo -n docker inspect aw-emu --format '{{range .Mounts}}{{.Source}}:{{.Destination}} {{end}}' \
      2>/dev/null | grep -qF "$RUNDIR:/out" || need_restart=1
  fi
  [ "$need_restart" = 0 ] && return 0
  restart_emulator || return 1
}

mkdir -p "$RUNDIR/episodes" "$RUNDIR/screens"

for round in $(seq 1 "$MAXROUNDS"); do
  log "################ ROUND $round ################"
  ensure_container || exit 1
  install_harness || { log "harness install failed"; exit 1; }

  # THE GATE. If any invariant is broken, stop here instead of burning a restart to find out.
  # The exit status must be honoured: an earlier version piped preflight through `tail`, so the
  # pipeline's status was tail's (always 0) and a FAILED gate still let the chunk run. It found a
  # real defect (index 18 vs 19 files) and the run proceeded anyway.
  if ! sudo -n docker exec aw-emu python3 /tmp/preflight.py \
        --run-root "$RUNDIR" --run-id "$RUNID" --expect-contract ${CONTRACT:?set CONTRACT} > /tmp/preflight.out 2>&1; then
    tail -25 /tmp/preflight.out
    log "PREFLIGHT FAILED — not starting a chunk"
    if grep -q "a11y chain alive.*FAIL" /tmp/preflight.out; then
      log "a11y is the failure -> one restart, then re-check"
      restart_emulator || exit 1
      install_harness || exit 1
      sudo -n docker exec aw-emu python3 /tmp/preflight.py \
        --run-root "$RUNDIR" --run-id "$RUNID" --expect-contract ${CONTRACT:?set CONTRACT} > /tmp/preflight2.out 2>&1 \
        || { tail -20 /tmp/preflight2.out; log "still failing after restart"; exit 2; }
    elif grep -q "resume index consistent.*FAIL" /tmp/preflight.out; then
      log "index disagrees with files on disk -> rebuilding from the artifacts"
      sudo -n docker exec aw-emu python3 /tmp/awh/aw/rebuild_index.py /out 2>&1 | tail -2
      sudo -n docker exec aw-emu python3 /tmp/preflight.py \
        --run-root "$RUNDIR" --run-id "$RUNID" --expect-contract ${CONTRACT:?set CONTRACT} > /tmp/preflight2.out 2>&1 \
        || { tail -20 /tmp/preflight2.out; log "still failing after index rebuild"; exit 2; }
    else
      exit 2
    fi
  else
    tail -18 /tmp/preflight.out
  fi

  log "chunk: up to $CHUNK new episodes"
  # NO `--run-id`: it exists only to make a subdirectory under `--run-root`, and `/out` IS the
  # run directory (the bind mount maps the host run dir onto it). Passing it joined the id a
  # second time and wrote into `/out/aw116_episodic/`, a directory resumption never reads. The
  # recorder's own run id comes from `new_run_id(plan.agent)` and is unaffected.
  sudo -n docker exec -e PYTHONPATH=/tmp/awh/aw -w /tmp/awh/aw aw-emu \
    python3 run_bench.py --config configs/aw116.yaml \
    --run-root "$CONTAINER_OUT" --max-new "$CHUNK" 2>&1 | tail -14

  # Count SCORED episodes only, from the run's index (status 'ok', the rule resumption uses). Counting
  # files was wrong twice: `*.jsonl` matched `<id>.abandoned.N.jsonl` (104 scored + 2 abandoned read as
  # 106), and a partial episode killed with its container keeps the plain `<id>.jsonl` name until the same
  # task restarts (R1 s0: 105 files, 104 scored) — 115 scored + 1 partial would have read as complete.
  done_n=$(python3 "$(dirname "$0")/scored_count.py" "$RUNDIR") || done_n=0
  log "scored episodes on host: $done_n / $TARGET"
  if [ "$done_n" -ge "$TARGET" ]; then log "SUITE COMPLETE"; break; fi
done
