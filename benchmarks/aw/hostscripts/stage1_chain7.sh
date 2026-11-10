#!/usr/bin/env bash
# Stage 1 chain v7, emulator host (2026-11-10). Replaces chain6, stopped before it ran anything.
#
# Why: R1 s0 ended at 20:34 UTC with 104 of 116 tasks scored. The a11y forwarder died, preflight refused the next
# chunk, and the container died during the one restart run_arm allows, so run_arm exited 1. The 12 unscored tasks
# are the last 12 of the run order (SystemWifiTurnOn .. VlcCreateTwoPlaylists), none of them shadowed by H11, so
# neither the primary (116) nor the A3 sensitivity (104) set is complete for R1. Its completion is on the critical
# path of the pre-registered A1 decision and Q11's is not, so it runs first.
#
# Every completion is checked from the run's index (make_completion_cfg exit 3 = nothing left) and gets up to 3
# passes, because one pass can end early on a container death. run_arm.sh now counts scored episodes from the
# index too (scored_count.py): R1 had 105 episode files for 104 scored episodes, the extra one a partial episode
# killed with its container.
#
# Usage: stage1_chain7.sh <chain5 pid>
set -u
A=$HOME/aw
LOG=$A/stage1_chain.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
scored() { python3 "$A/scored_count.py" ~/awruns/$1; }
P=${1:?chain5 pid}

serve() {  # serve <contract> <ckpt>: switch only if the right checkpoint under the right contract is not up already
  local h
  h=$(curl -s -m 5 -H "x-api-key: tok" http://172.17.0.1:8701/health)
  if echo "$h" | grep -q "\"contract\":\"$1\"" && echo "$h" | grep -q "/$(basename "$2")\""; then return 0; fi
  "$A/serve_switch.sh" "$2" "$1" >> "$LOG" 2>&1
}

complete() {  # complete <runid> <parent cfg> <contract> <ckpt> <harness> <chunk> <rounds>
  local id=$1 parent=$2 contract=$3 ckpt=$4 harness=$5 chunk=$6 rounds=$7 pass rc
  for pass in 1 2 3 4; do
    python3 "$A/staged_idfix/make_completion_cfg.py" "$parent" ~/awruns/$id "$A/staged_idfix/aw116_tasks.txt" \
      "$A/${id}_completion.yaml" >> "$LOG" 2>&1
    rc=$?
    if [ $rc = 3 ]; then note "$id complete: $(scored $id) scored"; return 0; fi
    [ $rc = 0 ] || { note "ABORT: completion config for $id failed rc=$rc"; return 1; }
    [ $pass = 4 ] && break
    serve "$contract" "$ckpt" || { note "ABORT: $contract server not up"; return 1; }
    note "START $id completion pass $pass harness=$(basename "$harness")"
    RUNID=$id CFG="$A/${id}_completion.yaml" CONTRACT=$contract HARNESS_TGZ="$harness" TARGET=116 \
      bash "$A/run_arm.sh" "$chunk" "$rounds" > "$A/$id.completion_c7_$pass.log" 2>&1
    note "END $id completion pass $pass exit $? scored $(scored $id)"
  done
  note "$id still incomplete after 3 passes: $(scored $id) scored"; return 1
}

while kill -0 "$P" 2>/dev/null; do sleep 60; done
while ps -eo args | grep -q "[r]un_arm.sh"; do sleep 60; done
note "chain7: chain5 finished; B0' re-check, then R1 completion, then Q11 completion"

complete aw116_b0prime "$A/aw116_b0prime.yaml" e2 '~/runs/e2-final' "$HOME/awhV2f.tgz" 12 4
complete aw116_r1s0 "$A/aw116_r1s0.yaml" v1 '~/runs/v1_s0' "$HOME/awhV4.tgz" 12 4
complete aw116_q11_status_now "$A/aw116_q11_status_now.yaml" e2 '~/runs/e2-final' "$HOME/awhV2f.tgz" 40 4
note "STAGE1_CHAIN7_DONE"
