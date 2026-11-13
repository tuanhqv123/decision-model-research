#!/usr/bin/env bash
# Stage 1 chain v6, emulator host (2026-11-10). Runs after chain5: the Q11 completion.
#
# Why here: on the train host the Q11 completion was refused by preflight at 18:59 ("model reachable" FAIL). Our
# E2 server there had been shut down cleanly (SIGTERM, not OOM) after the host check, and a notebook kernel under
# the same shared account had taken 14.5 GB of that GPU. We do not compete for that GPU. Q11 calls no model and the
# image is byte-identical, so its 12 unscored tasks run here; the run directory was copied from the train host
# (210 files, index.db md5 3cd4b86d… on both).
#
# Usage: stage1_chain6.sh <chain5 pid>
set -u
A=$HOME/aw
LOG=$A/stage1_chain.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
scored() { ls ~/awruns/$1/episodes/*.jsonl 2>/dev/null | grep -vc '\.abandoned\.'; }
P=${1:?chain5 pid}

while kill -0 "$P" 2>/dev/null; do sleep 60; done
while ps -eo args | grep -q "[r]un_arm.sh"; do sleep 60; done
note "chain6: chain5 finished; Q11 completion next"

python3 "$A/staged_idfix/make_completion_cfg.py" "$A/aw116_q11_status_now.yaml" ~/awruns/aw116_q11_status_now \
  "$A/staged_idfix/aw116_tasks.txt" "$A/aw116_q11_completion.yaml" >> "$LOG" 2>&1
rc=$?
if [ $rc = 3 ]; then note "Q11 already complete"; exit 0; fi
[ $rc = 0 ] || { note "ABORT: completion config failed rc=$rc"; exit 1; }
curl -s -m 5 -H "x-api-key: tok" http://172.17.0.1:8701/health | grep -q '"contract":"e2"' \
  || "$A/serve_switch.sh" '~/runs/e2-final' e2 >> "$LOG" 2>&1 || { note "ABORT: e2 server not up"; exit 1; }
note "START aw116_q11_status_now completion harness=awhV2f.tgz (moved from the train host)"
RUNID=aw116_q11_status_now CFG="$A/aw116_q11_completion.yaml" CONTRACT=e2 HARNESS_TGZ="$HOME/awhV2f.tgz" TARGET=116 \
  bash "$A/run_arm.sh" 40 4 > "$A/aw116_q11_status_now.completion.log" 2>&1
note "END aw116_q11_status_now completion exit $? scored $(scored aw116_q11_status_now)"
note "STAGE1_CHAIN6_DONE"
