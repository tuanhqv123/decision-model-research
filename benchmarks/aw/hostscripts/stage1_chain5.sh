#!/usr/bin/env bash
# Stage 1 chain v5, emulator host (2026-11-10). Follow-up to chain4, started while chain4 runs (a running bash
# script must not be edited in place).
#
# Why: chain4's B0' completion was refused by preflight at 18:55 (no episode ran): the first awhV2f/awhV4
# packages had been re-packed from `aw/` only and lost `train/ac_common.py`, which the agent's history renderer
# imports. Both packages were rebuilt from the whole tree (whole-tree diff vs parent = runner.py [+ its test]) and
# replaced at 18:57, before R1's first install. chain4 then went on to R1 and Q10; this runs the B0' completion
# after them, on the E2 server chain4 leaves switched in.
#
# Usage: stage1_chain5.sh <chain4 pid>
set -u
A=$HOME/aw
LOG=$A/stage1_chain.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
scored() { ls ~/awruns/$1/episodes/*.jsonl 2>/dev/null | grep -vc '\.abandoned\.'; }
P=${1:?chain4 pid}

while kill -0 "$P" 2>/dev/null; do sleep 60; done
while ps -eo args | grep -q "[r]un_arm.sh"; do sleep 60; done
note "chain5: chain4 finished; B0' completion next"

"$A/serve_switch.sh" '~/runs/e2-final' e2 >> "$LOG" 2>&1 || { note "ABORT: e2 server not up"; exit 1; }
python3 "$A/staged_idfix/make_completion_cfg.py" "$A/aw116_b0prime.yaml" ~/awruns/aw116_b0prime \
  "$A/staged_idfix/aw116_tasks.txt" "$A/aw116_b0prime_completion.yaml" >> "$LOG" 2>&1
rc=$?
if [ $rc = 3 ]; then note "B0' already complete"; exit 0; fi
[ $rc = 0 ] || { note "ABORT: completion config failed rc=$rc"; exit 1; }
note "START aw116_b0prime completion harness=awhV2f.tgz (rebuilt 18:57)"
RUNID=aw116_b0prime CFG="$A/aw116_b0prime_completion.yaml" CONTRACT=e2 HARNESS_TGZ="$HOME/awhV2f.tgz" TARGET=116 \
  bash "$A/run_arm.sh" 12 4 > "$A/aw116_b0prime.completion2.log" 2>&1
note "END aw116_b0prime completion exit $? scored $(scored aw116_b0prime)"
note "STAGE1_CHAIN5_DONE"
