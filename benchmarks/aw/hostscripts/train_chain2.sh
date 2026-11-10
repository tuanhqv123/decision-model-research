#!/usr/bin/env bash
# Train-host chain v2 (2026-11-10). After the host-equivalence check (harness v2, 20 tasks, none shadowed)
# ends: install the runner and preflight with the episode-id fix, then complete Q11 — every suite task with no
# scored episode in its run directory (the 12 that `_slug` hid, plus any task an infra abort left), harness v2f.
set -u
A=$HOME/aw
LOG=$A/train_chain.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
scored() { ls ~/awruns/$1/episodes/*.jsonl 2>/dev/null | grep -vc '\.abandoned\.'; }

while ps -eo args | grep -q "[r]un_arm.sh"; do sleep 60; done
note "chain2: no runner active (host check ended, scored $(scored aw20_hostcheck_train))"

cp "$A/run_arm.sh" "$A/run_arm.sh.pre_idfix"; cp "$A/preflight.py" "$A/preflight.py.pre_idfix"
cp "$A/staged_idfix/run_arm.sh" "$A/run_arm.sh.new" && mv "$A/run_arm.sh.new" "$A/run_arm.sh"
cp "$A/staged_idfix/preflight.py" "$A/preflight.py"
note "installed run_arm.sh + preflight.py with the episode-id fix"

python3 "$A/staged_idfix/make_completion_cfg.py" "$A/aw116_q11_status_now.yaml" ~/awruns/aw116_q11_status_now \
  "$A/staged_idfix/aw116_tasks.txt" "$A/aw116_q11_completion.yaml" >> "$LOG" 2>&1
rc=$?
if [ $rc = 3 ]; then note "Q11 already complete"; exit 0; fi
[ $rc = 0 ] || { note "ABORT: completion config failed rc=$rc"; exit 1; }
note "START aw116_q11_status_now completion harness=awhV2f.tgz"
RUNID=aw116_q11_status_now CFG="$A/aw116_q11_completion.yaml" CONTRACT=e2 HARNESS_TGZ="$HOME/awhV2f.tgz" TARGET=116 \
  bash "$A/run_arm.sh" 40 4 > "$A/aw116_q11_status_now.completion.log" 2>&1
note "END aw116_q11_status_now completion exit $? scored $(scored aw116_q11_status_now)"
