#!/usr/bin/env bash
# Stage 1 chain v4, emulator host (2026-11-10). Replaces chain3, which was stopped before it ran anything.
#
# Why: every 116-task run so far skipped 12 tasks. `_slug` cut task names at 24 characters, so e.g.
# SimpleCalendarAddOneEventTomorrow shared SimpleCalendarAddOneEvent's episode id and resumption treated it
# as finished. Fixed harness packages: v2f = v2 + the id fix only; v4 = v3 + the id fix only.
#
# Order (everything that is paired with B0' runs on B0''s host):
#   B0' (harness v2, already running) -> B0' completion: every suite task with no scored episode in its run
#   directory, harness v2f -> R1 s0, all 116, harness v4 -> Q10 noise (20 tasks, none of them shadowed), v2f.
set -u
A=$HOME/aw
LOG=$A/stage1_chain.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
scored() { ls ~/awruns/$1/episodes/*.jsonl 2>/dev/null | grep -vc '\.abandoned\.'; }

while ps -eo args | grep -q "[r]un_arm.sh"; do sleep 60; done
note "END aw116_b0prime (harness v2) scored $(scored aw116_b0prime)"

# The fixed runner and preflight go in only now: the new preflight refuses an unfixed harness, so swapping it
# in while B0' was running would have stopped B0' before its last ordinary tasks. mv gives run_arm.sh a new
# inode, so no running bash ever reads a half-replaced script.
cp "$A/run_arm.sh" "$A/run_arm.sh.pre_idfix"; cp "$A/preflight.py" "$A/preflight.py.pre_idfix"
cp "$A/staged_idfix/run_arm.sh" "$A/run_arm.sh.new" && mv "$A/run_arm.sh.new" "$A/run_arm.sh"
cp "$A/staged_idfix/preflight.py" "$A/preflight.py"
note "installed run_arm.sh + preflight.py with the episode-id fix"

run() {  # run <runid> <cfg> <contract> <harness> <target> <chunk> <rounds> <logtag>
  note "START $1 cfg=$(basename $2) harness=$(basename $4) target=$5"
  RUNID=$1 CFG=$2 CONTRACT=$3 HARNESS_TGZ=$4 TARGET=$5 bash "$A/run_arm.sh" $6 $7 > "$A/$1.$8.log" 2>&1
  note "END $1 exit $? scored $(scored $1)"
}

python3 "$A/staged_idfix/make_completion_cfg.py" "$A/aw116_b0prime.yaml" ~/awruns/aw116_b0prime \
  "$A/staged_idfix/aw116_tasks.txt" "$A/aw116_b0prime_completion.yaml" >> "$LOG" 2>&1
rc=$?
if [ $rc = 0 ]; then
  run aw116_b0prime "$A/aw116_b0prime_completion.yaml" e2 "$HOME/awhV2f.tgz" 116 12 4 completion
elif [ $rc != 3 ]; then
  note "ABORT: completion config failed rc=$rc"; exit 1
fi

"$A/serve_switch.sh" '~/runs/v1_s0' v1 >> "$LOG" 2>&1 || { note "ABORT: v1 server not up"; exit 1; }
run aw116_r1s0 "$A/aw116_r1s0.yaml" v1 "$HOME/awhV4.tgz" 116 12 16 main
"$A/serve_switch.sh" '~/runs/e2-final' e2 >> "$LOG" 2>&1 || { note "ABORT: e2 server not up"; exit 1; }
run aw20_q10_noise "$A/aw20_q10_noise.yaml" e2 "$HOME/awhV2f.tgz" 20 12 3 main
note "STAGE1_CHAIN4_DONE"
