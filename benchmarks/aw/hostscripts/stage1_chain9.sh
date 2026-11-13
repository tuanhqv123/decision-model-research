#!/usr/bin/env bash
# Stage 1 chain v9, emulator host (2026-11-10). After chain7: R1 seeds 1 and 2 online.
#
# Why: amendment A1 fired on seed 0 (|w - l| = 6 on the primary 116, 7 on the A3 sensitivity 104; the two sets agree
# on the decision and on the sign). Seeds 1 and 2 run on B0''s host with seed 0's harness (v4), protocol, chunking
# and completion-from-the-index procedure; the configs differ from seed 0's only in `agent` and `checkpoint`.
#
# serve_switch.sh accepts any server that speaks the right CONTRACT, and all three seeds speak v1, so a switch that
# failed (e.g. the new server cannot bind while the old one still answers) would leave v1_s0 serving under a seed-1
# label. This chain therefore (a) requires the health check to name the exact checkpoint before every run and
# (b) after a seed is complete, requires every preflight "model reachable" line of that seed's logs to name it.
#
# chain9 (2026-11-11): R1 seed 2 only. The emulator died inside seed 2's first chunk after episode 3 (adb: device
# 'emulator-5554' not found); the other 10 episodes of the chunk were recorded as setup errors and counted as scored.
# They were set aside as <id>.abandoned.1.jsonl and the index rebuilt, so this chain RESUMES seed 2 by task name.
# Usage: stage1_chain9.sh
set -u
A=$HOME/aw
LOG=$A/stage1_chain.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
scored() { python3 "$A/scored_count.py" ~/awruns/$1; }
health() { curl -s -m 5 -H "x-api-key: tok" http://172.17.0.1:8701/health; }
serving() { local h; h=$(health); echo "$h" | grep -q "\"contract\":\"$1\"" && echo "$h" | grep -q "/$2\""; }

serve_exact() {  # serve_exact <contract> <ckpt dir on the serving host>
  local b i
  b=$(basename "$2")
  serving "$1" "$b" && return 0
  "$A/serve_switch.sh" "$2" "$1" >> "$LOG" 2>&1
  for i in 1 2 3 4 5 6; do serving "$1" "$b" && return 0; sleep 10; done
  note "SERVE MISMATCH: wanted $b, health says $(health | cut -c1-120)"
  return 1
}

audit() {  # audit <runid> <ckpt basename> (PASS lines only: a refused chunk ran no episode)
  local total bad
  total=$(cat "$A/$1".*.log 2>/dev/null | grep -c "model reachable *PASS")
  bad=$(cat "$A/$1".*.log 2>/dev/null | grep "model reachable *PASS" | grep -vc "/$2\"")
  if [ "$total" -gt 0 ] && [ "$bad" = 0 ]; then
    note "AUDIT OK $1: $total preflight health lines, all name $2"
  else
    note "AUDIT FAILED $1: $bad of $total preflight health lines do not name $2 — DO NOT ANALYSE"
  fi
}

seed() {  # seed <s>
  local s=$1 id=aw116_r1s$1 ck=v1_s$1 pass rc
  for pass in 0 1 2 3 4; do
    if [ $pass -gt 0 ]; then
      python3 "$A/staged_idfix/make_completion_cfg.py" "$A/$id.yaml" ~/awruns/$id "$A/staged_idfix/aw116_tasks.txt" \
        "$A/${id}_completion.yaml" >> "$LOG" 2>&1
      rc=$?
      if [ $rc = 3 ]; then note "$id complete: $(scored $id) scored"; audit $id $ck; return 0; fi
      [ $rc = 0 ] || { note "ABORT: completion config for $id failed rc=$rc"; return 1; }
      [ $pass = 4 ] && break
    fi
    serve_exact v1 "~/runs/$ck" || return 1
    if [ $pass = 0 ]; then
      note "START $id cfg=$id.yaml harness=awhV4.tgz target=116"
      RUNID=$id CFG="$A/$id.yaml" CONTRACT=v1 HARNESS_TGZ="$HOME/awhV4.tgz" TARGET=116 \
        bash "$A/run_arm.sh" 12 16 > "$A/$id.main.log" 2>&1
      note "END $id exit $? scored $(scored $id)"
    else
      note "START $id completion pass $pass"
      RUNID=$id CFG="$A/${id}_completion.yaml" CONTRACT=v1 HARNESS_TGZ="$HOME/awhV4.tgz" TARGET=116 \
        bash "$A/run_arm.sh" 12 4 > "$A/$id.completion_c8_$pass.log" 2>&1
      note "END $id completion pass $pass exit $? scored $(scored $id)"
    fi
  done
  note "$id still incomplete after 3 completion passes: $(scored $id) scored"; audit $id $ck; return 1
}

note "chain9: R1 seed 2 resumes"
seed 2
note "STAGE1_CHAIN9_DONE"
