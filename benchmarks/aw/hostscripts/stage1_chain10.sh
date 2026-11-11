#!/usr/bin/env bash
# Stage 1 chain v10, emulator host (2026-11-10). After chain7: R1 seeds 1 and 2 online.
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
# chain10 (2026-11-11): seed 2 again died on BrowserDraw (env_crash after 8 steps, then every episode of the chunk a setup error).
# Chunks are now 3 episodes and `sweep_infra.py` sets aside emulator-death episodes after each chunk (max 2 retries per task).
# Usage: stage1_chain10.sh
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

seed2_loop() {
  local id=aw116_r1s2 ck=v1_s2 i n
  serve_exact v1 "~/runs/$ck" || return 1
  for i in $(seq 1 80); do
    n=$(scored $id); [ "$n" -ge 116 ] && break
    RUNID=$id CFG="$A/$id.yaml" CONTRACT=v1 HARNESS_TGZ="$HOME/awhV4.tgz" TARGET=116 \
      bash "$A/run_arm.sh" 3 1 > "$A/$id.loop_$i.log" 2>&1
    moved=$(sudo -n python3 "$A/sweep_infra.py" ~/awruns/$id | tr '\n' ' ')
    # evidence for the cause of the death, taken BEFORE the next round restarts the container (which discards its log)
    if [ -n "$moved" ]; then
      sudo -n docker logs --tail 600 aw-emu 2>&1 | grep -v "closure of unused" | tail -150 > ~/awruns/$id/abandoned_infra/emu_log_loop_$i.txt
      sudo -n docker exec aw-emu bash -c 'dmesg 2>/dev/null | tail -20; ls /root/.android/avd/*.avd 2>/dev/null | head -3' >> ~/awruns/$id/abandoned_infra/emu_log_loop_$i.txt 2>&1
    fi
    note "$id loop $i: scored $(scored $id)${moved:+, set aside (emulator death): $moved}"
  done
  note "$id loop ended: $(scored $id) scored"; audit $id $ck
}

note "chain10: R1 seed 2 resumes in chunks of 3 with infra sweep"
seed2_loop
note "STAGE1_CHAIN10_DONE"
