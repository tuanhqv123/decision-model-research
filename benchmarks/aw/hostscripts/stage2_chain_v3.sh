#!/usr/bin/env bash
# v3 seed 0 online (2026-11-11). Pre-registration: docs/research/PRE-REG-2026-11-11-v3.md.
# Serves the v2a checkpoint (health must name it), runs the 116 tasks in chunks of 3 with the infra sweep after every
# chunk, keeps a run_bench watchdog alive during the run, then audits every preflight health line.
# Usage: stage2_chain_v3.sh   (the checkpoint must already be at ~/runs/v3_s0_ckpt1824 on the serving host; serving_v3 sets D1_APPS)
set -u
A=$HOME/aw
LOG=$A/stage1_chain.log
ID=aw116_v3_s0; CK=v3_s0_ckpt1824; CONTRACT=v3   # ~/runs/v3_s0_ckpt1824 -> ~/runs/v3_s0/ckpt_1824 (final, pre-registered choice)
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
scored() { python3 "$A/scored_count.py" ~/awruns/$1; }
health() { curl -s -m 5 -H "x-api-key: tok" http://172.17.0.1:8701/health; }
serving() { local h; h=$(health); echo "$h" | grep -q "\"contract\":\"$1\"" && echo "$h" | grep -q "/$2\""; }

serving "$CONTRACT" "$CK" || bash "$A/serve_switch_v3.sh" "~/runs/$CK" "$CONTRACT" >> "$LOG" 2>&1
for i in 1 2 3 4 5 6; do serving "$CONTRACT" "$CK" && break; sleep 10; done
serving "$CONTRACT" "$CK" || { note "SERVE MISMATCH: wanted $CK, health says $(health | cut -c1-120)"; exit 1; }
note "START $ID cfg=$ID.yaml contract=$CONTRACT"
mkdir -p ~/awruns/$ID
bash "$A/watchdog_runbench.sh" ~/awruns/$ID 25 > "$A/watchdog_$ID.out" 2>&1 &
WD=$!
for i in $(seq 1 90); do
  [ "$(scored $ID)" -ge 116 ] && break
  RUNID=$ID CFG="$A/$ID.yaml" CONTRACT=$CONTRACT HARNESS_TGZ="$HOME/awhV5.tgz" TARGET=116 \
    bash "$A/run_arm.sh" 3 1 > "$A/$ID.loop_$i.log" 2>&1
  moved=$(sudo -n python3 "$A/sweep_infra.py" ~/awruns/$ID | tr '\n' ' ')
  note "$ID loop $i: scored $(scored $ID)${moved:+, set aside (emulator death): $moved}"
done
kill "$WD" 2>/dev/null
total=$(cat "$A/$ID".loop_*.log 2>/dev/null | grep -c "model reachable *PASS")
bad=$(cat "$A/$ID".loop_*.log 2>/dev/null | grep "model reachable *PASS" | grep -vc "/$CK\"")
if [ "$total" -gt 0 ] && [ "$bad" = 0 ]; then note "AUDIT OK $ID: $total preflight health lines, all name $CK"
else note "AUDIT FAILED $ID: $bad of $total preflight health lines do not name $CK — DO NOT ANALYSE"; fi
note "$ID done: $(scored $ID) scored"
note "STAGE2_V3_DONE"
