#!/usr/bin/env bash
# PARALLEL variant (2026-11-12): model on the second API (port 8702); never switches the shared server.
# v4 seed 0 online (2026-11-12), paired with aw116_v3h_s0 (same harness awhV8). Worklog docs/research/2026-11-12-V4-WORKLOG-vi.md.
# Serves the v2a checkpoint (health must name it), runs the 116 tasks in chunks of 3 with the infra sweep after every
# chunk, keeps a run_bench watchdog alive during the run, then audits every preflight health line.
# Usage: ID=... TARGET=... EMU=... PORT=... [HARNESS=~/awhV6.tgz] stage2_chain_par2.sh   (one chain per emulator; each its own run dir and config $ID.yaml)
# The checkpoint must already be at ~/runs/$CK on the serving host; serving_v3 sets D1_APPS.
set -u
A=$HOME/aw
LOG=$A/stage1_chain.log
ID=${ID:-aw116_v3_s0}; CK=${CK:-v3_s0_ckpt1824}; CONTRACT=${CONTRACT:-v3}
TARGET=${TARGET:-116}; export EMU=${EMU:-aw-emu} PORT=${PORT:-5051}   # one chain per emulator container
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
scored() { python3 "$A/scored_count.py" ~/awruns/$1; }
health() { curl -s -m 5 -H "x-api-key: tok" http://172.17.0.1:8702/health; }
serving() { local h; h=$(health); echo "$h" | grep -q "\"contract\":\"$1\"" && echo "$h" | grep -q "/$2\""; }

serving "$CONTRACT" "$CK" || { note "SERVE MISMATCH on 8702: wanted $CK"; exit 1; }
for i in 1 2 3 4 5 6; do serving "$CONTRACT" "$CK" && break; sleep 10; done
serving "$CONTRACT" "$CK" || { note "SERVE MISMATCH: wanted $CK, health says $(health | cut -c1-120)"; exit 1; }
note "START $ID cfg=$ID.yaml contract=$CONTRACT"
mkdir -p ~/awruns/$ID
bash "$A/watchdog_runbench.sh" ~/awruns/$ID 25 > "$A/watchdog_$ID.out" 2>&1 &
WD=$!
for i in $(seq 1 90); do
  [ "$(scored $ID)" -ge "$TARGET" ] && break
  RUNID=$ID CFG="$A/$ID.yaml" CONTRACT=$CONTRACT HARNESS_TGZ="${HARNESS:-$HOME/awhV5.tgz}" TARGET=$TARGET \
    bash "$A/run_arm_v2.sh" 3 1 > "$A/$ID.loop_$i.log" 2>&1
  moved=$(sudo -n python3 "$A/sweep_infra.py" ~/awruns/$ID --container "$EMU" | tr '\n' ' ')
  note "$ID loop $i: scored $(scored $ID)${moved:+, set aside (emulator death): $moved}"
done
kill "$WD" 2>/dev/null
total=$(cat "$A/$ID".loop_*.log 2>/dev/null | grep -c "model reachable *PASS")
bad=$(cat "$A/$ID".loop_*.log 2>/dev/null | grep "model reachable *PASS" | grep -vc "/$CK\"")
if [ "$total" -gt 0 ] && [ "$bad" = 0 ]; then note "AUDIT OK $ID: $total preflight health lines, all name $CK"
else note "AUDIT FAILED $ID: $bad of $total preflight health lines do not name $CK — DO NOT ANALYSE"; fi
note "$ID done: $(scored $ID) scored"
note "STAGE2_PAR_DONE $ID"
