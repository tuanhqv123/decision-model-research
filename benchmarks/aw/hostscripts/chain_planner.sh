#!/usr/bin/env bash
# LLM planner + executor trial (2026-11-16): one chain per emulator; rounds of run_arm_v2 until TARGET scored episodes.
# Usage: ID=... TARGET=... EMU=... PORT=... CK=... MODEL_HEALTH=http://172.17.0.1:8711 HARNESS=~/awhV8p.tgz LLM_KEY_FILE=~/aw/llm_key LLM_URL_FILE=~/aw/llm_url chain_planner.sh
set -u
A=$HOME/aw; LOG=$A/stage1_chain.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
scored() { python3 "$A/scored_count.py" ~/awruns/$1; }
curl -s -m 5 -H "x-api-key: tok" "$MODEL_HEALTH/health" | grep -q "/$CK\"" || { note "SERVE MISMATCH $ID: wanted $CK"; exit 1; }
note "START $ID cfg=$ID.yaml (planner)"
mkdir -p ~/awruns/$ID
for i in $(seq 1 30); do
  [ "$(scored $ID)" -ge "$TARGET" ] && break
  RUNID=$ID CFG="$A/$ID.yaml" CONTRACT=v4 HARNESS_TGZ="$HARNESS" TARGET=$TARGET LLM_KEY_FILE="$LLM_KEY_FILE" LLM_URL_FILE="$LLM_URL_FILE" \
    bash "$A/run_arm_v2.sh" 3 1 > "$A/$ID.loop_$i.log" 2>&1
  moved=$(sudo -n python3 "$A/sweep_infra.py" ~/awruns/$ID --container "$EMU" | tr '\n' ' ')
  note "$ID loop $i: scored $(scored $ID)${moved:+, set aside (emulator death): $moved}"
done
note "PLANNER_DONE $ID: $(scored $ID) scored"
