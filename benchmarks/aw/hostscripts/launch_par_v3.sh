#!/usr/bin/env bash
# Switch the running single-emulator v3 chain to two emulators in parallel (2026-11-11), without touching a running script:
#   1. stop the old chain (parent + its watchdog); its current run_arm chunk is left to finish,
#   2. wait until no run_arm.sh is running, then install the parameterized scripts and the two half configs,
#   3. start half A (aw-emu, run dir aw116_v3_s0, which keeps its scored episodes) and half B (aw-emu2, aw116_v3_s0_b).
set -u
A=$HOME/aw; LOG=$A/stage1_chain.log; NEW=$A/par_new
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
pkill -f "bash stage2_chain_v3.sh" && note "PAR SWITCH: old single-emulator chain stopped (its current chunk finishes)"
pkill -f "watchdog_runbench.sh $HOME/awruns/aw116_v3_s0 " || true
until ! pgrep -f "/run_arm\.sh" >/dev/null; do sleep 15; done
note "PAR SWITCH: old chunk finished; installing parameterized run_arm/watchdog/sweep and the half configs"
cp "$NEW/run_arm.sh" "$NEW/watchdog_runbench.sh" "$NEW/sweep_infra.py" "$NEW/stage2_chain_par.sh" "$A/"
cp "$NEW/aw116_v3_s0_a.yaml" "$A/aw116_v3_s0.yaml"
cp "$NEW/aw116_v3_s0_b.yaml" "$A/aw116_v3_s0_b.yaml"
cd "$A"
ID=aw116_v3_s0   TARGET=58 EMU=aw-emu  PORT=5051 setsid nohup bash stage2_chain_par.sh > "$A/stage2_par_a.out" 2>&1 < /dev/null &
sleep 30
ID=aw116_v3_s0_b TARGET=58 EMU=aw-emu2 PORT=5052 setsid nohup bash stage2_chain_par.sh > "$A/stage2_par_b.out" 2>&1 < /dev/null &
note "PAR SWITCH: started half A (aw-emu:5051) and half B (aw-emu2:5052), 58 tasks each"
