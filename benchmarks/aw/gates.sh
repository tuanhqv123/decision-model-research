#!/usr/bin/env bash
# Gate A -> Gate B, in one run, with every claim checked by a command.
#
# Gate A: the environment is correct (four liveness layers, every emittable action round-trips,
#         type_text handles spaces and non-ASCII, open_app reaches all 24 canonical apps).
# Gate B: two runs of the same 5-task config at the same seed produce an identical per-episode
#         success vector, and the run is replayable from JSONL.
#
# Exits non-zero at the first gate that fails. Nothing is reported as a benchmark number.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="${RUNS:-$HERE/runs}"
mkdir -p "$RUNS"

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "Gate A: environment correctness"
# `python -m pytest`, never bare `pytest`: inside the android_world image the entry point uses a
# different interpreter than the one android_world is installed into (found 2026-11-06).
if ! AW_GATE=1 python3 -m pytest "$HERE/tests/test_device_gate_a.py" -q 2>&1 | tee "$RUNS/gate_a.log" | tail -3; then
  log "GATE A FAILED - no benchmark number is produced"
  exit 1
fi
if grep -qE "[0-9]+ (failed|error)" "$RUNS/gate_a.log"; then
  log "GATE A FAILED - see $RUNS/gate_a.log"
  exit 1
fi
log "Gate A passed"

log "Gate B: run 1"
python3 "$HERE/run_bench.py" --config "$HERE/configs/gate_b.yaml" --run-root "$RUNS/gate_b/run1" \
  --json > "$RUNS/gate_b_run1.json" 2>>"$RUNS/gate_b.log" || { log "GATE B run 1 failed"; exit 1; }

log "Gate B: run 2 (same seed, fresh run dir)"
python3 "$HERE/run_bench.py" --config "$HERE/configs/gate_b.yaml" --run-root "$RUNS/gate_b/run2" \
  --json > "$RUNS/gate_b_run2.json" 2>>"$RUNS/gate_b.log" || { log "GATE B run 2 failed"; exit 1; }

python3 - "$RUNS/gate_b_run1.json" "$RUNS/gate_b_run2.json" <<'PY'
import json, sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
va = {e["episode_id"]: e["success"] for e in a["episodes"]}
vb = {e["episode_id"]: e["success"] for e in b["episodes"]}

missing = set(va) ^ set(vb)
if missing:
    print("GATE B FAILED: different episode sets: %s" % sorted(missing)[:4])
    raise SystemExit(1)

moved = [k for k in va if va[k] != vb[k]]
if moved:
    print("GATE B FAILED: %d episode(s) changed at the same seed: %s" % (len(moved), moved[:4]))
    raise SystemExit(1)

if not va:
    print("GATE B FAILED: no episodes recorded")
    raise SystemExit(1)

print("GATE B PASSED: %d/%d identical, SR %d/%d" % (
    len(va), len(va), sum(va.values()), len(va)))
PY
rc=$?
if [ $rc -ne 0 ]; then log "GATE B FAILED"; exit 1; fi

log "Gate B passed - the harness is trustworthy enough to produce a number"
log "Next: Gate C (15-20 task pilot, error taxonomy, stable across two seeds)"
