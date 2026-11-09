#!/usr/bin/env bash
# Nightly AndroidWorld regression: 8 fixed tasks, ~5 minutes, one agent.
#
# The alerts are the point. A harness that breaks silently is worse than no harness: it
# keeps producing numbers that look current. This exits non-zero on any alert.
#
# Deliberate difference from the plan's §8 step 1: it records the checked-out SHA instead of
# `git checkout`-ing a pinned one. A cron job that moves HEAD would silently change the code
# under a running session on a shared host.
#
# Usage:  ./run_nightly.sh                 # full run
#         DRY_RUN=1 ./run_nightly.sh       # plan + probe only, no episodes
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$HERE/configs/nightly_smoke.yaml}"
RUNS="${RUNS:-$HERE/runs}"
BASELINE="${BASELINE:-$RUNS/nightly_baseline.json}"
LOG="${LOG:-$RUNS/nightly.log}"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "$RUNS"

log() { printf '%s %s\n' "$STAMP" "$*" | tee -a "$LOG"; }

alert() {
  log "ALERT: $*"
  exit 1
}

SHA="$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "start sha=$SHA config=$CONFIG"

# 1. Syntax + plan first: a bad config must fail before any device is touched.
if ! python3 "$HERE/run_bench.py" --config "$CONFIG" --dry-run >"$RUNS/nightly_plan.txt" 2>&1; then
  cat "$RUNS/nightly_plan.txt" >>"$LOG"
  alert "config rejected (see nightly_plan.txt)"
fi
log "plan ok: $(grep -c 'pairing=' "$RUNS"/nightly_plan.txt) episodes"

# 2. Environment must be provably alive. /health answering is not evidence.
URL="$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['environment']['url'])" "$CONFIG")"
if ! python3 -c "import sys; sys.path.insert(0, '$HERE'); from awenv.liveness import main; main()" \
     --url "$URL" --layers all >>"$LOG" 2>&1; then
  alert "liveness probe failed for $URL"
fi
log "liveness ok"

if [ "${DRY_RUN:-0}" = "1" ]; then
  log "dry run complete (no episodes executed)"
  exit 0
fi

# 3. Run the suite. run_bench exits 1 when the regression comparison alerts.
if [ ! -f "$BASELINE" ]; then
  log "no baseline at $BASELINE yet - this run becomes the baseline"
  python3 "$HERE/run_bench.py" --config "$CONFIG" --run-root "$RUNS" --json >"$RUNS/latest.json" 2>>"$LOG" \
    || alert "run failed"
  cp "$RUNS/latest.json" "$BASELINE"
  log "baseline written: $BASELINE"
  exit 0
fi

if python3 "$HERE/run_bench.py" --config "$CONFIG" --run-root "$RUNS" --json >"$RUNS/latest.json" 2>>"$LOG"; then
  log "ok: no regression"
  exit 0
else
  alert "regression detected - see $RUNS/latest.json"
fi
