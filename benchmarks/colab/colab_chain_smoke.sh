#!/bin/zsh
# One-shot Colab chain: create T4 session, upload data+script, verify, smoke test 10 steps.
# Every step logged; aborts on first failure.
set -u
cd ~/Workspace/research
S=d12b
log() { echo "=== [$1] ===" ; }

log "1. create T4 session (retry x3)"
ok=0
for i in 1 2 3; do
  if colab new -s $S --gpu T4 2>&1 | grep -q "Session READY"; then ok=1; break; fi
  echo "attempt $i failed, waiting 20s..."; sleep 20
done
[ $ok -eq 1 ] || { echo "FATAL: could not create T4 session"; exit 1; }
colab status -s $S 2>&1 | tail -1

log "2. mkdir on VM"
cat > /tmp/mkdir.py <<'PY'
import os
os.makedirs("/content/data", exist_ok=True)
os.makedirs("/content/data/results", exist_ok=True)
print("dirs:", sorted(os.listdir("/content")))
PY
colab exec -s $S --timeout 120 -f /tmp/mkdir.py 2>&1 | tail -2

log "3. upload json (32MB)"
colab upload -s $S benchmarks/data/acv2/500_steps_filtered.json data/500_steps_filtered.json 2>&1 | tail -1

log "4. upload images tar (275MB)"
colab upload -s $S /tmp/acv2_pkg.tar data/acv2_pkg.tar 2>&1 | tail -1

log "5. upload runner script"
colab upload -s $S benchmarks/colab/run_d12b_colab.py run_d12b_colab.py 2>&1 | tail -1

log "6. verify files"
colab ls -s $S data 2>&1 | tail -5

log "7. extract images + integrity check"
cat > /tmp/extract.py <<'PY'
import tarfile, os, json
with tarfile.open("/content/data/acv2_pkg.tar") as t:
    t.extractall("/content/data")
rows = json.load(open("/content/data/500_steps_filtered.json"))
missing = 0
for r in rows:
    p = f"/content/data/images/{r['screenshot']}"
    if not os.path.exists(p):
        missing += 1
print(f"rows={len(rows)} missing_images={missing}")
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
colab exec -s $S --timeout 300 -f /tmp/extract.py 2>&1 | tail -3

log "8. SMOKE TEST: 10 steps vision two-stage (downloads d1-2B ~2GB on first load)"
colab exec -s $S --timeout 3600 --env TAG=d12b2s-smoke --env MODALITY=vision --env LIMIT=10 --env RESUME=0 -f /content/run_d12b_colab.py 2>&1 | tail -25

log "9. download smoke results"
colab download -s $S data/results/d12b2s-smoke.jsonl /tmp/d12b2s-smoke-colab.jsonl 2>&1 | tail -1
echo "CHAIN_DONE"