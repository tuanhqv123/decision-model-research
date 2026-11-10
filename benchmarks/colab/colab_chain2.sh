#!/bin/zsh
# Chain part 2: tar upload to root, extract, smoke 10 steps.
set -u
cd ~/Workspace/research
S=d12b
log() { echo "=== [$1] ==="; }

log "1. upload images tar (275MB) to root"
colab upload -s $S /tmp/acv2_pkg.tar acv2_pkg.tar 2>&1 | tail -1

log "2. extract + move json + integrity"
cat > /tmp/extract.py <<'PY'
import tarfile, os, json, shutil
os.makedirs("/content/data", exist_ok=True)
os.makedirs("/content/data/results", exist_ok=True)
if os.path.exists("/content/data/500_steps_filtered.json"):
    os.remove("/content/data/500_steps_filtered.json")
shutil.move("/content/500_steps_filtered.json", "/content/data/500_steps_filtered.json")
with tarfile.open("/content/acv2_pkg.tar") as t:
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
colab exec -s $S --timeout 600 -f /tmp/extract.py 2>&1 | tail -3

log "3. SMOKE: 10 steps vision two-stage (-f LOCAL path; first run downloads d1-2B)"
colab exec -s $S --timeout 5400 --env TAG=d12b2s-smoke --env MODALITY=vision --env LIMIT=10 --env RESUME=0 -f benchmarks/colab/run_d12b_colab.py 2>&1 | tail -25

log "4. copy results to root + download"
cat > /tmp/copyres.py <<'PY'
import shutil, os
src = "/content/data/results/d12b2s-smoke.jsonl"
if os.path.exists(src):
    shutil.copy(src, "/content/d12b2s-smoke.jsonl")
    print("copied", src)
else:
    print("NO RESULT FILE:", src)
PY
colab exec -s $S --timeout 120 -f /tmp/copyres.py 2>&1 | tail -1
colab download -s $S d12b2s-smoke.jsonl /tmp/d12b2s-smoke-colab.jsonl 2>&1 | tail -1
echo "CHAIN2_DONE"