#!/bin/zsh
# Chain 4: create T4 session, VM pulls public data, smoke 10 steps — one uninterrupted chain.
set -u
cd /Users/macbookpro/Workspace/research
S=d12b
log() { echo "=== [$1] ==="; }

log "1. create T4 session (retry x4, 30s apart)"
ok=0
for i in 1 2 3 4; do
  if colab new -s $S --gpu T4 2>&1 | grep -q "Session READY"; then ok=1; break; fi
  echo "attempt $i failed, waiting 30s..."; sleep 30
done
[ $ok -eq 1 ] || { echo "FATAL: no T4 session"; exit 1; }
colab status -s $S 2>&1 | tail -1

log "2. VM prep: JSON (33MB) + images zip (6.6GB) + extract 398 shots"
colab exec -s $S --timeout 5400 -f benchmarks/colab/prep_vm.py 2>&1 | tail -8

log "3. SMOKE: 10 steps vision two-stage (downloads d1-2B ~2GB on fresh VM)"
colab exec -s $S --timeout 5400 --env TAG=d12b2s-smoke --env MODALITY=vision --env LIMIT=10 --env RESUME=0 -f benchmarks/colab/run_d12b_colab.py 2>&1 | tail -22

log "4. fetch smoke result (base64 via stdout)"
cat > /tmp/fetchres.py <<'PY'
import base64, os
p = "/content/data/results/d12b2s-smoke.jsonl"
if os.path.exists(p):
    b = base64.b64encode(open(p, "rb").read()).decode()
    for i in range(0, len(b), 100000):
        print(f"B64CHUNK {i} {b[i:i+100000]}")
    print(f"B64DONE total={len(b)}")
else:
    print("B64DONE total=0 NOFILE")
PY
colab exec -s $S --timeout 300 -f /tmp/fetchres.py 2>&1 | grep -oE "B64CHUNK [0-9]+ [A-Za-z0-9+/=]+|B64DONE total=[0-9]+ [A-Z]*" > /tmp/smoke_b64.txt
python3 - <<'PY'
import re, base64
txt = open("/tmp/smoke_b64.txt").read()
chunks = re.findall(r"B64CHUNK \d+ ([A-Za-z0-9+/=]+)", txt)
b64 = "".join(chunks)
done = re.search(r"B64DONE total=(\d+)", txt)
if b64:
    open("/tmp/d12b2s-smoke-colab.jsonl", "wb").write(base64.b64decode(b64))
    print(f"decoded {len(b64)} b64 chars -> /tmp/d12b2s-smoke-colab.jsonl (expect {done.group(1) if done else '?'})")
else:
    print("NO CHUNKS:", txt[:300])
PY
echo "CHAIN4_DONE"