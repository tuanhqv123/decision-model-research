#!/bin/bash
# diag_ck.sh CKPT_DIR TAG : two diagnostic evals of one checkpoint (validation AC-Low style, and a train-episode sample in
# the executor format). Diagnostic only, never used to pick a checkpoint. Exits 3 if another user holds the GPU.
CK=$1; T=$2; O=~/d1pilot/diag/out; mkdir -p $O
foreign() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  a=$(ps -o args= -p $p 2>/dev/null); case "$a" in *src.main*|*eval_d1*|*diag*) ;; *) echo $p;; esac; done; }
[ -n "$(foreign)" ] && { echo "$(date -Is) YIELD before $T: $(foreign)" | tee -a $O/YIELDED; exit 3; }
cd ~/d1pilot/code_exec/benchmarks/train
export D1_IMG_W=720 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { ~/acenv/bin/python -c "
import sys, runpy, torch
torch.cuda.set_per_process_memory_fraction(0.40)
sys.argv = ['eval_d1.py', '--ckpt', '$CK', '--data', '$1', '--val', '--val-set', '$2', '--track', 'VA', '--tag', '$3', '--out', '$O', '--batch', '4'] + '$4'.split()
runpy.run_path('eval_d1.py', run_name='__main__')" > $O/eval_$3.log 2>&1; echo "EVAL $3 exit $?"; }
run ~/d1pilot/g2val validation_v4 ${T}_vallow "--instruction" &
run ~/d1pilot/diag/data trainsamp_exec ${T}_train "--instruction-per-row" &
( while sleep 60; do f=$(foreign); [ -n "$f" ] && { echo "$(date -Is) YIELD during $T: $f" >> $O/YIELDED; pkill -f "diag/out"; break; }; done ) & W=$!
wait %1 %2; kill $W 2>/dev/null
[ -f $O/YIELDED ] && exit 3 || exit 0
