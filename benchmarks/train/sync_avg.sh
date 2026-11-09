#!/bin/bash
# Local-SGD hub. Runs ON the hub host (its own replica writes to ~/acwork/runs/$RUN/sync) under `ssh -A`.
# For each round k: wait for rep_r<k>.done on every replica, gather, average, push avg_r<k>, mark done.
# Args: RUN ROUNDS "user@host user@host ..." (the OTHER replicas)
RUN=$1; ROUNDS=$2; REMOTES=$3
HUB=~/acwork/runs/$RUN/sync
SSH="ssh -p 234 -o StrictHostKeyChecking=no -o BatchMode=yes -o LogLevel=ERROR"
mkdir -p $HUB/gather
for k in $(seq 1 $ROUNDS); do
  if [ -f $HUB/avg_r$k.done ]; then echo "[hub] round $k: already done"; continue; fi  # resumable
  echo "[hub] round $k: waiting for replicas $(date +%T)"
  while true; do
    ok=1; [ -f $HUB/rep_r$k.done ] || ok=0
    for r in $REMOTES; do $SSH $r "test -f acwork/runs/$RUN/sync/rep_r$k.done" || ok=0; done
    [ $ok = 1 ] && break; sleep 10
  done
  t0=$(date +%s); files="$HUB/rep_r$k.safetensors"; i=0
  for r in $REMOTES; do i=$((i+1)); rsync -a -e "$SSH" $r:acwork/runs/$RUN/sync/rep_r$k.safetensors $HUB/gather/rep$i.safetensors & files="$files $HUB/gather/rep$i.safetensors"; done
  wait
  ~/acwork/train-venv/bin/python ~/acwork/avg_weights.py $HUB/avg_r$k.safetensors $files || { echo "[hub] AVG FAILED round $k"; exit 1; }
  for r in $REMOTES; do rsync -a -e "$SSH" $HUB/avg_r$k.safetensors $r:acwork/runs/$RUN/sync/ & done
  wait
  for r in $REMOTES; do $SSH $r "touch acwork/runs/$RUN/sync/avg_r$k.done; rm -f acwork/runs/$RUN/sync/avg_r$((k-1)).safetensors"; done
  touch $HUB/avg_r$k.done; rm -f $HUB/avg_r$((k-1)).safetensors $HUB/gather/*.safetensors
  echo "[hub] round $k: averaged + pushed in $(( $(date +%s) - t0 ))s"
done
echo HUB_DONE
