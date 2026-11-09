#!/bin/bash
# Deadline job (user, 2026-10-29): if the work is not finished by 05:30, back up everything to this Mac,
# wipe our code/data from the 3 servers, and restart what we stopped.
# Skip switch: touch ~/Workspace/research/.acv2_done  (finished and cleaned up by hand before 05:30).
# A server is wiped ONLY if its backup rsync exited 0.
set -u
ROOT=~/Workspace/research
BK=$ROOT/backup-2026-10-29
LOG=$BK/deadline.log
mkdir -p $BK
exec >> $LOG 2>&1
echo "=== manual cleanup started $(date) (all runs finished 05:12)"

HOSTS="prd-survey-x1-5090 Model_Training_Server s4090-1x-100"
for h in $HOSTS; do  # 1) stop our jobs (anchored: never matches this ssh's own shell)
  ssh -n -o ConnectTimeout=20 $h 'pkill -f "^train-venv/bin/python"; pkill -f "^bash sync_avg.sh"; sleep 2; echo "$(hostname): our procs left=$(pgrep -f "^train-venv/bin/python" | wc -l)"'
done

for h in $HOSTS; do  # 2) back up (venvs and the public GCS mirror are reproducible: dl_ac.sh / setup_env.sh)
  mkdir -p $BK/$h
  EXTRA=""
  [ $h != prd-survey-x1-5090 ] && EXTRA="--exclude acdata/imgs --exclude acdata/records"  # copies of the 5090's
  rsync -a --exclude train-venv --exclude vllm-venv --exclude __pycache__ --exclude ac_raw --exclude ac_raw2 --exclude "runs/E2-cur15k-V/best" --exclude "runs/E2c-cont-cur8kB/best" --exclude "runs/R2b-V-cur24k/best" --exclude "runs/E2-final-ckpt" $EXTRA \
    $h:acwork/ $BK/$h/acwork/
  r1=$?
  mkdir -p $BK/$h/home
  for f in setup_env.sh setup_env.log dl_ac.sh dl_ac.log rs_s4090.log; do rsync -a $h:$f $BK/$h/home/ 2>/dev/null; done
  echo $r1 > $BK/$h.rsync_exit
  echo "$h backup rsync exit=$r1 size=$(du -sh $BK/$h | cut -f1)"
done

for h in $HOSTS; do  # 3) wipe ours, only where the backup succeeded
  if [ "$(cat $BK/$h.rsync_exit 2>/dev/null)" = 0 ]; then
    ssh -n $h 'rm -rf ~/acwork ~/setup_env.sh ~/setup_env.log ~/dl_ac.sh ~/dl_ac.log ~/rs_s4090.log \
      ~/.cache/huggingface/hub/models--LiquidAI--d1-2B ~/.cache/huggingface/modules/transformers_modules/LiquidAI \
      ~/.cache/huggingface/modules/transformers_modules/ckpt; echo "$(hostname): wiped; acwork exists? $(test -e ~/acwork && echo yes || echo no)"'
  else
    echo "$h: backup FAILED -> NOT wiped"
  fi
done

# 4) restart what we stopped (docs/compute/STOPPED_CONTAINERS.md)
ssh -n s4090-1x-100 'docker start syngenta-display-api-display-tracking-syngenta-1 >/dev/null; docker ps --filter name=syngenta --format "{{.Names}} {{.Status}}"; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
echo "=== deadline job finished $(date)"
