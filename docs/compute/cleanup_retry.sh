#!/bin/bash
# Resume the failed backups (rsync --partial, up to 10 tries each), wipe a host only after its rsync exits 0.
BK=~/Workspace/research/backup-2026-10-29; LOG=$BK/deadline.log
exec >> $LOG 2>&1
echo "=== retry started $(date)"
for h in prd-survey-x1-5090 s4090-1x-100; do
  EXTRA=""; [ $h != prd-survey-x1-5090 ] && EXTRA="--exclude acdata/imgs --exclude acdata/records"
  r=1
  for i in $(seq 1 10); do
    rsync -a --partial --timeout=120 -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=8" \
      --exclude train-venv --exclude vllm-venv --exclude __pycache__ --exclude ac_raw --exclude ac_raw2 \
      --exclude "runs/E2-cur15k-V/best" --exclude "runs/E2c-cont-cur8kB/best" --exclude "runs/R2b-V-cur24k/best" \
      $EXTRA $h:acwork/ $BK/$h/acwork/
    r=$?; echo "$h try $i rsync exit=$r $(date +%T) size=$(du -sh $BK/$h | cut -f1)"
    [ $r = 0 ] && break; sleep 20
  done
  echo $r > $BK/$h.rsync_exit
  if [ $r = 0 ]; then
    for f in setup_env.sh setup_env.log dl_ac.sh dl_ac.log rs_s4090.log; do rsync -a $h:$f $BK/$h/home/ 2>/dev/null; done
    ssh -n $h 'rm -rf ~/acwork ~/setup_env.sh ~/setup_env.log ~/dl_ac.sh ~/dl_ac.log ~/rs_s4090.log \
      ~/.cache/huggingface/hub/models--LiquidAI--d1-2B ~/.cache/huggingface/modules/transformers_modules/LiquidAI \
      ~/.cache/huggingface/modules/transformers_modules/ckpt; echo "$(hostname): wiped; acwork exists? $(test -e ~/acwork && echo yes || echo no)"'
  else
    echo "$h: backup still FAILING after 10 tries -> NOT wiped"
  fi
done
ssh -n s4090-1x-100 'docker ps --filter name=syngenta --format "{{.Names}} {{.Status}}"'
echo "=== retry finished $(date)"
