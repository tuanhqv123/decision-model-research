#!/usr/bin/env bash
# After rebalancing (2026-11-12): a half-chain keeps TARGET=58 but now owns fewer tasks; stop it (and its watchdog)
# once its run dir has scored every task in its rebalanced config. Logs REBALANCED_DONE to the chain log.
LOG=~/aw/stage1_chain.log
declare -A N=([aw116_v4f_s0]=48 [aw116_v4f_s0_b]=43 [aw116_m03f_s0]=43 [aw116_m03f_s0_b]=41)
while [ ${#N[@]} -gt 0 ]; do
  for id in "${!N[@]}"; do
    s=$(python3 ~/aw/scored_count.py ~/awruns/$id 2>/dev/null || echo 0)
    if [ "$s" -ge "${N[$id]}" ]; then
      for p in $(pgrep -f "^bash stage2_chain_v[0-9]+p?.sh"); do   tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "ID=$id" && kill $p; done
      for p in $(pgrep -f "[w]atchdog_runbench.sh $HOME/awruns/$id "); do kill $p; done
      for p in $(pgrep -f "[r]un_arm.sh"); do tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "RUNID=$id" && kill $p; done
      echo "$(date -u +%FT%TZ) REBALANCED_DONE $id: $s/${N[$id]} scored" >> $LOG; unset "N[$id]"
    fi
  done; sleep 60
done
