#!/bin/bash
# Cleanup for the 2026-11-07 server session. See docs/compute/CLEANUP-2026-11-07.md.
#
#   ./scripts/cleanup_2026-11-07.sh            # dry run: prints what would move / what would be deleted
#   ./scripts/cleanup_2026-11-07.sh --execute  # backs up, verifies, then deletes everything we created
#
# Rule: back up and verify BEFORE deleting, and never touch anything pre-existing.
set -uo pipefail

EXECUTE=0
[ "${1:-}" = "--execute" ] && EXECUTE=1

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO/backup-2026-11-07"
run() { if [ "$EXECUTE" = "1" ]; then eval "$@"; else echo "[dry] $*"; fi; }

H=$(while read -r h; do ssh -o BatchMode=yes -o ConnectTimeout=6 "$h" 'true' >/dev/null 2>&1 && { echo "$h"; break; }; done < /tmp/aliases.txt)
if [ -z "${H:-}" ]; then echo "no reachable host alias; abort"; exit 1; fi

HOP=$(ssh -o BatchMode=yes "$H" 'awk "/^Host[[:space:]]/{for(i=2;i<=NF;i++) if(\$i !~ /[*?]/) print \$i}" $HOME/.ssh/config 2>/dev/null | sort -u | while read -r x; do ssh -o BatchMode=yes -o ConnectTimeout=5 "$x" true >/dev/null 2>&1 && { echo "$x"; break; }; done')
if [ -z "${HOP:-}" ]; then echo "no reachable second hop; abort"; exit 1; fi

echo "== backup =="
run "mkdir -p '$DEST'/arm '$DEST'/logs '$DEST'/val"
# worker -> jump -> mac (the worker is not directly reachable from here)
run "ssh -o BatchMode=yes '$H' 'ssh -o BatchMode=yes $HOP \"cd ~/d1work && tar czf /tmp/arm_out.tgz runs/arm_status20/ckpt runs/*.jsonl runs/*.summary.json logs\"'"
run "scp -q '$H:/tmp/arm_out.tgz' '$DEST/arm_out.tgz'"

if [ "$EXECUTE" = "1" ]; then
  echo "== verify backup =="
  tar -tzf "$DEST/arm_out.tgz" >/dev/null || { echo "backup archive unreadable; abort"; exit 1; }
  SZ=$(tar -tzvf "$DEST/arm_out.tgz" | awk '/model\.safetensors$/ {print $5}')
  echo "model.safetensors bytes in archive: ${SZ:-MISSING}"
  [ -n "${SZ:-}" ] || { echo "checkpoint missing from backup; abort"; exit 1; }
fi

echo "== delete (only what this session created) =="
run "ssh -o BatchMode=yes '$H' 'rm -rf ~/d1work /tmp/ckpt_pkg /tmp/ckpt.tgz /tmp/d1code.tgz /tmp/d1_setup.sh /tmp/d1_setup2.sh /tmp/prep_chain.sh /tmp/run_val.sh /tmp/train_arm.sh /tmp/run_full.sh /tmp/run_arm_eval.sh /tmp/compact_cur15k_status20.jsonl /tmp/eval_eps.json /tmp/todo.txt /tmp/h2.txt'"
run "ssh -o BatchMode=yes '$H' 'ssh -o BatchMode=yes $HOP \"rm -rf ~/d1work; rm -rf ~/.cache/huggingface/hub/models--LiquidAI--d1-2B ~/.cache/huggingface/modules; rm -f /tmp/d1code.tgz /tmp/d1_setup.sh /tmp/d1_setup2.sh /tmp/prep_chain.sh /tmp/run_val.sh /tmp/train_arm.sh /tmp/compact_cur15k_status20.jsonl /tmp/eval_eps.json /tmp/todo.txt /tmp/h2.txt /tmp/ckpt.tgz /tmp/arm_out.tgz\"'"

echo "== local temp =="
run "rm -rf /tmp/d1code.tgz /tmp/compact_cur15k_status20.jsonl /tmp/eval_eps.json /tmp/aliases.txt /tmp/h2.txt /tmp/todo.txt /tmp/awrep"
run "rm -f /tmp/upckpt.sh /tmp/upckpt2.sh /tmp/watch_prep.sh /tmp/watch_arm.sh /tmp/smoke_train.sh /tmp/run_val.sh /tmp/train_arm.sh /tmp/run_full.sh /tmp/run_arm_eval.sh /tmp/d1_setup_5090.sh /tmp/prep_chain.sh /tmp/d1_setup2.sh /tmp/fetch_eval.sh"

echo "== definition of done (run these and expect empty) =="
echo "  ssh <hostA> 'ls ~/d1work'                 -> No such file"
echo "  ssh <hostA> 'ssh <hop> \"ls ~/d1work\"'      -> No such file"
echo "  ssh <hostA> 'ssh <hop> \"pgrep -af train_d1.py|eval_d1.py\"' -> empty"
echo "  docker ps -a | grep -c aw-                -> 0 (we created no container this session)"
echo "  ls $DEST                                  -> arm_out.tgz present and readable"
echo
echo "This script never touches a container, an image, or a process it did not create."
