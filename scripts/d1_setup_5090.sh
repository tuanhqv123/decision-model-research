#!/bin/bash
# d1work bootstrap on the 5090 worker: raw AC download + venv (no GPU use).
set -x
mkdir -p "$HOME/d1work/ac_raw" "$HOME/d1work/acdata" "$HOME/d1work/runs" "$HOME/d1work/logs"
cd "$HOME/d1work/ac_raw" || exit 1
B=https://storage.googleapis.com/gresearch/android_control
curl -sfLO "$B/splits.json"
curl -sfLO "$B/test_subsplits.json"
for i in $(seq -w 0 19); do
  f=android_control-000$i-of-00020
  [ -s "$f.done" ] && continue
  curl -sfL --retry 5 -o "$f" "$B/$f" && touch "$f.done" && echo "got $f $(du -h "$f" | cut -f1)"
done
echo "RAW_DONE $(du -sh "$HOME/d1work/ac_raw" | cut -f1)"
if [ ! -x "$HOME/d1work/venv/bin/python" ]; then
  "$HOME/miniconda3/bin/python" -m venv --system-site-packages "$HOME/d1work/venv"
fi
"$HOME/d1work/venv/bin/pip" install -q --upgrade pip
"$HOME/d1work/venv/bin/pip" install -q "transformers>=4.57" accelerate pillow tfrecord protobuf "android-env" numpy
"$HOME/d1work/venv/bin/python" - <<'PY'
import torch, transformers
print("VENV_OK torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__)
PY
echo SETUP_DONE
