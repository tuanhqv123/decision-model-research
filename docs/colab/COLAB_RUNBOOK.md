# Colab Runbook — Mobile Decision Model

The whole flow is **one notebook** + the **`colab` CLI**. No local python files to maintain.
Read this top-to-bottom; it's the only doc you need to run and monitor it.

## Files
- `notebooks/train_mobile_decision.ipynb` — the single artifact you edit/run.
- This `docs/colab/COLAB_RUNBOOK.md` — the commands.

## 0. One-time authorization (browser)
The `colab` CLI talks to your Google Colab account. Run this once; it prints a link —
open it, sign in, click **Allow**, then paste the authorization code back into the CLI.

```bash
colab --auth=oauth2 sessions
```

## Commands (run from the repo root)

### 1. Provision a free T4 GPU VM
```bash
colab --auth=oauth2 new -s mobile-dec --gpu T4
```

### 2. Run the notebook (executes each cell, writes back an output notebook)
```bash
# IMPORTANT: --timeout 7200 so the long training cell isn't cut off
colab --auth=oauth2 exec -s mobile-dec -f notebooks/train_mobile_decision.ipynb --timeout 7200
```

**What it writes:** `notebooks/train_mobile_decision_output.ipynb` (with all outputs).

### 3. Monitor (pick one)
- Live web UI attached to *this* session: `colab --auth=oauth2 url -s mobile-dec` (add `--open`)
- Status / GPU: `colab --auth=oauth2 status -s mobile-dec`
- Stream recent events: `colab --auth=oauth2 log -s mobile-dec -n 30`
- Export full session history as markdown/notebook:
  `colab --auth=oauth2 log -s mobile-dec -o execution_log.md`

### 4. Pull the trained model back to your Mac
```bash
mkdir -p models/decision
colab --auth=oauth2 download -s mobile-dec /content/outputs/qwen_lora models/decision/qwen_lora
colab --auth=oauth2 download -s mobile-dec /content/outputs/qwen_merged_16bit models/decision/qwen_merged_16bit
```

### 5. Stop the VM (always do this — idle GPUs burn quota)
```bash
colab --auth=oauth2 stop -s mobile-dec
```

## Shortcuts
- **One-shot job** (provision → run → auto-teardown): `colab --auth=oauth2 run --gpu T4 notebooks/train_mobile_decision.py`
- **Just install packages on the VM**: `colab --auth=oauth2 install -s mobile-dec unsloth transformers`

## Typical timeline
| step | time |
|---|---|
| provision T4 | ~30 s |
| notebook install cell | ~2–3 min |
| data prep + before-acc | ~30 s |
| **training (60 steps, 2B)** | **~40–90 min** |
| calibrate + test + save | ~2 min |

## Notes
- Free Colab GPU quota varies by account; if `colab new --gpu T4` returns a `400`,
  fall back to `--gpu L4`, or omit `--gpu` for CPU (training becomes slow but still runs).
- The output notebook (`..._output.ipynb`) is the record of the run — commit it with git.
