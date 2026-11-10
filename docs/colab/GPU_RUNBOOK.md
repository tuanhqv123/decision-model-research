# GPU / Compute Orchestration Runbook

> Hard-won lessons for running the AC-v2 SOTA baseline (and later training) on free GPU. **Read before touching compute.** Kept in sync with `AGENTS.md` (Program log). Every problem here was hit and fixed.

## 0. Golden rule — gate on GPU BEFORE any heavy job
Never launch a heavy model-load job until you've **proven a GPU is allocated**. The 8B `UI-Venus-1.5-8B` (16GB) on CPU hangs >55 min with empty logs. Add this **fail-fast guard** at the very top of the run cell:
```python
import torch
assert torch.cuda.is_available(), "NO GPU — aborting before slow CPU load"
# or: subprocess.run(["nvidia-smi",...]) and check it exists
```
If `torch` prints `2.11.0+cpu` or `cuda False`, the job is on CPU — abort, don't wait.

## 1. Platform availability (checked 2026-10-15) — both blocked
| Platform | Status | Evidence |
|---|---|---|
| **Google Colab T4** | `Service Unavailable` on assign | `colab usage` balance 0; `colab new --gpu T4` → `ColabRequestError ... Service Unavailable`. Free T4 quota exhausted (used for prior train + evals). |
| **Kaggle (CLI `kernels push`)** | GPU **never** granted | `torch 2.11.0+cpu cuda False`; `nvidia-smi` missing (Errno 2). Proven on **two accounts** (`tuannnnn2222222` AND `tuanatautonomous`) — BOTH show clean 30h GPU quota (`kaggle quota`) but run **CPU** on push. |

**Conclusion (corrected): it is NOT account-level — it is the CLI push path.** `kaggle kernels push --accelerator NvidiaTeslaT4` (even with the correct/default accelerator ID) never allocates a GPU on these accounts. GPU is only obtainable via the **Kaggle web UI** (browser run), or Colab after T4 reset. No CLI flag fixes this.
#### Definitive (tested 2026-10-15, 3 configs x 2 accounts)
Even the community-confirmed fix **`machine_shape: "NvidiaTeslaT4"` in metadata (no `--accelerator`)** still allocates **CPU** (`torch 2.11.0+cpu cuda False`, no nvidia-smi). Combined with `--accelerator NvidiaTeslaT4` and `enable_gpu` variants, ALL fail. **Conclusion: Kaggle CLI `kernels push` cannot allocate GPU on these accounts — the only Kaggle GPU path is the web UI (browser notebook).**


## 2. Correct Kaggle CLI usage (do NOT rediscover)
- **Accelerator ID is `NvidiaTeslaT4`** (= GPU T4 ×2 dual GPU, also the **default**). `GPU_T4X2` is NOT a valid value. Valid IDs (per `docs/kernels.md`): `NvidiaTeslaT4`, `NvidiaTeslaA100`, `NvidiaL4`, `NvidiaL4X1`, `NvidiaH100`, `NvidiaRtxPro6000`, `TpuV5E8`, `TpuV6E8`. Some are competition/admin-only.
- `kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4` pushes AND runs. `--no-run` saves without running.
- `kernel-metadata.json` fields: `enable_gpu`/`enable_internet`/`is_private` are **STRINGS** (`"true"`), not booleans. `"kernel_type": "notebook"` (lowercase). `machine_shape` = GPU type (can leave `""` to default).
- `kaggle kernels init -p <dir>` → correct metadata template.
- **`kaggle datasets create` skips directories by default** → use `-r zip` to include an `images/` dir. A bare `.tar` silently mis-uploads.
- Strip macOS junk (`find <dir> -name '._*' -delete`) before upload.
- **Dataset mounts nested:** `/kaggle/input/datasets/<owner>/<slug>/` (NOT `/kaggle/input/<slug>/`). Discover via `glob("/kaggle/input/**/<file>", recursive=True)`.
- Kernel files (`.py` companions) don't reliably land in cwd → **embed `run_acv2.py` + `calib_metrics.py` as strings** and write to `/kaggle/working`.

## 3. Orchestration rules (teach the system)
1. **One heavy job in background with `notifyOnCompletion` + auto-trigger.** Do NOT poll `bg_status`/`bg_logs`/`kaggle kernels status` in a loop from the chat — the completion notification wakes you.
2. **Validate cheap first.** For an 8B model eval, first run `--limit 5` to prove mount+GPU+model-load work, THEN run the full 398. Never jump to a 4-hour full run on an unproven pipeline.
3. **Fail-fast guards** on: GPU presence, dataset mount (glob discovery), package install.
4. **Record every learning in `AGENTS.md` (Program log) + this runbook**, and `kaggle kernels init`/`--help` before guessing.
5. **Dependencies are ordered** (Step 0 → 1 → 3). Each has a pass-condition. Don't parallelize dependent computes.

## 4. Unblock + resume path
- **Kaggle:** web UI → Create Notebook → Settings → Accelerator → **GPU T4 ×2** → run a cell `!nvidia-smi`. If it lists a GPU, the account HAS GPU (web) — then run this job as that notebook (or drive it from CLI once GPU shows up). 
- **Colab:** wait for T4 quota reset (daily), then the harness (`run_acv2.py`, High-v2, resume) is ready.
- Then: Step 0 (baseline UI-Venus-1.5-8B) → Step 1 (calibration) → Step 3 (train). `run_acv2.py` already outputs ECE/Brier/coverage/latency for Step 1 (no code change).


## 5. ROOT CAUSE (CONFIRMED 2026-10-15 — phone-verified account gets GPU)
> **WIN:** account `trnanh140999` after phone verification -> `torch 2.11.0+cu128 cuda True`, **Tesla T4** (`nvidia-smi` OK). So phone-verify is THE gate; unblock by verifying.
 — account needs PHONE VERIFICATION (NOT code, NOT quota)
GPU options greyed out / CPU-only even on **web UI** because the account is **not phone-verified**. Kaggle requires phone verification to unlock GPU + internet access (**"Get phone verified"** in Notebook Settings / profile icon). Free GPU = **NVIDIA Tesla P100** (16GB, fine for the 8B baseline). Fix (user-only): login -> profile icon -> Verify phone number -> OTP -> then Settings -> Accelerator -> GPU is enabled (not greyed).
**Evidence chain (2026-10-15):** 3 configs (`--accelerator NvidiaTeslaT4`, `machine_shape:NvidiaTeslaT4`, `enable_gpu`) x 2 accounts (both clean 30h quota) x 2 credential formats (access_token KGAT + kaggle.json) -> **ALL CPU** (`torch 2.11.0+cpu cuda False`, no nvidia-smi), even on web UI. => Not code/quota; **phone verification is the gate**.

## 6. Script kernel for training (2026-10-15, learnings)
- `kernel_type:"script"` + `code_file` = the ONLY file guaranteed to land in `/kaggle/working`.
- **Do NOT rely on pushed companion `.py` or `.jsonl` files** landing in cwd (same unreliability as notebook kernels). Two robust options:
  1. **Self-contained script** (inline all helpers, no cross-file imports) — simplest.
  2. Mount data as a **dataset** and glob `/kaggle/input/datasets/<owner>/<slug>/`.
- **`enable_internet:"true"` is REQUIRED** to download a model from HuggingFace (`AutoModelForCausalLM.from_pretrained('LiquidAI/...')`). `false` → download fails.
- Use `machine_shape:"NvidiaTeslaT4"` + `enable_gpu:"true"` (matching the working smoke/val kernels).
- Base64-embedding scripts into a notebook repeatedly gave `SyntaxError: invalid syntax` on Kaggle → avoided entirely with self-contained script kernel.

## 7. Notebook vs script kernel for mounting datasets (2026-10-15)
- **NoteBook kernel PROVABLY mounts `dataset_sources`** (the val kernel did). Script kernels (`kernel_type:"script"`) are UNVERIFIED for dataset mount → use a notebook kernel for jobs that need mounted data.
- **Base64-embed ONLY small scripts, NEVER the data.** Embedding a ~100KB jsonl as base64 in a notebook repeatedly gave `SyntaxError: invalid syntax`. Embed small `.py` files only (val kernel did this fine); load the data from the dataset mount.
- **find_data must glob `/kaggle/input/**`** — the exact dataset mount path is not guaranteed (`/kaggle/input/datasets/<owner>/<slug>/` vs alternatives). Glob + print the resolved path.

## 8. SyntaxError root cause in notebook kernels (2026-10-15)
- The recurring `SyntaxError: invalid syntax` was **NOT the base64** — it was **cell.source as a LIST of lines lacking a trailing `\n`**. nbconvert joins the list with `''` (no separator), so `makedirs(x)open(y)` collapse into one token.
- **Fix:** build each cell's `source` as a SINGLE STRING with explicit `\n` between statements. Verify with `ast.parse(source)` in the builder before push.

## 2026-10-29 — own-server fleet (GPU_HOST_C, GPU_HOST_A, GPU_HOST_B) — hard-won rules
- **Launching remote jobs:** `ssh -f host "cd ~/acwork && setsid nohup <cmd> > log 2>&1 < /dev/null &"`. Plain
  `ssh host 'nohup … &'` can hang the local ssh and a delayed second launch then ran a DUPLICATE replica on the
  5090 (prod GPU). Always `pgrep -af "^train-venv/bin/python"` before launching; kill with the `^` anchor
  (`pkill -f "^train-venv/bin/python train_d1.py"`) — an unanchored `pkill -f` kills its own ssh shell.
- **Prod GPU (5090) cap:** `--mem-frac 0.55` (17.25GB of 31.4; prod [service] processes use ~10.6GB). The cap made our
  duplicate OOM instead of the prod services — keep it on every run there.
- **Disk:** GPU_HOST_A is ~96–98% full and hosts prod postgres → keep only its replica's transient sync
  files there; the local-SGD hub and checkpoints live on GPU_HOST_B. `vllm-venv` removed everywhere (unused).
- **Server-to-server copies:** `ssh-add -t <sec> <keys>` on the Mac + `ssh -A`, never copy private keys to servers.
  5090 ↔ 49.213.89.x moves ~3.8GB in ~1–2 min; both 4090s share [ip]/24.
- **RAM:** never load the full a11y records (8GB JSON) into Python on a prod host — use `compact_*.jsonl` (lazy strings).
- **d1-2B training:** grad-ckpt cuts peak 17.1→8.2GB/sample; the backward MUST run inside `state_attention` (gate:
  `test_gradckpt.py`, rel grad diff 6e-4). 4090 ≈ 2.2 samples/s (micro 2, ~2.5k tokens/sample).
- **Checkpoint:** `save_pretrained` rewrites config.json with a wrong auto_map → keep the release's config.json.

## 2026-11-10 — remote process control (learned twice in one day)
- **Never `pgrep -f` / `pkill -f` a pattern over `ssh host '...'`**: the remote `bash -c` carries the pattern in its
  own command line, so it matches itself and the kill takes out your shell (and anything after it in that command).
  List with `ps -eo pid,args | grep "[s]tage1_chain2"` (the bracket stops the grep matching itself), check the PID's
  command line, then `kill <pid>`.
- **`ssh host 'A && B & ...'` hangs the ssh client**: the backgrounded `A && B` subshell keeps the ssh channel open even
  when `B`'s own output is redirected. Start detached jobs as one simple command:
  `ssh host 'setsid nohup bash script.sh > log 2>&1 < /dev/null &'`.
  Third time (20:39): a preamble `cd .. && mv .. && kill .. && setsid nohup bash chain.sh .. &` — the `&` binds to
  the whole `&&` list, so the preamble and the launch went to the background together and the ssh client hung for
  the job's lifetime (setsid still protected the job). **Preamble in one ssh call, launch alone in the next.**
  Worse, the leftover `bash -c "cd ~/aw && cp run_arm.sh …"` stayed alive as chain7's parent, and its command line
  contains `run_arm.sh`, so chain5's `ps -eo args | grep "[r]un_arm.sh"` ("is a run going?") matched it: chain5 waited
  on it, it waited on chain7, chain7 waited on chain5 — a deadlock, 8 min lost before the verified kill. A
  "process running?" test by command-line substring matches any shell that merely mentions the name.
- **docker image IDs are not comparable across docker versions**: docker 29 with the containerd snapshotter reports
  the manifest digest, docker 28 with overlay2 the config digest. To prove two hosts run the same image, compare
  `sha256(json .RootFS.Layers)` and `sha256(json .Config)`.
