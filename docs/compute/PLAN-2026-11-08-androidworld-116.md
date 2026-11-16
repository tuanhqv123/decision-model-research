# PLAN — AndroidWorld 116-task standard run (not yet run, awaiting user confirmation)

*Written 2026-11-08 after the user asked for "standard mode, then run". This is a plan to confirm, NOT a results report.*

## Current state (checked by command)

| Item | State | Notes |
|---|---|---|
| Emulator `aw-emu` | **up 6h, health OK** | Gate A 32 passed / 0 failed |
| Suite | **116 tasks** | `suite/task_list` returns exactly 116 |
| Emulator executes `navigate_home`/`keyboard_enter`/`double_tap`/`swipe` | **✅ all succeed** | I sent each one as a test |
| Checkpoint `ckpt-E2-final` | present on host A | `~/acv2/ckpt/ckpt-E2-final` |
| Model server (port 8700) | **FAILED, not running** | `ModuleNotFoundError: train_d1` — I launched it with a missing path |
| Free VRAM | **21.9 GB** | prod uses only 2.1 GB (`[user]`) |
| Container network | **`bridge`** | ⚠️ see Risk #1 |

## What I ALREADY did (you need to know)

1. Copied `serving/` + `benchmarks/train/ac_common.py` to `~/d1-api` (84 KB) — harmless.
2. **Launched the model server → failed** (`ModuleNotFoundError: train_d1`). No process is running.
3. Copied `~/awh` (the test harness) over earlier — harmless.
4. Cleaned up an 11.4 GB tarball in `/tmp` (previous turn).

**No task has been run.** There is no new AndroidWorld number. 1/8 is still the only number we have.

## Risk #1 — the container uses the `bridge` network, not `host`

Earlier (2026-11-06) I noted `--network host` in the notes because container→host traffic was firewalled. The container is now `bridge`, so **the model server at 127.0.0.1:8700 on the host will NOT be reachable from inside the container**.

Three ways to handle it:
- **(a)** Run the harness **on the host** (the in-process env needs `android_world` + `adb` to see the emulator). But adb on the host **does not see** the emulator in the container (different network namespace) — tried today, `emulator-5554 offline`.
- **(b)** Run the harness **inside the container**, with the model client pointing at `host.docker.internal:8700` (needs `--add-host` or the gateway IP).
- **(c)** Restart the container with `--network host` (as on 2026-11-06), after which the model server at `127.0.0.1:8700` is reachable.

I lean towards **(c)** because it has already been shown to work, but restarting the container costs ~5 minutes of boot + lifespan.

## Standard run plan (awaiting your approval)

### Step 1 — Fix the model server (10 minutes)
- `PYTHONPATH` must include both `benchmarks/train` (which has `train_d1.py`) and `serving/`
- Relaunch, smoke 1 request, confirm `/health` = 200 and that a real decision is returned
- **Gate: if the smoke fails → stop, do not touch any task**

### Step 2 — Fix the container→model path (10 minutes)
- Choose (b) or (c) above; I will tell you which one after measuring
- **Gate: one model request from inside the container must return 200**

### Step 3 — Smoke 8 tasks (20 minutes)
- `run_bench.py --config configs/gate_e8_pure.yaml --limit 8`
- **Gate: episodes are recorded correctly, no crash, timeout/invalid are countable**
- If it fails → stop and report, do not run the full suite

### Step 4 — Standard run of 116 tasks, d1-2B condition (90 minutes, background)
- `run_bench.py --config configs/main_116.yaml`
- Pre-register first: `docs/research/PRE-REG-2026-11-08-androidworld-116.md`
- Write the prediction down first so the bar is not set afterwards: **SR_all expected 2–8%**
- Resumable, so if it breaks we continue

### Step 5 — `wait` control on 116 tasks (30 minutes)
- Same 116 tasks, policy `wait`
- **Harness check**: if `wait` also scores >0, the harness is scoring by itself → wrong, must stop

### Step 6 — Report
- Table: `SR_all` + Wilson, `SR_reachable` + coverage, invalid/crash/timeout, latency p50
- Fix paper §5.7: 8 tasks → 116 tasks
- State clearly that the 2 controller actions (`keyboard_enter`, `navigate_home`) are scaffolding if I add them

## What I NEED you to decide

1. **Confirm the run** — 116 tasks × 2 conditions ≈ **2 machine-hours**, plus retries.
2. **Add the 2 controller rules or not?** (`type_text` → `keyboard_enter`; stuck → `navigate_home`). If yes, they must be declared `scaffolding` in the paper. If no, SR will be lower but "clean".
3. **How to connect container→model**: (b) or (c)?

I will **not** do anything more until you confirm.
