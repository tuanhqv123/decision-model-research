# AndroidWorld Docker runbook (learned 2026-11-02, `Model_Training_Server`)

Everything here was hit and fixed in one evening. Do not re-derive.

## 1. The upstream Dockerfile needs three patches
Repo `google-research/android_world` @ `55ddaf5`.

| # | Problem | Symptom | Fix |
|---|---|---|---|
| 1 | `FROM openjdk:18-jdk-slim` | image removed from Docker Hub; build fails instantly | `FROM eclipse-temurin:17-jdk-jammy` (satisfies cmdline-tools' Java 17+) |
| 2 | `COPY . /` **and** `WORKDIR /` | `uv pip install .` packages **the whole filesystem** (Android SDK included): 21 GB RSS, 90+ min, never finishes | `COPY . /app` + `WORKDIR /app` before the install. Rebuild then takes ~1 min off the cached layers |
| 3 | `PIL` imported but Pillow undeclared | container starts, then `ModuleNotFoundError: No module named 'PIL'` in `user_data_generation.py` | `RUN uv pip install --system pillow` after the project install |

Docker layer caching is worth using: after patch #1 the SDK + Python layers are cached, so patches #2/#3
cost seconds, not minutes. **Do not delete the image/build cache to "start clean".**

## 2. Run command (KVM, capped, localhost-only)
```bash
docker run -d --name aw-emu \
  --device /dev/kvm --security-opt seccomp=unconfined \
  --cpus=6 --memory=24g --memory-swap=24g \
  -p 127.0.0.1:5051:5000 \
  android_world:latest
```
- `--device /dev/kvm` is enough (no `--privileged` needed) **on a host with `/dev/kvm`**.
- Host port: **5051**. On `Model_Training_Server`, **5000 is a prod service** — never bind it.
- `--memory` must be **≥24 GB**. At 16 GB the emulator process gets killed inside the container while
  the FastAPI server keeps answering `/health` — a very confusing half-dead state
  (`docker inspect` shows `OOMKilled=false` because the container's main process did not die).
  Tell: `adb devices` is empty and `/reset` returns 500 with `device 'emulator-5554' not found`.

## 3. Expect a long, noisy startup
- Boot ≈ 65 s (KVM), then `adb root` restarts adbd (a transient "device not found" is normal).
- `emulator_setup=True` installs ~17 APKs and clicks through onboarding. Chrome/Contacts onboarding
  often logs `Failed to automatically setup app ... "Accept & continue" not found` — **warning, not fatal**.
- The server only answers `/health` **after** the whole lifespan setup completes.

## 4. HTTP API limits that shape any agent
- `/screenshot` returns **pixels only** — `{"pixels": [...]}` as a nested Python list of
  **~7.8 M ints** for 1080×2400×3. Hundreds of MB per response; this is the memory problem in §2.
- There is **no a11y tree over HTTP**. `StateResponse.ui_elements` exists but is never populated.
  In-process `EnvironmentInteractingAgent` gets `state.ui_elements`; the HTTP client does not.
- No server-side step budget: the client owns `reset → initialize → act → score → tear_down`.
- Score = `is_successful(env)` from device state, so it is **not** gated on the agent declaring done.

## 5. d1-2B-specific gotchas
- The released AC-v2 checkpoint is **V-track** (screenshot + goal + history). Over HTTP it returns an
  **empty `app_name`** and empty `text` for `open_app` / `type_text`: those are "read" slots that need
  the a11y text the server does not expose. Harness workarounds (goal-keyword app map) are
  **scaffolding, not model capability** — say so in any write-up.
- `open_app` must receive a name from AndroidWorld's own vocabulary (`clock`, not `stopwatch`;
  see `android_world/env/adb_utils.py::_PATTERN_TO_ACTIVITY` and `setup_device/apps.py`).
- The model rarely emits `terminate`/`status` online, so episodes run to the step cap.

## 6. Operational
- **Detach long runs**: `(setsid nohup bash run.sh > logs/run.log 2>&1 &)`. Sessions tied to SSH die
  with `client_loop: send disconnect: Broken pipe` and take the experiment with them.
- **Reuse one SSH connection**: `ssh -o ControlMaster=auto -o ControlPath=/tmp/cc-aw -o ControlPersist=10m ...`
  Parallel SSH churn caused `Can't assign requested address`.
- Poll with **one** combined command (smoke log + upload + GPU + health), not five.

## 7. First real measurement (2 tasks, smoke — not a benchmark)
`ClockStopWatchRunning` **1.0**, `ContactsAddContact` **0.0**; model p50 ≈ **113–122 ms**;
12 steps both. Full context: `benchmarks/results/2026-11-02-online-aw-smoke.md`.
