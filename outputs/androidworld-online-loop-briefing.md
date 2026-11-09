# Running the AndroidWorld Benchmark as an Online Loop on a Linux x86_64 + KVM Host

**Purpose.** Concise, source-backed technical briefing for standing up `google-research/android_world` as an **online** mobile-agent evaluation loop (live emulator, agent acts step-by-step, reward from system state), on a Linux x86_64 host with KVM. Four sections: (1) Docker path / failure modes, (2) environment server + suite/task/checker framework, (3) published baselines & attribution, (4) operational pitfalls.

**Provenance flags used below.** `PRIMARY` = read directly from a source file in the repo / the paper abstract. `VERIFIED` = I looked it up / recomputed and can point to a URL. `UNVERIFIED` = claim I could *not* reproduce from a primary source. **All baseline papers are arXiv preprints; I did not confirm any of them as peer-reviewed** (see §3).

---

## 1. Docker path: deployment & failure modes

### 1.1 What the Docker image does

Repo root `Dockerfile` (base image at HEAD is `FROM openjdk:18-jdk-slim`; **you patched it to `eclipse-temurin:17-jdk-jammy`**, which is the right fix — see failure #1). Chain of events in the build:

1. Install system libs: `curl sudo wget unzip bzip2 libdrm-dev libxkbcommon-dev libgbm-dev libasound-dev libnss3 libxcursor1 libpulse-dev libxshmfence-dev xauth xvfb x11vnc fluxbox wmctrl libdbus-glib-1-2 ffmpeg`.
2. Android SDK args: `ARCH=x86_64`, `TARGET=google_apis`, `API_LEVEL=33`, `BUILD_TOOLS=33.0.0`, `EMULATOR_PACKAGE=system-images;android-33;google_apis;x86_64`, `ANDROID_CMD=commandlinetools-linux-11076708_latest.zip`. `ANDROID_SDK_ROOT=/opt/android`.
3. Download that `cmdline-tools` zip from `dl.google.com`, unzip, then perform the SDK-layout move into `cmdline-tools/tools/` (`mkdir cmdline-tools/tools && mv NOTICE.txt source.properties bin lib tools/`).
4. `sdkmanager --licenses` then `sdkmanager --verbose --no_https <packages>` for `emulator platform-tools platforms;android-33 build-tools;33.0.0 system-images;android-33;google_apis;x86_64`.
5. `avdmanager create avd --force --name Pixel_6_API_33 --device pixel_6 --package system-images;android-33;google_apis;x86_64`.
6. Build **Python 3.11.3 from source** (`--enable-optimizations`, `make -j $(nproc)`, `make altinstall`), symlink `python3`/`python` → `python3.11`, then `apt-get install python3-pip`.
7. Install `uv` (astral.sh), `COPY . /`, `chmod a+x` the three `docker_setup` scripts, `uv pip install . --system`.
8. Entrypoint `./docker_setup/entrypoint.sh` → `./docker_setup/start_emu_headless.sh` → `adb root` → `python3 -m server.android_server`.

`start_emu_headless.sh` (PRIMARY, read): detects HW accel from `HW_ACCEL_OVERRIDE` or by grepping `/proc/cpuinfo` for `vmx`/`svm` (on Linux) vs `sysctl` (macOS); emits the warning *"This Docker image is experimental and has only been tested on linux devices with KVM enabled"* when no accelerator is found; sets `-accel on`/`-accel off` accordingly. `start_emu.sh` is the headed variant (`-avd ${EMULATOR_NAME} -no-boot-anim -gpu off`); the headless variant adds `-no-window -no-snapshot`.

`env_launcher.load_and_setup_env` (PRIMARY): connects to the *already-running* emulator at `console_port=5554`, `grpc_port=8554`; `emulator_setup=True` installs all apps once; `freeze_datetime=True` pins the clock to October 2023.

### 1.2 Failure modes / breakage at HEAD (audited)

| # | Failure | Cause / detail | Fix |
|---|---|---|---|
| 1 | **`openjdk:18-jdk-slim` unresolvable** 🔴 | Base image was removed from Docker Hub; `docker build` fails immediately. | Already patched → `eclipse-temurin:17-jdk-jammy`. ✓ (Temurin 17 satisfies cmdline-tools' Java 17+ requirement.) |
| 2 | **Pin `commandlinetools-linux-11076708_latest.zip` 404s** | Google rotates cmdline-tools revisions; old URLs are eventually pruned. `dl.google.com` may 404. | Curl it first; update `ANDROID_CMD` to current `commandlinetools-linux-*_latest.zip`. |
| 3 | **`sdkmanager --no_https` rejected** | Legacy flag; modern cmdline-tools may error on unknown flags. | Drop `--no_https`. |
| 4 | **Emulator pacakge install fails on ARM build host** | `system-images;android-33;google_apis;x86_64` is x86_64-only. | `docker buildx build --platform linux/amd64 ...` (README documents this for Apple Silicon). |
| 5 | **`python3-pip` targets the wrong interpreter** 🔴 | `make altinstall` puts 3.11.3 in `/usr/local/bin`; `apt python3-pip` is for the distro Python (jammy's 3.10). `pip` shelling goes to the wrong interpreter. | Package install uses `uv`, so mostly avoided; don't depend on the apt `pip` for the source-built python. |
| 6 | **`uv pip install . --system` fails** | Needs `uv` on PATH (`/root/.local/bin`, which the Dockerfile adds) and network to resolve deps (`android_env`, `thefuzz`, `numpy`, `pandas`, `fastapi`, `uvicorn`, `requests`, `absl-py`). | Ensure `uv` on PATH + network; confirm the repo's `pyproject.toml` is buildable. |
| 7 | **Emulator won't boot / slow-to-timeout** 🔴 (most common real-world failure) | No `/dev/kvm` or nested virt off → script sets `-accel off` → pure-software emulation. Emulator can hang on boot or take many minutes. | Run `--privileged`, confirm `/dev/kvm` on host, ensure vmx/svm present. |
| 8 | **`adb root` denied** | `adb root` needs a *debuggable* image. `google_apis` (userdebug) is fine; `google_apis_playstore` or production images aren't. | Keep `TARGET=google_apis`. |
| 9 | **Port collision (5554 / 8554)** | Emulator uses console `5554` + gRPC `8554`; server connects to the same. If a host process owns them, clash. | Check `adb devices` (shows `emulator-5554`); free the ports. |
| 10 | **`COPY . /` bloats the layer** | Copies the whole repo root to `/`. | Keep repo clean / add `.dockerignore`; builds from a shallow clone. |
| 11 | **`sdkmanager` never exits on a new license prompt** | `yes Y |` may not cover newly-introduced SDK licenses. | Pre-accept licenses / re-pin; monitor the build for a hang. |
| 12 | **GCS APK download 4xx → build/runtime error** | Open-source APKs fetched from `https://storage.googleapis.com/gresearch/android_world/<apk>`. | Verify network; a failed APK raises `RuntimeError` in `setup_apps`. |

---

## 2. Environment server + suite/task/checker framework

### 2.1 The FastAPI server (`server/android_server.py`, PRIMARY)

Started via `python3 -m server.android_server`, uvicorn on `0.0.0.0:5000`. `lifespan` builds:
`env_launcher.load_and_setup_env(console_port=5554, emulator_setup=True, freeze_datetime=True, adb_path="/opt/android/platform-tools/adb")`, then `TaskRegistry()`, then `create_suite(aw_registry, n_task_combinations=2, seed=42)`.

**Endpoints** (exact, from source):

| Method | Path | Params | Returns |
|---|---|---|---|
| POST | `/reset` | `go_home` | `{status, message}` |
| GET | `/screenshot` | `wait_to_stabilize` | `{"pixels": [int,...]}` |
| POST | `/execute_action` | JSON action body | `{status, message}` |
| GET | `/suite/task_list` | `max_index` | `{"task_list": [...]}` |
| GET | `/suite/task_length` | `task_type` | `{"length": int}` |
| GET | `/suite/reinitialize` | `n_task_combinations`, `seed`, `task_family` | `{status, message}` (rebuilds suite in-memory) |
| POST | `/task/initialize` | `task_type`, `task_idx` | `{status, message}` |
| POST | `/task/tear_down` | `task_type`, `task_idx` | `{status, message}` |
| GET | `/task/score` | `task_type`, `task_idx` | `{"score": is_successful(env)}` |
| GET | `/task/goal` | `task_type`, `task_idx` | `{"goal": str}` |
| GET | `/task/template` | `task_type`, `task_idx` | `{"template": str}` |
| POST | `/close` | — | `{status}` |
| GET | `/health` | — | `{status}` |

**Two important limits of the HTTP path:**
- The `/screenshot` endpoint only returns `pixels`. The pydantic model `StateResponse` declares `ui_elements`, but the endpoint never populates it → **a remote HTTP client gets raw pixels, not the accessibility/UI-element tree.** (In-process agents read `agent.env` for the full state.)
- There is **no server-side episode runner / step-budget / `is_successful`-gating** in the HTTP server. `scripts/run_suite_on_docker.py` (PRIMARY) shows the manual client loop: `reset` → `get_screenshot` → `execute_action` … → `initialize_task` → `get_task_score` → `tear_down_task` → `reset(go_home=True)`. The loop contract is the *client's* responsibility.

### 2.2 Suite / task / checker mechanics (`suite_utils.py`, `episode_runner.py`, `checkpointer.py`, PRIMARY)

- **Suite** = ordered `dict[task_name -> list[TaskEval]]`; one `TaskEval` per `n_task_combinations`. `create_suite(registry, n_task_combinations, seed, tasks, use_identical_params, env)`. Server default: `n_task_combinations=2, seed=42`. `run.py` default: `n_task_combinations=1, task_random_seed=30` (different protocol!).
- **Per-instance seed**: `int(sha256(f"{seed}_{name}_{i}").hexdigest(),16) % 2**32`.
- **Task lifecycle** (`_run_task`): `task.initialize_task(env)` → `run_episode(...)` → `task.is_successful(env)` → `task.tear_down(env)`. Any exception → a "failed" result with traceback, **and the run keeps going** (skips the task, doesn't abort the suite).
- **Episode** (`episode_runner.run_episode`): `agent.reset(start_on_home_screen)`, `agent.set_max_steps(max_n_steps)`, loop `agent.step(goal)` until `termination_fn` triggers, `result.done`, or `max_n_steps` reached.
- **Step budget**: `_allocate_step_budget(task_complexity) = int(10 * task_complexity)`. README: max_steps ≈ **2× human-average completion time** (per-task spreadsheet, updated 11/18/2024).
- **Success is gated on `done`** 🔴: `agent_successful = task_successful if interaction_results.done else 0.0`. So (a) an agent that hits the step budget without declaring `done`, or (b) one that declares `done` prematurely, **scores 0 even if the device state is actually correct.** This is the same class of scoring quirk we hit in the AC-v2 port.
- **Reward = durable system state** (not LLM/UI judge): `task.is_successful(env)` inspects Android state via `adb` — SQLite DBs, file system, settings; UI-element validation is only a fallback. This is the paper's "durable reward" claim.
- **Checkpointer / resume** (`IncrementalCheckpointer`): saves each `(task_name, instance_idx)` as `<TaskName>_<idx>.pkl.gz` (gzip+pickle). `_run_task_suite` skips instances already completed (and not failed). `run.py` creates `run_<YYYYMMDDTHHMMSS>` under `~/android_world/runs` unless `--checkpoint_dir`. Note: `suite_utils.run`'s docstring says "resume from the last fully completed *task template*", but the code actually resumes at **per-instance** granularity.
- **Metric aggregation** (`process_episodes`): per-task `mean_success_rate`, `mean_episode_length`, `total_runtime_s`, `num_fail_trials`, then merges `android_world/task_metadata.json` (`difficulty`, `optimal_steps`, `tags`). The headline AndroidWorld number = mean of per-task success rates under the chosen protocol.
- **Env launcher** (`env_launcher.py`): `emulator_setup=True` → `setup.setup_apps(env)` (install the ~17 external APKs from GCS + click through each app-specific onboarding); `freeze_datetime=True` → sets device clock to **October 2023**.
- **App set** (`setup_device/setup.py` + `apps.py`): `_APPS` = 24 classes. First-party apps (Camera, Chrome, Clock, Contacts, Dialer, Files, Settings) have no APK (`apk_names=""`); the rest are downloaded from the `gresearch/android_world` GCS bucket. `maybe_install_app` tries each `apk_name` in order (arch fallback, e.g. VLC has separate x86_64 vs arm APK).
- **Suite families** (`registry`): `ANDROID_WORLD_FAMILY`, `MINIWOB_FAMILY_SUBSET`, `MINIWOB_FAMILY`, `ANDROID_FAMILY`, `INFORMATION_RETRIEVAL_FAMILY`. Default = `android_world`.
- **In-box agents** (`run.py _get_agent`): `human_agent`, `random_agent`, `m3a_gemini_gcp`, `t3a_gemini_gcp`, `t3a_gpt4`, `m3a_gpt4v` (default), `seeact`. Custom agent = subclass `EnvironmentInteractingAgent` + implement `step()`.

**Online-vs-offline reframe (from our master map):** AndroidWorld is an *online, multi-turn* environment — the policy is an autoregressive VLM sampling one action per timestep over a long trajectory, evaluated on **per-task success**. This is fundamentally different from the offline, single-pass, per-action classification framing we used for AC-v2. An honest comparison to these baselines must use trajectory-level success rate (§3), and any single-pass model must be justified on a calibration/latency axis rather than raw success.

---

## 3. Published baselines & attribution

All sources are **arXiv preprints** (none verified as peer-reviewed here). AndroidWorld protocol: **116 tasks across 20 apps**, dynamic parameterization, system-state reward, per-task success rate.

| Model (baseline) | AndroidWorld | AndroidLab | Type / provenance | Source |
|---|---|---|---|---|
| Human | **80.0%** | — | human | Rawles et al., *AndroidWorld* 2024 [2405.14573] |
| **MobileRL-9B** (GLM-4.1V-9B) | **80.2%** | 53.6% | open VLM + online agentic RL (**self-reported SOTA**) | Xu et al., *MobileRL* [2509.18119v2] |
| MobileRL-7B (Qwen2.5-VL-7B) | **72.0%** | 42.5% | open VLM + RL | [2509.18119v2] |
| **GUI-Owl-1.5-8B-Thinking** | **71.6%** | — | native GUI agent (Qwen3-VL) | Ye et al., *Mobile-Agent-v3.5* [2602.16855] |
| GUI-Owl-7B | **66.4%** | — | native GUI agent | Ye et al., *Mobile-Agent-v3* [2508.15144] |
| UI-TARS-1.5 | 64.2% | 38.3% | closed agent (as tabulated by MobileRL) | [2509.18119v2] Table 1 |
| V-Droid | 59.5% | 38.3% | open VLM | [2509.18119v2] Table 1 |
| Claude-Sonnet-4 | 41.0% | 40.6% | proprietary | [2509.18119v2] Table 1 |
| GPT-4o | 34.5% | 31.2% | proprietary | [2509.18119v2] Table 1 |
| UI-Tars-7B | 33.0% | 32.6% | open VLM | [2509.18119v2] Table 1 |
| M3A GPT-4 Turbo (**a11y tree**) | **30.6%** | — | multimodal agent (a11y) | [2405.14573] |
| M3A GPT-4 Turbo (**SoM**) | **25.4%** | — | multimodal (Set-of-Mark) | [2405.14573] |
| SeeAct (web-adapted) | **15.5%** | — | web agent | [2405.14573] |

### Attribution & caveats (important)
- **AndroidWorld, 30.6% / 80.0% human** — Google DeepMind's own paper baseline; the M3A numbers are from the paper. `MobileMiniWoB++` in the same paper: M3A 59.7% (text) / 67.7% (SoM), human 100%.
- **MobileRL 80.2%** — Tsinghua + Z.AI; self-reported best, on the AndroidWorld **test set** (RL trained on a 2,000-task AndroidWorld train split). It is *not* a drop-in for the canonical Google leaderboard protocol. Also gives AndroidControl-Low/High SR (Table 2), e.g. MobileRL-9B @ 94.3% High, if you want a bridge to AC.
- **GUI-Owl-7B 66.4%** — Alibaba Tongyi Lab; self-reported.
- **GUI-Owl-1.5 71.6%** — Alibaba; this figure is the **8B-Thinking** variant, on par with UI-TARS-2 (73.3%). The family (2B/4B/8B/32B/235B) is Qwen3-VL based.
- ⚠️ **`GUI-Owl-1.5-2B = 67.9% AndroidWorld` is UNVERIFIED.** The primary sources I read report the 8B-Thinking figure (71.6%) and OSWorld-Verified 43.5% for the 2B, but I found **no per-size AndroidWorld figure for the 2B** in [2602.16855] or the HF card I checked. Do not cite 67.9% without its exact source.
- **Protocol incomparability.** Published AndroidWorld numbers are only comparable if the protocol matches: same test set (116 tasks), same complexity-scaled step budget, number of instances, pass@1 vs averaged over k rolls, whether `--perform_emulator_setup` (app install) was done, and model-specific scaffolding (SoM, reasoning traces). MobileRL reports pass@1 at temperature 1.0, evaluated over 8 trials and broken down by complexity (Figure 4). The canonical "official" leaderboard is Google's spreadsheet, but those entries are self-reported and use varied protocols.

---

## 4. Operational pitfalls (checked against source)

1. **`openjdk:18` is a dead base image** — already patched to `eclipse-temurin:17-jdk-jammy`. ✓
2. **KVM / acceleration is the #1 runtime killer** — the headless script sets `-accel off` when `vmx`/`svm` are absent; container must be `--privileged` and the **host must expose `/dev/kvm`**. Without it the emulator runs in software and either never boots or takes many minutes. This is the most likely reason a container "runs but the emulator won't come up" on a non-KVM host.
3. **`adb root` requires a `google_apis` (userdebug) image** — a `google_apis_playstore` / production image will reject it.
4. **Fixed ports 5554 (console) + 8554 (gRPC)** — emulator is launched with `-grpc 8554`; server connects at both. Check `adb devices`; free the ports if occupied.
5. **`freeze_datetime` pins the clock to Oct 2023** — don't be surprised when app onboarding states assume that date.
6. **First-run app setup is slow & network-sensitive** — `emulator_setup=True` downloads ~17 external APKs from GCS and clicks through each onboarding. Any GCS 4xx raises `RuntimeError`. A fresh container is the clean reset.
7. **HTTP client only gets `pixels`, not `ui_elements`** — there is no accessibility-tree endpoint in the server. A remote HTTP agent needs its own vision; only in-process agents get the full `get_state()` (pixels + a11y tree + UI elements).
8. **No server-side step budget / episode gating** — the client must call `initialize_task` → act → `get_task_score` → `tear_down_task` → `reset`. Forgetting any of these leaks state and corrupts scores.
9. **Known fragile tasks** (documented inline in `run_suite_on_docker.py`): `RetroPlayingQueue` throws `sqlite3.OperationalError: no such table: playing_queue`; `SimpleSmsReplyMostRecent` throws `IndexError: list index out of range`. Catch and skip them in the client loop.
10. **Success requires BOTH agent-`done` AND `is_successful`=1** — a model that stalls (hits the step budget) or exits without declaring done scores 0. This mirrors the AC-v2 "grading artifact" finding.
11. **Resume granularity & config coupling** — `IncrementalCheckpointer` writes `<TaskName>_<idx>.pkl.gz`. If you change `seed`/`n_task_combinations` into the same `--checkpoint_dir`, instance names won't match → confusing partial re-runs. Use a fresh run dir per config.
12. **Seed/combination protocol mismatch** — server default is `n_task_combinations=2, seed=42`, whereas the paper/`run.py` default is `1, seed=30`. To reproduce a published number you must match the protocol, not the defaults.
13. **Model cost** — `minimal_task_runner.py` and default `m3a_gpt4v` use `gpt-4-turbo-2024-04-09`; swap `model_name` for repeated runs.
14. **Apple Silicon** — only works via `docker buildx build --platform linux/amd64`; emulator-inside-emulator is slow (README).
15. **File-descriptor limit** — `load_and_setup_env` raises `RLIMIT_NOFILE` to 32768 to avoid "Too many open files"; keep the host `ulimit` permissive.

---

## Sources

- Repo: `https://github.com/google-research/android_world` (Apache-2.0, "Copyright 2026 The android_world Authors")
  - `Dockerfile`, `docker_setup/start_emu_headless.sh`, `docker_setup/start_emu.sh`, `docker_setup/entrypoint.sh`
  - `server/android_server.py`, `scripts/run_suite_on_docker.py`, `minimal_task_runner.py`, `run.py`
  - `android_world/suite_utils.py`, `android_world/episode_runner.py`, `android_world/checkpointer.py`, `android_world/env/env_launcher.py`, `android_world/env/setup_device/{setup.py,apps.py}`, `android_world/agents/README.md`
- AndroidWorld paper (Rawles et al., 2024): `https://arxiv.org/abs/2405.14573`
- MobileRL (Xu et al., 2025): `https://arxiv.org/abs/2509.18119` (v2 2025-10-24); repo `https://github.com/THUDM/MobileRL`
- Mobile-Agent-v3 / GUI-Owl (Ye et al., 2025): `https://arxiv.org/abs/2508.15144`
- Mobile-Agent-v3.5 / GUI-Owl-1.5 (Ye et al., 2026): `https://arxiv.org/abs/2602.16855`; HF release `https://huggingface.co/mPLUG/GUI-Owl-1.5-2B-Instruct`
- Canonical task list / leaderboard (linked from repo README): `https://google-research.github.io/android_world/task_list.html` and the Google Sheets leaderboard.
