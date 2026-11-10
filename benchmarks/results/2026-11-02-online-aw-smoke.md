# Online AndroidWorld — pilot results (2026-11-02)

**Headline: the loop runs end-to-end; the measured numbers are weak and the emulator is unstable.
This is an infrastructure + honesty result, not a benchmark result.** Two tasks cannot support a
success-rate claim, and every number below is shown with its trivial controls.

## Setup
- Image: `android_world:latest` built from repo HEAD `55ddaf5` (Pixel 6 / API 33, `google_apis`).
- Container `aw-emu`: `--device /dev/kvm`, 6 CPUs, 24 GB, `127.0.0.1:5051 -> 5000`.
- Policy: d1-2B `e2-final` (AC-v2 SFT, V-track, 720 px), served locally at `127.0.0.1:8700`,
  memory fraction 0.35.
- Harness: `benchmarks/aw/aw_smoke.py`. Observation = screenshot pixels only; history = the model's
  own previous actions. Score = AndroidWorld's `is_successful(env)` from device state.
- Suite: `n_task_combinations=1, seed=30`, one-time `--perform_emulator_setup`.

## Measured (2 tasks, 12-step cap, repeated)
| Run | Policy | ContactsAddContact | ClockStopWatchRunning | Model p50 |
|---|---|---|---|---|
| A | d1 | 0.0 | **1.0** | 122 ms |
| B | d1 | 0.0 (12 steps) | 0.0 (`done` after 3 steps) | 116 ms |
| C | d1 | 0.0 (12 steps) | 0.0 | 117 ms |
| D | `wait` | 0.0 | 0.0 | — |
| E | `random` | 0.0 | 0.0 | — |

**Reading:** the only success anywhere is one stopwatch episode in run A. It did not reproduce; in run B
the model instead declared `done` after three steps and scored zero. `wait` and `random` also score zero,
which is the correct control behaviour — it tells us the task is not trivially passable, but it does not
tell us the model is skilled. **Net finding: run-to-run variance dominates at this sample size.**

## 6-task subset (8-step cap) — the number to quote, with its controls
Tasks: `ClockStopWatchRunning, ClockTimerEntry, ContactsAddContact, MarkorCreateNote, SimpleSmsSend,
SystemBrightnessMax`. Coarse-grained tasks only; no typing-heavy or multi-app tasks.

| Policy | Success | Notes |
|---|---|---|
| d1-2B (`e2-final`) | **1/6** (16.7%) | only `ClockStopWatchRunning`; model p50 189–262 ms (GPU shared with training) |
| `wait` | **0/6** | 8 steps of `wait`, ~10 s per episode |
| `random` | **0/6** | random click/scroll/wait/back, ~140–170 s per episode |

With 6 tasks the Wilson 95% interval on 1/6 is roughly **[3%, 56%]** — the point estimate is meaningless at
this size, and it is reported only because the controls are exactly matched. The consistent element across
every run is that `ClockStopWatchRunning` is solvable and the rest are not.


## 12-task subset: aborted by emulator instability
`ClockStopWatchRunning, ClockTimerEntry, ContactsAddContact, ContactsNewContactDraft, FilesDeleteFile,
MarkorAddNoteHeader, MarkorCreateFolder, MarkorCreateNote, NotesIsTodo, SimpleSmsSend,
SystemBrightnessMax, AudioRecorderRecordAudio` (10-step cap) was launched at 18:03 UTC. The emulator
became unresponsive during the first task (`/screenshot` 500, then `adb: device 'emulator-5554' not
found`), logging repeated `Could not get a11y tree, retrying`, and every later task failed at `/reset`.
No success-rate is reported for this subset.

## Emulator stability (the real blocker)
The emulator process dies under sustained load while the FastAPI server keeps answering `/health`:
`docker inspect` reports `OOMKilled=false` because the container's main process survives, but
`adb devices` is empty and every environment call returns 500. Raising the container cap 16 → 24 GB did
not prevent it. Restarting the container also re-does the one-time app install (~5 min), so each crash
costs ~10 minutes of wall clock.

## Interface limits that shape the numbers
1. The HTTP server returns **pixels only** (no accessibility tree). App names and typed text are text
   slots, so the V-track checkpoint returns an **empty `app_name`**; resolving it from goal keywords is
   harness scaffolding and is reported as such.
2. `/screenshot` returns the frame as a nested integer list — millions of values per call, hundreds of
   MB per response. This is a real per-step memory cost.
3. There is no server-side step budget; the client owns the episode, and the model rarely emits `status`.
4. The model sometimes terminates early (`done` after 3 steps), which zeroes a task that was on track.

## What would make this a result
- 116 tasks × several instances, multiple seeds, Wilson/bootstrap CIs: one instance per task gives a
  ±9-point interval at p = 0.5, so a 4-point separation needs order 1.9k scored decisions.
- A restart-supervisor inside the container so the emulator self-heals without losing the app install.
- The in-process agent path (accessibility tree) instead of the pixels-only HTTP path.

## Reproduce
```bash
# on GPU_HOST_A
cd ~/d1-api
AW_URL=http://127.0.0.1:5051 D1_URL=http://127.0.0.1:8700 \
  venv/bin/python aw_smoke.py --tasks ClockStopWatchRunning,ContactsAddContact --max-steps 12
```
