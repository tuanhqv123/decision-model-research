# VRAM audit, 2026-11-07 — who is using the GPUs, and it is not us

Asked to check whether we are silently holding VRAM. Checked, read-only. **Nothing was stopped or killed.**

## Host A — RTX 4090, 24,564 MiB total

| | value |
|---|---|
| used | **2,176 MiB** |
| free | **21,917 MiB** |
| our processes | **none** |
| the 2,176 MiB | `pid 2641764`, owner `[user]`, `python -m src.main` — **pre-existing, not ours** |

12 containers are up, all pre-existing ([service], ocr-tunnel, cf-tunnel, [service] + its db, loki, grafana,
prometheus, [service], display-tracking-db, deployment-postgres, survey-management-db). The [service] display
container is `Restarting (1)` — that is a **pre-existing fault documented in `STOPPED_CONTAINERS.md`**, not
something this session did.

## Host B — RTX 5090, 32,607 MiB total

| | value |
|---|---|
| used | **18,168 MiB** |
| free | **13,944 MiB** |
| our processes | **none** |
| the 18,168 MiB | 4 × `gunicorn app.main:app`, **owner root** — pre-existing prod |

Breakdown: 5,260 + 2,808 + 7,442 + 2,630 MiB.

## Our actual footprint

| host | disk | VRAM |
|---|---|---|
| A | `~/acv2` 15 GB | **0** |
| B | `~/acv2` 22 GB | **0** |

Training finished and exited; no `train_d1.py`, `eval_d1.py` or `acv2` process survives on either host.

## Why this matters for the numbers

This is the mechanism behind the three OOM failures today: **our training needed ~11.8 GB and host B only has
13.9 GB free**, which is enough only when the state is short. With the 36-candidate list the state was 2,712
tokens and the run died at step 20; capping candidates to 14 (while pinning the gold candidate in place) cut the
state to 757 chars typical and the run completed 938 steps at 11.8 GB.

**Nobody else was harmed and nothing was throttled.** The 18,168 MiB of prod usage on B is exactly the same total
observed before this session began, so we neither added to it nor took from it.

## Rule going forward

Before any GPU run: read `nvidia-smi --query-compute-apps=pid,used_memory` **and** resolve each pid to an owner.
If a pid is `root`/another user, it is prod and is untouchable. Only the *free* number is ours to spend, and the
cap (`--mem-frac`) must be set below it minus 2 GB headroom.

---

## Update 2026-11-07 evening — one of ours is running, and it is tight

While two experiments run in parallel:

| host | prod (not ours) | ours | free |
|---|---|---|---|
| A (4090, 24.5 GB) | 2,176 MiB (`[user]`, `python -m src.main`) | **12.0 GB** (`arm_best`) | 9.9 GB |
| B (5090, 32.6 GB) | **18,168 MiB** (4 × root `gunicorn`) | **12.0 GB** (`arm_ctrl`) | **1.9 GB** |

Host B is at **30.2 / 32.6 GB**. The control run is alive at step 360/938 but has under 2 GB of headroom, so a
transient allocation can still kill it mid-run — the same failure mode that cost three runs earlier today.

**Rule reinforced:** on host B, `--mem-frac` must be ≤ 0.38 (≈12.4 GB) *and* the run must be started with the
knowledge that prod's 18.1 GB could grow. A safer default is 0.35, or wait for host A (9.9 GB spare with a 12 GB
job already running, so also tight).

**Nothing was stopped.** The four gunicorn workers and the `[user]` process are pre-existing prod and were left
untouched.
