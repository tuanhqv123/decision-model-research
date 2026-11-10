# d1 API on GPU_HOST_B (deployed 2026-10-29, user-authorised) - REMOVED 2026-10-30 at the user's request (stop_server.sh --purge); kept as a record and a how-to

What runs: `~/d1-api` on GPU_HOST_B (user `[user]`): venv (6.7 GB), checkpoint `models/e2-final` (3.6 GB, E2: AC-v2 64.07 pure protocol (255/398; written 64.32/256 before the 2026-11-07 audit),
720 px, sha256 verified against the Mac backup), `d1_api.py` (FastAPI, uvicorn, 1 worker). Bound to 127.0.0.1:8700 ONLY (not reachable from
outside); reach it through an SSH tunnel. Token in `~/d1-api/.env` (server) and `~/.d1-api-token` (Mac), never committed.
GPU: process capped at 35% of the card (D1_MEM_FRAC=0.35); idle it holds ~5.7 GB, the [service]/[service] services keep their own memory.
Nothing outside `~/d1-api` was touched; the [service] container and its DB were not modified.

Use:  ssh -N -f -L 8700:127.0.0.1:8700 GPU_HOST_B   then   POST http://127.0.0.1:8700/v1/decide   (see serving/d1_api.py)
Start: `ssh GPU_HOST_B 'bash ~/d1-api/run_server.sh models/e2-final 720 0.35'`
Stop:  `ssh GPU_HOST_B 'bash ~/d1-api/stop_server.sh'`      Remove everything: `... stop_server.sh --purge`
Swap model: copy a new checkpoint to `~/d1-api/models/<name>`, then `run_server.sh models/<name> <img_width> 0.35`
  (grounding-run checkpoint = 858 px; E2 = 720 px; the width must match the checkpoint).

Verified: parity 16/16 with the offline evaluation (same action and click point); single-request latency (GPU shared with prod)
model p50 131 ms / p95 144 ms, end to end from the Mac through the tunnel p50 305 ms / p95 808 ms.

<!-- Numbers corrected 2026-11-07 after the internal audit: 64.32/256 -> 64.07/255; 56.50 -> 56.05 pure. -->
