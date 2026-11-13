# Containers stopped for training — MUST be restored after training

| When | Host | Container | Why | Restore command |
|---|---|---|---|---|
| 2026-10-29 | GPU_HOST_B ([ip]) | `[service]` (compose project `[service]`, `/opt/[service]/docker-compose.yml`, uvicorn :8200, held 21.9GB VRAM, restart=unless-stopped) | free the RTX 4090 for d1-2B fine-tuning (user-authorized) | `ssh GPU_HOST_B 'docker start [service]'` (or `cd /opt/[service] && docker compose up -d`) |

Not touched: its DB `[service]` (still up), `[service]`, postgres containers.
Verify after restore: `ssh GPU_HOST_B 'docker ps --filter name=[service]; nvidia-smi --query-gpu=memory.used --format=csv'` (expect ~23GB used again).

**RESTORED 2026-10-29 05:13 (+07):** `docker start` → Up, restarts=0, "Application startup complete", HTTP 200 on
host port 2104 (`/`, `/docs`); DB `[service]` healthy throughout. Nothing else on any server was stopped.

| 2026-11-10 16:10 UTC | GPU_HOST_B | `[service]` (app container only, held 11.3 GB VRAM, restart=unless-stopped; idle at 0% GPU) | free the RTX 4090 for the Q1 generative-arm training in parallel with R1 (operator-authorized: "stop only, do not delete") | `ssh GPU_HOST_B 'sudo docker start [service]'` |

Not touched: its DB container and every other container on the host. **Must be restored when the Q1 training on this host ends.**

Activity check before/after the stop (2026-11-10): real production traffic, 120–215 `POST /stitch` requests/day, all
between 00:00 and 09:59 UTC (07:00–16:59 +07). Last request 09:50:11 UTC; stopped 16:08 UTC. **Auto-restore armed on
the host for 23:30 UTC** (`~/restore_client_container.sh`, log `~/restore_client_container.log`); restore earlier as
soon as the Q1 training on that host ends.
