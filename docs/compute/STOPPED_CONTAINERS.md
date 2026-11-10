# Containers stopped for training — MUST be restored after training

| When | Host | Container | Why | Restore command |
|---|---|---|---|---|
| 2026-10-29 | GPU_HOST_B ([ip]) | `[service]` (compose project `[service]`, `/opt/[service]/docker-compose.yml`, uvicorn :8200, held 21.9GB VRAM, restart=unless-stopped) | free the RTX 4090 for d1-2B fine-tuning (user-authorized) | `ssh GPU_HOST_B 'docker start [service]'` (or `cd /opt/[service] && docker compose up -d`) |

Not touched: its DB `[service]` (still up), `[service]`, postgres containers.
Verify after restore: `ssh GPU_HOST_B 'docker ps --filter name=[service]; nvidia-smi --query-gpu=memory.used --format=csv'` (expect ~23GB used again).

**RESTORED 2026-10-29 05:13 (+07):** `docker start` → Up, restarts=0, "Application startup complete", HTTP 200 on
host port 2104 (`/`, `/docs`); DB `[service]` healthy throughout. Nothing else on any server was stopped.
