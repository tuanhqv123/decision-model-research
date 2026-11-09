# Containers stopped for training — MUST be restored after training

| When | Host | Container | Why | Restore command |
|---|---|---|---|---|
| 2026-10-29 | [redacted] ([redacted]) | `syngenta-display-api-display-tracking-syngenta-1` (compose project `syngenta-display-api`, `/opt/syngenta-display-api/docker-compose.yml`, uvicorn :8200, held 21.9GB VRAM, restart=unless-stopped) | free the RTX 4090 for d1-2B fine-tuning (user-authorized) | `ssh [redacted] 'docker start syngenta-display-api-display-tracking-syngenta-1'` (or `cd /opt/syngenta-display-api && docker compose up -d`) |

Not touched: its DB `syngenta-display-api-db-1` (still up), `masan-gpu0`, postgres containers.
Verify after restore: `ssh [redacted] 'docker ps --filter name=[redacted]; nvidia-smi --query-gpu=memory.used --format=csv'` (expect ~23GB used again).

**RESTORED 2026-10-29 05:13 (+07):** `docker start` → Up, restarts=0, "Application startup complete", HTTP 200 on
host port 2104 (`/`, `/docs`); DB `syngenta-display-api-db-1` healthy throughout. Nothing else on any server was stopped.
