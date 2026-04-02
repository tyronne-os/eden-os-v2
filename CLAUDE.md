# EDEN OS V2

Private, self-healing, triple-redundant bidirectional diffusion avatar system.

## Architecture

- **Gateway**: FastAPI (port 8000) — handles xAI Grok-4 chat + TTS, WebSocket frame streaming
- **Router**: Pipeline orchestrator with 5 auto-routing features (port 8100)
- **Watchdog**: Sidecar that monitors frame quality every 3s, triggers failover (port 8200)
- **5 Pipelines**: MuseTalk (P0), InfiniteTalk (P1), Ditto (P2), StableAvatar (P3), LiveAvatar 14B (P4)
- **Frontend**: React + Vite with WebSocket streaming and CSS fallback animation

## Key commands

```bash
# Local development
docker compose up                    # Start all services
docker compose up gateway router pipeline0 watchdog  # Minimal stack

# Build
docker build -t eden-os-v2 .

# Push to Docker Hub
docker tag eden-os-v2 edenberyl/eden-os-v2:latest
docker push edenberyl/eden-os-v2:latest

# Frontend dev
cd frontend && npm run dev
```

## Models

Models are stored on Seagate 5TB at `D:\eden-models\` and mounted into containers via Docker volumes. Models are NOT baked into the Docker image — they are pulled at runtime from HF Hub.

## Pipeline priority (smallest → largest for fastest failover)

```
P4 LiveAvatar    (1.26 GB) → fastest to load, primary
P0 MuseTalk      (6.37 GB) → fast backup
P2 Ditto         (6.45 GB) → mid backup
P3 StableAvatar (18.49 GB) → heavy backup
P1 InfiniteTalk (85.02 GB) → nuclear, largest, most capable
```

Dual-track architecture:
- **Track 1 (Main)**: Tries smallest pipeline first
- **Track 2 (Backup)**: Escalates through remaining on failure

Failover is agent-enhanced (Grok-4-fast via xAI API) with static-threshold fallback.
Watchdog detects static frames every 3s and triggers intelligent failover.

## RunPod idle sleep

Pod auto-stops after 5 minutes of inactivity to save costs.
Configure via `RUNPOD_IDLE_TIMEOUT_S` env var (default: 300s).

## Environment

All secrets go in `.env` (see `.env.template`). Never commit `.env`.
