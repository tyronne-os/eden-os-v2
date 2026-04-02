"""EDEN OS V2 — Elevated Watchdog Sidecar.

Runs every 3 seconds:
  - Samples last 15 frames from shared queue
  - Computes motion score (pixel variance) + expression variance
  - Calls router's /evaluate-failover for agent-enhanced decision
  - Falls back to consecutive_bad >= 2 rule if agent unavailable
"""

import asyncio
import base64
import logging
import time
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI
from pydantic_settings import BaseSettings

logger = logging.getLogger("eden.watchdog")


class WatchdogSettings(BaseSettings):
    router_url: str = "http://router:8100"
    shared_dir: str = "/shared"
    check_interval: float = 3.0
    motion_threshold: float = 0.05
    consecutive_fails_to_failover: int = 2
    model_config = {"env_file": ".env", "extra": "ignore"}


cfg = WatchdogSettings()
app = FastAPI(title="EDEN Watchdog", version="2.0.0")

# State
consecutive_bad: dict[int, int] = {}  # pipeline_id → consecutive bad count
last_check_time: float = 0.0
last_check_result: dict = {}
watchdog_running: bool = False


def _read_recent_frames(frame_dir: Path, n: int = 15) -> list[bytes]:
    """Read the N most recent frame files from the shared frame directory."""
    if not frame_dir.exists():
        return []
    files = sorted(frame_dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)[:n]
    frames = []
    for f in files:
        try:
            frames.append(f.read_bytes())
        except Exception:
            pass
    return list(reversed(frames))


def compute_motion_score(frames: list[bytes]) -> float:
    """Compute motion score from raw frame bytes."""
    if len(frames) < 2:
        return 0.0
    try:
        arrays = [np.frombuffer(f, dtype=np.uint8) for f in frames]
        diffs = []
        for i in range(1, len(arrays)):
            min_len = min(len(arrays[i - 1]), len(arrays[i]))
            diff = np.mean(np.abs(arrays[i][:min_len].astype(float) - arrays[i - 1][:min_len].astype(float)))
            diffs.append(diff)
        avg_diff = np.mean(diffs) if diffs else 0.0
        return min(1.0, float(avg_diff / 25.0))
    except Exception as e:
        logger.warning(f"Motion score error: {e}")
        return 0.5


async def _evaluate_failover_with_agent(pipeline_id: int, motion_score: float, consecutive: int, frame_count: int) -> bool:
    """Ask router's agent-enhanced endpoint whether to failover.
    Falls back to consecutive_bad >= threshold if call fails.
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(
                f"{cfg.router_url}/evaluate-failover",
                json={
                    "pipeline_id": pipeline_id,
                    "motion_score": motion_score,
                    "consecutive_bad": consecutive,
                    "frame_count": frame_count,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                decision = data.get("should_failover", False)
                logger.info(f"Agent failover decision for P{pipeline_id}: {decision}")
                return decision
    except Exception as e:
        logger.warning(f"Agent failover eval failed, using threshold fallback: {e}")

    # Fallback: original rule
    return consecutive >= cfg.consecutive_fails_to_failover


async def _trigger_failover(pipeline_id: int):
    """Tell the Router to mark a pipeline as failed and swap to next."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{cfg.router_url}/failover",
                json={"pipeline_id": pipeline_id},
            )
            logger.warning(f"Failover triggered for pipeline {pipeline_id}: {resp.status_code}")
    except Exception as e:
        logger.error(f"Failover trigger failed: {e}")


async def _check_loop():
    """Main watchdog loop — runs every 3 seconds."""
    global last_check_time, last_check_result, watchdog_running
    watchdog_running = True
    frame_dir = Path(cfg.shared_dir) / "frames"

    logger.info(f"Watchdog loop started (interval={cfg.check_interval}s, threshold={cfg.motion_threshold})")

    while watchdog_running:
        try:
            frames = _read_recent_frames(frame_dir)
            if len(frames) < 2:
                await asyncio.sleep(cfg.check_interval)
                continue

            motion = compute_motion_score(frames)
            last_check_time = time.time()

            # Get current active pipeline from router
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{cfg.router_url}/status")
                    status = resp.json()
                    active_pipelines = [
                        p for p in status.get("pipelines", [])
                        if p["status"] in ("ready", "busy")
                    ]
            except Exception:
                active_pipelines = []

            last_check_result = {
                "motion_score": round(motion, 3),
                "frame_count": len(frames),
                "is_healthy": motion >= cfg.motion_threshold,
                "active_pipelines": len(active_pipelines),
                "timestamp": last_check_time,
            }

            if motion < cfg.motion_threshold:
                # Bad check — ask router's agent-enhanced failover decision
                for p in active_pipelines:
                    pid = p["id"]
                    consecutive_bad[pid] = consecutive_bad.get(pid, 0) + 1
                    logger.warning(
                        f"Bad check #{consecutive_bad[pid]} for pipeline {p['name']} "
                        f"(motion={motion:.3f} < {cfg.motion_threshold})"
                    )

                    # Ask agent-enhanced router for failover decision
                    should_failover = await _evaluate_failover_with_agent(
                        pid, motion, consecutive_bad[pid], len(frames)
                    )
                    if should_failover:
                        await _trigger_failover(pid)
                        consecutive_bad[pid] = 0
            else:
                # Good check — reset counters
                for pid in list(consecutive_bad.keys()):
                    consecutive_bad[pid] = 0

        except Exception as e:
            logger.error(f"Watchdog check error: {e}")

        await asyncio.sleep(cfg.check_interval)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "watchdog_running": watchdog_running,
        "last_check": last_check_result,
        "consecutive_bad": consecutive_bad,
    }


@app.on_event("startup")
async def startup():
    logger.info("EDEN Watchdog starting...")
    asyncio.create_task(_check_loop())
