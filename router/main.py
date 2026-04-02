"""EDEN OS V2 — Intelligent Pipeline Router.

Dual-Track Architecture:
  Track 1 (Main Pipeline): Smallest/fastest pipeline first (P4 → P0 → P2)
  Track 2 (Backup Router): Escalates through remaining (P3 → P1)

5 Auto-Routing Features (preserved):
  Feature 1: Pre-Warm Greeting Sequence
  Feature 2: Dedicated Eve-Greeting Sub-Pipeline (force strongest)
  Feature 3: Auto-Retry with Immediate Failover (up to 3 retries)
  Feature 4: Heartbeat Pre-Check (quality gate before client delivery)
  Feature 5: Client-Side Animation Fallback flag + server push

Intelligence Layer:
  - 4 Claude-powered agents for speed, quality, failover, and warmup decisions
  - Graceful fallback to static SIZE_ORDER if agents unavailable

RunPod Cost Management:
  - 5-minute idle sleep timer (auto-stop pod after inactivity)
"""

import asyncio
import base64
import logging
import os
import time
from enum import Enum

import httpx
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from .metrics import MetricsStore, SIZE_ORDER, PIPELINE_SIZES_GB, PIPELINE_NAMES
from .agents import AgentManager

logger = logging.getLogger("eden.router")


class RouterSettings(BaseSettings):
    p0_url: str = "http://pipeline0:8010"
    p1_url: str = "http://pipeline1:8011"
    p2_url: str = "http://pipeline2:8012"
    p3_url: str = "http://pipeline3:8013"
    p4_url: str = "http://pipeline4:8014"
    models_dir: str = "/models"
    xai_api_key: str = ""
    runpod_idle_timeout_s: float = 300.0  # 5 minutes
    runpod_api_key: str = ""
    runpod_pod_id: str = ""
    model_config = {"env_file": ".env", "extra": "ignore"}


cfg = RouterSettings()

app = FastAPI(title="EDEN Router", version="2.1.0")


# ── Pipeline Status ──────────────────────────────────────────────────────────
class PipelineStatus(str, Enum):
    COLD = "cold"
    WARMING = "warming"
    READY = "ready"
    BUSY = "busy"
    FAILED = "failed"


class PipelineInfo:
    def __init__(self, pid: int, name: str, url: str, size_gb: float):
        self.pid = pid
        self.name = name
        self.url = url
        self.size_gb = size_gb
        self.status = PipelineStatus.COLD
        self.last_motion_score: float = 0.0
        self.fail_count: int = 0

    def to_dict(self):
        return {
            "id": self.pid,
            "name": self.name,
            "status": self.status.value,
            "size_gb": self.size_gb,
            "last_motion_score": self.last_motion_score,
            "fail_count": self.fail_count,
        }


# ── Pipeline Registry (ordered by SIZE: smallest → largest) ──────────────────
_pipeline_urls = {
    0: cfg.p0_url,
    1: cfg.p1_url,
    2: cfg.p2_url,
    3: cfg.p3_url,
    4: cfg.p4_url,
}

pipelines_by_id: dict[int, PipelineInfo] = {
    pid: PipelineInfo(
        pid=pid,
        name=PIPELINE_NAMES[pid],
        url=_pipeline_urls[pid],
        size_gb=PIPELINE_SIZES_GB[pid],
    )
    for pid in range(5)
}

# Canonical order: smallest → largest (for iteration and failover)
pipelines: list[PipelineInfo] = [pipelines_by_id[pid] for pid in SIZE_ORDER]


# ── Intelligence Layer ───────────────────────────────────────────────────────
metrics_store = MetricsStore()
agent_manager = AgentManager(metrics_store)


# ── RunPod Idle Sleep Timer ──────────────────────────────────────────────────
_last_activity: float = time.time()
_sleep_task: asyncio.Task | None = None


def _touch_activity():
    """Reset the idle timer on any request."""
    global _last_activity
    _last_activity = time.time()


async def _idle_sleep_loop():
    """Monitor for inactivity and stop RunPod pod after 5 minutes idle."""
    while True:
        await asyncio.sleep(30)  # check every 30s
        idle_seconds = time.time() - _last_activity
        if idle_seconds >= cfg.runpod_idle_timeout_s:
            logger.warning(
                f"RunPod idle for {idle_seconds:.0f}s (limit={cfg.runpod_idle_timeout_s:.0f}s). "
                "Requesting pod stop..."
            )
            await _stop_runpod_pod()
            break  # stop the loop after requesting shutdown


async def _stop_runpod_pod():
    """Stop the RunPod pod via API to save costs."""
    if not cfg.runpod_api_key or not cfg.runpod_pod_id:
        logger.info("RunPod auto-sleep: no API key or pod ID configured, skipping")
        return
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"https://api.runpod.io/v2/{cfg.runpod_pod_id}/stop",
                headers={"Authorization": f"Bearer {cfg.runpod_api_key}"},
            )
            logger.info(f"RunPod pod stop requested: {resp.status_code}")
    except Exception as e:
        logger.error(f"Failed to stop RunPod pod: {e}")


# ── Request Models ───────────────────────────────────────────────────────────
class AnimateRequest(BaseModel):
    audio_b64: str
    reference_image: str = "eve-NATURAL.png"
    force_strong: bool = False
    request_id: str = ""


class EvaluateFailoverRequest(BaseModel):
    pipeline_id: int
    motion_score: float
    consecutive_bad: int = 0
    frame_count: int = 0


# ── Motion Scoring ───────────────────────────────────────────────────────────
def compute_motion_score(frames_b64: list[str]) -> float:
    """Compute motion score from base64-encoded frames. Returns 0.0–1.0."""
    if len(frames_b64) < 2:
        return 0.0
    try:
        decoded = []
        for f in frames_b64[:15]:
            raw = base64.b64decode(f)
            arr = np.frombuffer(raw, dtype=np.uint8)
            decoded.append(arr)

        diffs = []
        for i in range(1, len(decoded)):
            min_len = min(len(decoded[i - 1]), len(decoded[i]))
            diff = np.mean(np.abs(decoded[i][:min_len].astype(float) - decoded[i - 1][:min_len].astype(float)))
            diffs.append(diff)

        avg_diff = np.mean(diffs) if diffs else 0.0
        return float(min(1.0, avg_diff / 25.0))
    except Exception as e:
        logger.warning(f"Motion score computation failed: {e}")
        return 0.5


# ── Dual-Track Architecture ─────────────────────────────────────────────────
def _reset_stale_failures():
    """Reset pipelines that were marked failed during startup or transiently.
    This allows recovery when a pipeline comes online after the router starts."""
    for p in pipelines:
        if p.status == PipelineStatus.FAILED and p.fail_count < 10:
            p.status = PipelineStatus.COLD
            p.fail_count = 0
            logger.info(f"Reset stale failure for {p.name}")


class MainPipeline:
    """Track 1: Always tries the smallest/fastest available pipeline."""

    @staticmethod
    def get_primary(agent_order: list[int] | None = None) -> PipelineInfo | None:
        order = agent_order or SIZE_ORDER
        for pid in order:
            p = pipelines_by_id.get(pid)
            if p and p.status != PipelineStatus.FAILED:
                return p
        # If all failed, reset and try again (recovery from startup failures)
        _reset_stale_failures()
        for pid in order:
            p = pipelines_by_id.get(pid)
            if p:
                return p
        return None


class BackupRouter:
    """Track 2: Escalates through remaining pipelines on failure."""

    @staticmethod
    def get_escalation_order(exclude_pid: int, agent_order: list[int] | None = None) -> list[PipelineInfo]:
        order = agent_order or SIZE_ORDER
        result = []
        for pid in order:
            if pid == exclude_pid:
                continue
            p = pipelines_by_id.get(pid)
            if p and p.status != PipelineStatus.FAILED:
                result.append(p)
        # If empty, reset failures and try again
        if not result:
            _reset_stale_failures()
            for pid in order:
                if pid == exclude_pid:
                    continue
                p = pipelines_by_id.get(pid)
                if p:
                    result.append(p)
        return result


# ── Pipeline Communication ───────────────────────────────────────────────────
async def _call_pipeline(pipeline: PipelineInfo, request: AnimateRequest) -> dict | None:
    """Call a single pipeline and return result or None on failure."""
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{pipeline.url}/animate",
                json={
                    "audio_b64": request.audio_b64,
                    "reference_image": request.reference_image,
                    "request_id": request.request_id,
                },
            )
            elapsed_ms = (time.time() - t0) * 1000
            if resp.status_code == 200:
                result = resp.json()
                pipeline.status = PipelineStatus.READY
                pipeline.fail_count = 0
                # Record success metrics
                motion = compute_motion_score(result.get("frames", []))
                metrics_store.record_call(pipeline.pid, elapsed_ms, motion, success=True)
                return result
            else:
                pipeline.fail_count += 1
                metrics_store.record_call(pipeline.pid, elapsed_ms, 0.0, success=False)
                logger.warning(f"Pipeline {pipeline.name} returned {resp.status_code}")
                return None
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        pipeline.fail_count += 1
        pipeline.status = PipelineStatus.FAILED
        metrics_store.record_call(pipeline.pid, elapsed_ms, 0.0, success=False)
        logger.error(f"Pipeline {pipeline.name} call failed: {e}")
        return None


# ── Feature 1: Pre-Warm ─────────────────────────────────────────────────────
prewarm_scores: dict[int, float] = {}


async def _prewarm_pipeline(pipeline: PipelineInfo):
    """Run a silent test animation to warm up the pipeline."""
    t0 = time.time()
    try:
        pipeline.status = PipelineStatus.WARMING
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{pipeline.url}/warmup",
                json={"reference_image": "eve-NATURAL.png"},
            )
            load_ms = (time.time() - t0) * 1000
            if resp.status_code == 200:
                data = resp.json()
                score = data.get("motion_score", 0.0)
                prewarm_scores[pipeline.pid] = score
                pipeline.status = PipelineStatus.READY
                pipeline.last_motion_score = score
                metrics_store.record_load_time(pipeline.pid, load_ms)
                logger.info(f"Pre-warm {pipeline.name} ({pipeline.size_gb}GB): score={score:.2f}, load={load_ms:.0f}ms")
            else:
                pipeline.status = PipelineStatus.COLD
    except Exception as e:
        logger.warning(f"Pre-warm {pipeline.name} failed: {e}")
        pipeline.status = PipelineStatus.COLD


# ── Main Animation Endpoint ─────────────────────────────────────────────────
@app.post("/animate")
async def animate(request: AnimateRequest):
    """Route animation through dual-track architecture with agent intelligence."""
    _touch_activity()
    t0 = time.time()

    # Get agent-recommended pipeline order
    agent_order = await agent_manager.get_routing_order(force_strong=request.force_strong)
    logger.info(f"Routing order: {[PIPELINE_NAMES[pid] for pid in agent_order]} (force_strong={request.force_strong})")

    # Track 1: Try primary pipeline (smallest/fastest)
    primary = MainPipeline.get_primary(agent_order)
    if not primary:
        return {"frames": [], "pipeline_used": "none", "error": "No pipelines available"}

    logger.info(f"Track 1 — Primary: {primary.name} ({primary.size_gb}GB)")
    result = await _call_pipeline(primary, request)

    if result:
        frames = result.get("frames", [])
        if frames:
            motion_score = compute_motion_score(frames)
            primary.last_motion_score = motion_score

            # Feature 4: Heartbeat Pre-Check
            should_fail = await agent_manager.should_failover(motion_score, primary.pid)
            if not should_fail:
                elapsed = time.time() - t0
                logger.info(f"Track 1 SUCCESS: {primary.name}, motion={motion_score:.3f}")

                # Fire-and-forget: agent recommends next warmup
                asyncio.create_task(_agent_warmup(primary.pid))

                return {
                    "frames": frames,
                    "pipeline_used": primary.name,
                    "pipeline_id": primary.pid,
                    "motion_score": round(motion_score, 3),
                    "attempt": 1,
                    "elapsed_s": round(elapsed, 2),
                    "track": "main",
                    "force_strong_pipeline": request.force_strong,
                }
            else:
                logger.warning(f"Track 1 QUALITY FAIL: {primary.name} motion={motion_score:.3f}, escalating to Track 2")
                primary.status = PipelineStatus.FAILED

    # Track 2: Backup Router — escalate through remaining pipelines
    backups = BackupRouter.get_escalation_order(primary.pid, agent_order)
    logger.info(f"Track 2 — Backups: {[p.name for p in backups]}")

    for attempt, backup in enumerate(backups[:3]):  # Feature 3: up to 3 retries
        logger.info(f"Track 2 attempt {attempt + 1}: trying {backup.name} ({backup.size_gb}GB)")

        result = await _call_pipeline(backup, request)
        if result is None:
            continue

        frames = result.get("frames", [])
        if not frames:
            continue

        motion_score = compute_motion_score(frames)
        backup.last_motion_score = motion_score

        should_fail = await agent_manager.should_failover(motion_score, backup.pid)
        if should_fail:
            logger.warning(f"Track 2: {backup.name} motion={motion_score:.3f}, trying next...")
            backup.status = PipelineStatus.FAILED
            continue

        elapsed = time.time() - t0
        logger.info(f"Track 2 SUCCESS: {backup.name}, motion={motion_score:.3f}")
        return {
            "frames": frames,
            "pipeline_used": backup.name,
            "pipeline_id": backup.pid,
            "motion_score": round(motion_score, 3),
            "attempt": attempt + 2,
            "elapsed_s": round(elapsed, 2),
            "track": "backup",
            "force_strong_pipeline": request.force_strong,
        }

    # All failed
    elapsed = time.time() - t0
    return {
        "frames": [],
        "pipeline_used": "none",
        "error": "All pipelines failed or produced static frames",
        "elapsed_s": round(elapsed, 2),
        "force_strong_pipeline": True,  # Feature 5: CSS fallback
    }


async def _agent_warmup(current_pid: int):
    """Fire-and-forget: ask agent which pipeline to pre-warm next."""
    try:
        warmup_pid = await agent_manager.recommend_warmup(current_pid)
        if warmup_pid is not None and warmup_pid in pipelines_by_id:
            p = pipelines_by_id[warmup_pid]
            if p.status == PipelineStatus.COLD:
                logger.info(f"Agent warmup: pre-warming {p.name}")
                await _prewarm_pipeline(p)
    except Exception as e:
        logger.warning(f"Agent warmup failed: {e}")


# ── Failover Endpoints ──────────────────────────────────────────────────────
@app.post("/failover")
async def failover(pipeline_id: int):
    """Watchdog-triggered failover: mark pipeline as failed."""
    _touch_activity()
    p = pipelines_by_id.get(pipeline_id)
    if p:
        p.status = PipelineStatus.FAILED
        p.fail_count += 1
        logger.warning(f"Failover: {p.name} marked FAILED (count={p.fail_count})")
    return {"status": "ok", "pipelines": [p.to_dict() for p in pipelines]}


@app.post("/evaluate-failover")
async def evaluate_failover(request: EvaluateFailoverRequest):
    """Agent-enhanced failover decision (called by watchdog)."""
    _touch_activity()
    should = await agent_manager.should_failover(
        motion_score=request.motion_score,
        pipeline_id=request.pipeline_id,
        consecutive_bad=request.consecutive_bad,
    )
    return {"should_failover": should}


# ── Status & Metrics Endpoints ───────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.1.0",
        "pipelines": [p.to_dict() for p in pipelines],
        "pipeline_order": [PIPELINE_NAMES[pid] for pid in SIZE_ORDER],
        "prewarm_scores": prewarm_scores,
        "agents": agent_manager.status(),
        "idle_timeout_s": cfg.runpod_idle_timeout_s,
    }


@app.get("/status")
async def status():
    return {
        "pipelines": [p.to_dict() for p in pipelines],
        "pipeline_order": SIZE_ORDER,
        "agents": agent_manager.status(),
    }


@app.get("/metrics")
async def metrics():
    """Pipeline performance metrics."""
    return metrics_store.get_summary()


# ── Startup ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global _sleep_task
    logger.info("=" * 60)
    logger.info("EDEN Router v2.1.0 starting...")
    logger.info(f"  Pipeline order (size): {[f'{PIPELINE_NAMES[pid]} ({PIPELINE_SIZES_GB[pid]}GB)' for pid in SIZE_ORDER]}")
    logger.info(f"  Agents enabled: {agent_manager.enabled}")
    logger.info(f"  RunPod idle timeout: {cfg.runpod_idle_timeout_s}s")
    logger.info("=" * 60)

    # Feature 1: Pre-Warm in SIZE_ORDER (smallest first = fastest to warm)
    logger.info("Pre-warming pipelines (smallest first)...")
    for pid in SIZE_ORDER:
        p = pipelines_by_id[pid]
        await _prewarm_pipeline(p)
    logger.info(f"Pre-warm complete. Scores: {prewarm_scores}")

    # Start RunPod idle sleep timer
    _sleep_task = asyncio.create_task(_idle_sleep_loop())
    logger.info("RunPod idle sleep timer started (5 min)")
