"""EDEN OS V2 — FastAPI Gateway.

Handles:
  - /welcome   → Eve's first greeting (Grok-4 text + xAI TTS + animation)
  - /chat      → Conversational loop (text + TTS + animation)
  - /ws        → WebSocket for real-time frame streaming
  - /health    → System health check
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from . import xai_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("eden.gateway")

app = FastAPI(title="EDEN OS V2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ────────────────────────────────────────────────────────────────────
conversation_history: list[dict] = []
active_connections: list[WebSocket] = []


# ── Helpers ──────────────────────────────────────────────────────────────────
async def _route_animation(audio_bytes: bytes, reference_image: str = "", force_strong: bool = False) -> dict:
    """Send audio + reference to the Pipeline Router and get animated frames back."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "audio_b64": base64.b64encode(audio_bytes).decode(),
            "reference_image": reference_image or settings.eve_reference_image,
            "force_strong": force_strong,
            "request_id": str(uuid.uuid4()),
        }
        resp = await client.post(f"{settings.router_url}/animate", json=payload)
        resp.raise_for_status()
        return resp.json()


async def _broadcast_frames(frames: list[str]):
    """Push base64 frames to all connected WebSocket clients."""
    dead = []
    for ws in active_connections:
        try:
            for frame_b64 in frames:
                await ws.send_json({"type": "frame", "data": frame_b64})
                await asyncio.sleep(1 / 30)  # ~30 FPS pacing
        except Exception:
            dead.append(ws)
    for ws in dead:
        active_connections.remove(ws)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check — validates gateway + downstream services."""
    checks = {"gateway": "ok", "xai": "unknown", "router": "unknown"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{settings.router_url}/health")
            checks["router"] = "ok" if r.status_code == 200 else "error"
    except Exception as e:
        checks["router"] = f"error: {e}"

    # Quick xAI check
    if settings.xai_api_key:
        checks["xai"] = "configured"
    else:
        checks["xai"] = "missing_key"

    status = "healthy" if checks["router"] == "ok" else "degraded"
    return {"status": status, "checks": checks, "version": "2.0.0"}


@app.post("/welcome")
async def welcome():
    """Eve's first greeting — guaranteed animated from first frame.

    Uses Feature 2 (Dedicated Eve-Greeting Sub-Pipeline) to force the
    strongest available pipeline for the initial greeting.
    """
    t0 = time.time()

    # Generate greeting text via Grok-4
    try:
        greeting_text = await xai_client.generate_greeting()
    except Exception as e:
        logger.warning(f"Grok greeting failed, using fallback: {e}")
        greeting_text = settings.greeting_text

    # Generate TTS audio via xAI Eve voice
    try:
        audio_bytes = await xai_client.text_to_speech(greeting_text)
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"error": "TTS unavailable", "text": greeting_text},
        )

    # Route to animation pipeline (force_strong=True for greeting)
    result = {}
    frames = []
    try:
        result = await _route_animation(
            audio_bytes,
            reference_image=settings.eve_reference_image,
            force_strong=True,
        )
        frames = result.get("frames", [])
    except Exception as e:
        logger.error(f"Animation routing failed (expected without GPU): {e}")

    # Broadcast frames via WebSocket
    if frames:
        asyncio.create_task(_broadcast_frames(frames))

    pipeline_used = result.get("pipeline_used", "css_fallback")
    elapsed = time.time() - t0
    logger.info(f"Welcome completed in {elapsed:.1f}s — {len(frames)} frames, pipeline={pipeline_used}")

    return {
        "text": greeting_text,
        "audio_b64": base64.b64encode(audio_bytes).decode(),
        "frame_count": len(frames),
        "pipeline_used": pipeline_used,
        "elapsed_s": round(elapsed, 2),
    }


@app.post("/chat")
async def chat(message: str = "", audio: UploadFile | None = File(None)):
    """Conversational endpoint — text or audio in, animated Eve response out."""
    t0 = time.time()

    # If audio input, transcribe first
    if audio:
        audio_bytes_in = await audio.read()
        try:
            message = await xai_client.speech_to_text(audio_bytes_in)
        except Exception as e:
            logger.error(f"STT failed: {e}")
            return JSONResponse(status_code=503, content={"error": f"STT failed: {e}"})

    if not message:
        return JSONResponse(status_code=400, content={"error": "No message provided"})

    # Generate response via Grok-4
    conversation_history.append({"role": "user", "content": message})
    try:
        response_text = await xai_client.generate_response(message, conversation_history[-20:])
    except Exception as e:
        logger.error(f"Grok response failed: {e}")
        response_text = "I'm having trouble thinking right now. Let me try again in a moment."

    conversation_history.append({"role": "assistant", "content": response_text})

    # TTS
    try:
        audio_bytes = await xai_client.text_to_speech(response_text)
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        audio_bytes = b""

    # Animate
    frames = []
    pipeline_used = "none"
    if audio_bytes:
        try:
            result = await _route_animation(audio_bytes)
            frames = result.get("frames", [])
            pipeline_used = result.get("pipeline_used", "unknown")
        except Exception as e:
            logger.error(f"Animation failed: {e}")

    if frames:
        asyncio.create_task(_broadcast_frames(frames))

    elapsed = time.time() - t0
    return {
        "user_message": message,
        "response": response_text,
        "audio_b64": base64.b64encode(audio_bytes).decode() if audio_bytes else "",
        "frame_count": len(frames),
        "pipeline_used": pipeline_used,
        "elapsed_s": round(elapsed, 2),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket for real-time frame streaming to the React frontend."""
    await ws.accept()
    active_connections.append(ws)
    logger.info(f"WebSocket connected. Total: {len(active_connections)}")
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        active_connections.remove(ws)
        logger.info(f"WebSocket disconnected. Total: {len(active_connections)}")


@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("EDEN OS V2 Gateway starting...")
    logger.info(f"  Router: {settings.router_url}")
    logger.info(f"  xAI key: {'configured' if settings.xai_api_key else 'MISSING'}")
    logger.info(f"  HF token: {'configured' if settings.hf_token else 'MISSING'}")
    logger.info("=" * 60)
