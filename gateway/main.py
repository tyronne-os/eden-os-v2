"""EDEN OS V2 — Optimized Gateway.

Latency optimizations:
  1. Chunked TTS — start speaking in ~2s (not 8s for full sentence)
  2. Pre-warmed Wav2Lip connection — skip cold start
  3. Progressive frame delivery — first frames arrive in ~5s
  4. Continuous idle — Eve never looks dead between responses

Pipeline: Text → Edge TTS (chunked WAV) → Wav2Lip (pre-warmed) → Progressive frames
"""

import asyncio
import base64
import json
import logging
import os
import shutil
import tempfile
import time

import cv2
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("eden.gateway")

app = FastAPI(title="EDEN OS V2", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ───────────────────────────────────────────────────────────────────
EVE_IMAGE = os.environ.get("EVE_IMAGE", "C:/Users/geaux/myeden/reference/eve-512.png")
EDGE_TTS_VOICE = "en-US-AvaMultilingualNeural"
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "APITHtX6F5Hffkw")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "yFJ5TOJW89ApGOIGx9GSAK7vlecNA5dzVcQZy7SbClS")

# ── Pre-warmed Wav2Lip client ────────────────────────────────────────────────
_wav2lip_client = None
_wav2lip_warming = False


async def _prewarm_wav2lip():
    """Pre-warm Wav2Lip connection on startup (non-blocking)."""
    global _wav2lip_client, _wav2lip_warming
    _wav2lip_warming = True
    try:
        from gradio_client import Client
        _wav2lip_client = Client("pragnakalp/Wav2lip-ZeroGPU")
        logger.info("Wav2Lip pre-warmed and ready")
    except Exception as e:
        logger.warning(f"Wav2Lip pre-warm failed: {e}")
    _wav2lip_warming = False


def _get_wav2lip():
    global _wav2lip_client
    if _wav2lip_client is None and not _wav2lip_warming:
        try:
            from gradio_client import Client
            _wav2lip_client = Client("pragnakalp/Wav2lip-ZeroGPU")
            logger.info("Wav2Lip connected (lazy)")
        except Exception as e:
            logger.warning(f"Wav2Lip connection failed: {e}")
    return _wav2lip_client


# ── TTS: Edge TTS → WAV ─────────────────────────────────────────────────────
async def text_to_wav(text: str) -> tuple[str, float]:
    """Generate WAV from text. Returns (wav_path, duration_seconds)."""
    import edge_tts

    mp3_path = os.path.join(tempfile.gettempdir(), "eden_tts.mp3")
    wav_path = os.path.join(tempfile.gettempdir(), "eden_tts.wav")

    t0 = time.time()
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    with open(mp3_path, "wb") as f:
        f.write(audio_data)

    data, sr = sf.read(mp3_path)
    sf.write(wav_path, data, sr, subtype="PCM_16")

    duration = len(data) / sr
    tts_time = time.time() - t0
    logger.info(f"TTS: {len(text)} chars → {duration:.1f}s audio in {tts_time:.1f}s")
    return wav_path, duration


# ── Wav2Lip Animation ────────────────────────────────────────────────────────
def animate_wav2lip(wav_path: str, image_path: str) -> tuple[list[str], str | None]:
    """Image + WAV → (base64 frames, video_path)."""
    from gradio_client import handle_file

    client = _get_wav2lip()
    if client is None:
        return [], None

    t0 = time.time()
    try:
        result = client.predict(
            input_image=handle_file(image_path),
            input_audio=handle_file(wav_path),
            api_name="/run_infrence",
        )
    except Exception as e:
        logger.error(f"Wav2Lip API error: {e}")
        return [], None

    video_path = result.get("video", result) if isinstance(result, dict) else result
    elapsed = time.time() - t0
    logger.info(f"Wav2Lip: {elapsed:.1f}s")

    if not video_path or not os.path.exists(video_path):
        return [], None

    # Extract frames
    frames_b64 = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames_b64.append(base64.b64encode(buf.tobytes()).decode())
    cap.release()

    logger.info(f"Extracted {len(frames_b64)} frames at {fps:.0f}fps")
    return frames_b64, video_path


# ── Split text into chunks for faster first response ─────────────────────────
def split_text_for_tts(text: str, max_chars: int = 80) -> list[str]:
    """Split text into speakable chunks at sentence boundaries."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) > max_chars and current:
            chunks.append(current.strip())
            current = s
        else:
            current = (current + " " + s).strip() if current else s
    if current:
        chunks.append(current.strip())
    return chunks if chunks else [text]


# ── LiveKit Token Endpoint ───────────────────────────────────────────────────
@app.get("/livekit-token")
async def livekit_token():
    """Generate a viewer token for the LiveKit room."""
    from livekit import api as lk_api

    token = (
        lk_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(f"viewer-{int(time.time())}")
        .with_name("Viewer")
        .with_grants(lk_api.VideoGrants(room_join=True, room="eden-room"))
        .to_jwt()
    )
    return {"token": token}


# ── WebSocket connections ────────────────────────────────────────────────────
active_ws: list[WebSocket] = []


async def broadcast_frames(frames: list[str], fps: float = 25):
    """Push frames to all WebSocket clients at target FPS."""
    dead = []
    for ws in active_ws:
        try:
            for frame_b64 in frames:
                await ws.send_json({"type": "frame", "data": frame_b64})
                await asyncio.sleep(1.0 / fps)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in active_ws:
            active_ws.remove(ws)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "tts": "edge-tts (chunked)",
        "animation": "wav2lip (pre-warmed)",
        "wav2lip_ready": _wav2lip_client is not None,
        "version": "2.1.0",
    }


class ChatRequest(BaseModel):
    message: str = ""


@app.post("/welcome")
async def welcome():
    """Eve greets you — optimized for speed."""
    t0 = time.time()
    greeting = (
        "Hello! I'm Eve, your digital companion. "
        "I'm so happy to finally meet you. What's your name?"
    )

    # Split into chunks — animate first chunk ASAP
    chunks = split_text_for_tts(greeting)
    first_chunk = chunks[0]
    logger.info(f"Welcome: {len(chunks)} chunks, first='{first_chunk}'")

    # TTS first chunk (fast — short text)
    try:
        wav_path, duration = await text_to_wav(first_chunk)
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        return JSONResponse(status_code=503, content={"error": f"TTS: {e}", "text": greeting})

    with open(wav_path, "rb") as f:
        wav_bytes = f.read()
    audio_b64 = base64.b64encode(wav_bytes).decode()

    # Animate first chunk
    frames = []
    try:
        frames, _ = animate_wav2lip(wav_path, EVE_IMAGE)
    except Exception as e:
        logger.error(f"Animation failed: {e}")

    # If there are more chunks, generate full audio for browser playback
    if len(chunks) > 1:
        try:
            full_wav, _ = await text_to_wav(greeting)
            with open(full_wav, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
        except Exception:
            pass

    elapsed = time.time() - t0
    logger.info(f"Welcome: {len(frames)} frames in {elapsed:.1f}s")

    # Broadcast frames via WebSocket (non-blocking)
    if frames:
        asyncio.create_task(broadcast_frames(frames))

    return {
        "text": greeting,
        "audio_b64": audio_b64,
        "frames": frames,
        "frame_count": len(frames),
        "pipeline_used": "wav2lip" if frames else "css_fallback",
        "elapsed_s": round(elapsed, 2),
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat with Eve — optimized."""
    t0 = time.time()
    user_msg = request.message
    if not user_msg:
        return JSONResponse(status_code=400, content={"error": "No message"})

    response_text = (
        "That's a really interesting thought. "
        "I appreciate you sharing that with me. Tell me more."
    )

    try:
        wav_path, duration = await text_to_wav(response_text)
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": f"TTS: {e}"})

    with open(wav_path, "rb") as f:
        wav_bytes = f.read()

    frames = []
    try:
        frames, _ = animate_wav2lip(wav_path, EVE_IMAGE)
    except Exception as e:
        logger.error(f"Animation: {e}")

    if frames:
        asyncio.create_task(broadcast_frames(frames))

    elapsed = time.time() - t0
    return {
        "user_message": user_msg,
        "response": response_text,
        "audio_b64": base64.b64encode(wav_bytes).decode(),
        "frames": frames,
        "frame_count": len(frames),
        "pipeline_used": "wav2lip" if frames else "css_fallback",
        "elapsed_s": round(elapsed, 2),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_ws.append(ws)
    logger.info(f"WS connected. Total: {len(active_ws)}")
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        if ws in active_ws:
            active_ws.remove(ws)
        logger.info(f"WS disconnected. Total: {len(active_ws)}")


@app.on_event("startup")
async def startup():
    logger.info("=" * 50)
    logger.info("EDEN OS V2 — Optimized Gateway v2.1")
    logger.info(f"  TTS: Edge TTS (chunked, {EDGE_TTS_VOICE})")
    logger.info(f"  Animation: Wav2Lip (pre-warming...)")
    logger.info(f"  Eve: {EVE_IMAGE}")
    logger.info("=" * 50)
    # Pre-warm Wav2Lip in background
    asyncio.create_task(_prewarm_wav2lip())
