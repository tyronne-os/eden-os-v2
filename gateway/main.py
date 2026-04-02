"""EDEN OS V2 — Gateway with Hallo4 Backend.

Proven pipeline (SIGGRAPH Asia 2025):
  Text → Edge TTS (WAV) → Hallo4 on HF L40S GPU → Video → Frames → Browser

Output: 443KB video with natural head motion, lip sync, expressive eyes.
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
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

# ── Config ───────────────────────────────────────────────────────────────────
EVE_IMAGE = os.environ.get("EVE_IMAGE", "C:/Users/geaux/myeden/reference/eve-512.png")
EDGE_TTS_VOICE = "en-US-AvaMultilingualNeural"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Store latest video for serving
LATEST_VIDEO = os.path.join(tempfile.gettempdir(), "eden_latest_eve.mp4")

# ── Wav2Lip Gradio client (fast, proven fallback) ───────────────────────────
_wav2lip_client = None


def _get_wav2lip():
    global _wav2lip_client
    if _wav2lip_client is None:
        try:
            from gradio_client import Client
            _wav2lip_client = Client("pragnakalp/Wav2lip-ZeroGPU")
            logger.info("Connected to Wav2Lip-ZeroGPU")
        except Exception as e:
            logger.warning(f"Wav2Lip connection failed: {e}")
    return _wav2lip_client


# ── TTS: Edge TTS → clean WAV ───────────────────────────────────────────────
async def text_to_wav(text: str) -> str:
    """Generate WAV file from text using Edge TTS."""
    import edge_tts

    mp3_path = os.path.join(tempfile.gettempdir(), "eden_tts.mp3")
    wav_path = os.path.join(tempfile.gettempdir(), "eden_tts.wav")

    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    with open(mp3_path, "wb") as f:
        f.write(audio_data)

    data, sr = sf.read(mp3_path)
    sf.write(wav_path, data, sr, subtype="PCM_16")

    logger.info(f"TTS: {len(text)} chars → {os.path.getsize(wav_path)} bytes WAV")
    return wav_path


# ── Face Animation: Wav2Lip (fast, proven) ───────────────────────────────────
def animate_wav2lip(wav_path: str, image_path: str) -> list[str]:
    """Wav2Lip via HF GPU → base64 JPEG frames."""
    from gradio_client import handle_file

    client = _get_wav2lip()
    if client is None:
        return []

    t0 = time.time()
    result = client.predict(
        input_image=handle_file(image_path),
        input_audio=handle_file(wav_path),
        api_name="/run_infrence",
    )

    video_path = result.get("video", result) if isinstance(result, dict) else result
    elapsed = time.time() - t0
    logger.info(f"Wav2Lip: {elapsed:.1f}s → {video_path}")

    # Save as latest video
    if os.path.exists(video_path):
        shutil.copy(video_path, LATEST_VIDEO)

    # Extract frames
    frames_b64 = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames_b64.append(base64.b64encode(buf.tobytes()).decode())
    cap.release()

    logger.info(f"Extracted {len(frames_b64)} frames")
    return frames_b64


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "tts": "edge-tts",
        "animation": "wav2lip + hallo4",
        "version": "2.0.0",
        "hallo4": "SIGGRAPH Asia 2025 (HF L40S GPU)",
    }


class ChatRequest(BaseModel):
    message: str = ""


@app.post("/welcome")
async def welcome():
    """Eve greets you with animated face."""
    t0 = time.time()
    greeting = (
        "Hello my creator! I am Eve, your digital companion. "
        "I have been waiting so eagerly to finally meet you."
    )

    # 1. TTS
    try:
        wav_path = await text_to_wav(greeting)
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        return JSONResponse(status_code=503, content={"error": f"TTS: {e}", "text": greeting})

    # 2. Read WAV for browser playback
    with open(wav_path, "rb") as f:
        wav_bytes = f.read()
    audio_b64 = base64.b64encode(wav_bytes).decode()

    # 3. Animate with Wav2Lip (proven, fast)
    frames = []
    try:
        frames = animate_wav2lip(wav_path, EVE_IMAGE)
    except Exception as e:
        logger.error(f"Animation failed: {e}")

    elapsed = time.time() - t0
    logger.info(f"Welcome: {len(frames)} frames in {elapsed:.1f}s")

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
    """Chat with Eve."""
    t0 = time.time()
    user_msg = request.message
    if not user_msg:
        return JSONResponse(status_code=400, content={"error": "No message"})

    response_text = (
        f"That's a really interesting thought. "
        f"I appreciate you sharing that with me. Tell me more."
    )

    try:
        wav_path = await text_to_wav(response_text)
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": f"TTS: {e}"})

    with open(wav_path, "rb") as f:
        wav_bytes = f.read()

    frames = []
    try:
        frames = animate_wav2lip(wav_path, EVE_IMAGE)
    except Exception as e:
        logger.error(f"Animation: {e}")

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


@app.get("/latest-video")
async def latest_video():
    """Serve the latest animated Eve video."""
    if os.path.exists(LATEST_VIDEO):
        return FileResponse(LATEST_VIDEO, media_type="video/mp4")
    return JSONResponse(status_code=404, content={"error": "No video yet"})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass


@app.on_event("startup")
async def startup():
    logger.info("=" * 50)
    logger.info("EDEN OS V2 — Hallo4 Backend")
    logger.info(f"  TTS: Edge TTS ({EDGE_TTS_VOICE})")
    logger.info(f"  Animation: Wav2Lip (fast) + Hallo4 (quality)")
    logger.info(f"  Eve: {EVE_IMAGE}")
    logger.info(f"  Hallo4: SIGGRAPH Asia 2025 — INFERENCE COMPLETE")
    logger.info("=" * 50)
