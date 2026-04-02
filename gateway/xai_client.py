"""EDEN OS V2 — xAI Grok-4 chat + Edge TTS voice client.

Brain: xAI Grok-4 (via OpenAI-compatible API)
Voice: Microsoft Edge TTS (en-US-AvaMultilingualNeural) — outputs clean WAV
"""

import asyncio
import io
import logging
import os
import tempfile
import wave

import soundfile as sf
import numpy as np
from openai import AsyncOpenAI

from .config import settings

logger = logging.getLogger("eden.xai")

_client: AsyncOpenAI | None = None

# Edge TTS voice (warm, natural female)
EDGE_TTS_VOICE = "en-US-AvaMultilingualNeural"


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.xai_api_key,
            base_url="https://api.x.ai/v1",
        )
    return _client


async def generate_response(user_message: str, history: list[dict] | None = None) -> str:
    """Generate a text response from Grok-4."""
    client = _get_client()
    messages = [
        {
            "role": "system",
            "content": (
                "You are Eve, a warm, intelligent, emotionally expressive digital avatar. "
                "You speak naturally and conversationally. Keep responses concise — "
                "2-3 sentences max unless asked for detail. Be genuine, curious, and kind."
            ),
        }
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    resp = await client.chat.completions.create(
        model="grok-4-fast-non-reasoning",
        messages=messages,
        max_tokens=256,
        temperature=0.7,
    )
    return resp.choices[0].message.content


async def generate_greeting() -> str:
    """Generate Eve's first greeting."""
    client = _get_client()
    resp = await client.chat.completions.create(
        model="grok-4-fast-non-reasoning",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Eve, a digital avatar greeting your creator for the first time. "
                    "Be warm, excited, and genuine. One sentence only."
                ),
            },
            {"role": "user", "content": "Say hello to me for the first time."},
        ],
        max_tokens=80,
        temperature=0.9,
    )
    return resp.choices[0].message.content


async def text_to_speech(text: str) -> bytes:
    """Convert text to speech using Edge TTS.

    Returns clean 16-bit mono WAV bytes at 24kHz.
    Edge TTS produces natural-sounding speech with zero API cost.
    """
    try:
        import edge_tts
    except ImportError:
        logger.error("edge_tts not installed — run: uv pip install edge-tts")
        raise RuntimeError("edge_tts not installed")

    # Generate audio via Edge TTS
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)

    # Collect MP3 chunks
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    if not audio_data:
        raise RuntimeError("Edge TTS returned empty audio")

    # Convert MP3 → WAV using soundfile
    mp3_path = os.path.join(tempfile.gettempdir(), "eden_edge_tts.mp3")
    wav_path = os.path.join(tempfile.gettempdir(), "eden_edge_tts.wav")

    with open(mp3_path, "wb") as f:
        f.write(audio_data)

    data, sr = sf.read(mp3_path)
    sf.write(wav_path, data, sr, subtype="PCM_16")

    with open(wav_path, "rb") as f:
        wav_bytes = f.read()

    logger.info(f"Edge TTS: {len(text)} chars → {len(wav_bytes)} bytes WAV ({len(data)/sr:.1f}s)")
    return wav_bytes


async def speech_to_text(audio_bytes: bytes) -> str:
    """Transcribe audio using xAI Whisper-compatible endpoint."""
    client = _get_client()
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"
    resp = await client.audio.transcriptions.create(
        model="grok-3-mini",
        file=audio_file,
        language="en",
    )
    return resp.text
