"""xAI Grok-4 chat + TTS client using OpenAI-compatible API."""

import io
import logging
from openai import AsyncOpenAI

from .config import settings

logger = logging.getLogger("eden.xai")

_client: AsyncOpenAI | None = None


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
    """Convert text to speech using xAI TTS with Eve voice."""
    import httpx as _httpx

    # Use xAI's dedicated TTS endpoint
    async with _httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.post(
            "https://api.x.ai/v1/tts",
            headers={
                "Authorization": f"Bearer {settings.xai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "voice_id": settings.eve_voice,
                "language": "en",
            },
        )
        resp.raise_for_status()
        return resp.content


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
