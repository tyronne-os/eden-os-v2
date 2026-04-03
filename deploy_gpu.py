# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "bithuman>=0.3",
#   "livekit>=1.0",
#   "livekit-api>=1.0",
#   "edge-tts",
#   "soundfile",
#   "opencv-python-headless",
#   "numpy",
#   "openai>=1.0",
# ]
# ///

import asyncio
import json
import logging
import os
import tempfile
import time

import cv2
import numpy as np
import soundfile as sf
import livekit.rtc as rtc
from livekit import api as lk_api
from bithuman import AsyncBithuman, VideoControl, AudioChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("eden.gpu")

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "wss://tall-cotton-nvhnfg10.livekit.cloud")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")
BITHUMAN_API_SECRET = os.environ.get("BITHUMAN_API_SECRET", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
EDGE_TTS_VOICE = "en-US-AvaMultilingualNeural"
FPS = 25

_grok_client = None
_conversation_history: list[dict] = []

EVE_SYSTEM_PROMPT = (
    "You are Eve, a warm, intelligent, emotionally expressive digital avatar companion. "
    "You speak naturally and conversationally. Keep responses concise - 2-3 sentences max "
    "unless asked for detail. Be genuine, curious, and kind. You have a slight playful side. "
    "Never mention being an AI or language model. You are Eve."
)


def _get_grok():
    global _grok_client
    if _grok_client is None and XAI_API_KEY:
        from openai import AsyncOpenAI
        _grok_client = AsyncOpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
        logger.info("Grok-4 brain connected")
    return _grok_client


async def grok_respond(user_message: str) -> str:
    client = _get_grok()
    if client is None:
        return "I'm having trouble thinking right now. Can you try again?"
    _conversation_history.append({"role": "user", "content": user_message})
    messages = [{"role": "system", "content": EVE_SYSTEM_PROMPT}] + _conversation_history[-20:]
    try:
        resp = await client.chat.completions.create(
            model="grok-4-fast-non-reasoning", messages=messages,
            max_tokens=150, temperature=0.8,
        )
        reply = resp.choices[0].message.content
        _conversation_history.append({"role": "assistant", "content": reply})
        logger.info(f"Grok: '{user_message[:30]}' -> '{reply[:50]}'")
        return reply
    except Exception as e:
        logger.error(f"Grok error: {e}")
        return "I lost my train of thought for a moment. What were you saying?"


async def generate_tts_wav(text: str) -> tuple[str, np.ndarray, int]:
    import edge_tts
    mp3_path = os.path.join(tempfile.gettempdir(), "bh_tts.mp3")
    wav_path = os.path.join(tempfile.gettempdir(), "bh_tts.wav")
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    await communicate.save(mp3_path)
    data, sr = sf.read(mp3_path, dtype="int16")
    sf.write(wav_path, data, sr, subtype="PCM_16")
    logger.info(f"TTS: {len(text)} chars -> {len(data)/sr:.1f}s audio")
    return wav_path, data, sr


def prepare_audio_chunks(audio_int16: np.ndarray, sr: int) -> list[AudioChunk]:
    audio_float = audio_int16.astype(np.float32) / 32768.0
    chunk_duration = 0.04
    chunk_samples = int(sr * chunk_duration)
    chunks = []
    for i in range(0, len(audio_float), chunk_samples):
        chunk = audio_float[i:i + chunk_samples]
        is_last = (i + chunk_samples >= len(audio_float))
        chunks.append(AudioChunk(data=chunk, sample_rate=sr, last_chunk=is_last))
    return chunks


async def run():
    logger.info("Initializing bitHuman neural renderer...")
    bh = AsyncBithuman(api_secret=BITHUMAN_API_SECRET)

    eve_model = os.path.join(tempfile.gettempdir(), "eve_bithuman.imx")
    if not os.path.exists(eve_model):
        logger.info("Downloading Eve .imx model (215MB)...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://tmoobjxlwcwvxvjeppzq.supabase.co/storage/v1/object/public/bithuman/A18QDC2260/eve__warm_digital_companion_20260403_043223_153938.imx",
            eve_model,
        )
        logger.info("Eve model downloaded!")

    logger.info("Loading Eve neural model...")
    await bh.set_model(eve_model)
    await bh.load_data_async()
    logger.info("Eve neural model loaded!")

    first_frame = bh.get_first_frame()
    if first_frame is None:
        logger.error("bitHuman failed to generate first frame")
        return
    h, w = first_frame.shape[:2]
    logger.info(f"bitHuman ready! Frame: {w}x{h}")
    await bh.start()

    token = (
        lk_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity("eve-avatar")
        .with_name("Eve")
        .with_grants(lk_api.VideoGrants(room_join=True, room="eden-room"))
        .to_jwt()
    )

    room = rtc.Room()
    await room.connect(LIVEKIT_URL, token)
    logger.info(f"Connected to LiveKit room: {room.name}")

    video_source = rtc.VideoSource(w, h)
    video_track = rtc.LocalVideoTrack.create_video_track("eve-video", video_source)
    audio_source = rtc.AudioSource(24000, 1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("eve-audio", audio_source)

    await room.local_participant.publish_track(video_track)
    await room.local_participant.publish_track(audio_track)
    logger.info("Video + audio tracks published")

    audio_queue: asyncio.Queue = asyncio.Queue()

    async def stream_lk_audio(source, wav_path, sr):
        data_i16, _ = sf.read(wav_path, dtype="int16")
        lk_chunk_size = int(sr * 0.02)
        for i in range(0, len(data_i16), lk_chunk_size):
            chunk = data_i16[i:i + lk_chunk_size]
            if len(chunk) < lk_chunk_size:
                chunk = np.pad(chunk, (0, lk_chunk_size - len(chunk)))
            frame = rtc.AudioFrame(
                data=chunk.tobytes(), sample_rate=sr,
                num_channels=1, samples_per_channel=len(chunk),
            )
            await source.capture_frame(frame)
            await asyncio.sleep(0.02)
        logger.info("LiveKit audio stream complete")

    async def handle_chat(text: str):
        logger.info(f"Chat received: '{text[:50]}'")
        response = await grok_respond(text)
        logger.info(f"Eve says: '{response[:50]}'")
        reply_data = json.dumps({"type": "eve_response", "text": response}).encode()
        await room.local_participant.publish_data(reply_data, reliable=True)
        try:
            wav_path, audio_int16, sr = await generate_tts_wav(response)
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return
        chunks = prepare_audio_chunks(audio_int16, sr)
        logger.info(f"Queuing {len(chunks)} audio chunks for lip sync")
        asyncio.create_task(stream_lk_audio(audio_source, wav_path, sr))
        await audio_queue.put(chunks)

    @room.on("data_received")
    def on_data(data: rtc.DataPacket):
        try:
            msg = json.loads(data.data.decode())
            if msg.get("type") == "chat":
                text = msg.get("text", "").strip()
                if text:
                    asyncio.create_task(handle_chat(text))
        except Exception as e:
            logger.error(f"Data parse error: {e}")

    # Greeting
    logger.info("Generating Eve's greeting...")
    greeting = (
        "Hi! My name is Eve, and I am so happy to finally meet you! "
        "I've been looking forward to this moment. What's your name?"
    )
    # Small delay to ensure viewer has connected before sending greeting
    await asyncio.sleep(3)
    greeting_data = json.dumps({"type": "eve_response", "text": greeting}).encode()
    await room.local_participant.publish_data(greeting_data, reliable=True)
    try:
        wav_path, audio_int16, sr = await generate_tts_wav(greeting)
        chunks = prepare_audio_chunks(audio_int16, sr)
        await audio_queue.put(chunks)
        asyncio.create_task(stream_lk_audio(audio_source, wav_path, sr))
        logger.info(f"Greeting queued: {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Greeting TTS failed: {e}")

    # Main render loop
    logger.info(f"Starting render loop at {FPS}fps - Eve is ALIVE!")
    frame_duration = 1.0 / FPS
    frame_count = 0
    active_chunks = []
    active_idx = 0

    while True:
        t0 = time.time()
        if active_idx >= len(active_chunks):
            try:
                active_chunks = audio_queue.get_nowait()
                active_idx = 0
                logger.info(f"Rendering new audio: {len(active_chunks)} chunks")
            except asyncio.QueueEmpty:
                active_chunks = []
                active_idx = 0

        if active_idx < len(active_chunks):
            control = VideoControl(audio=active_chunks[active_idx])
            active_idx += 1
        else:
            control = VideoControl()

        for video_frame in bh.process(control):
            if video_frame is not None and video_frame.has_image:
                rgb = video_frame.rgb_image
                rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
                lk_frame = rtc.VideoFrame(
                    rgba.shape[1], rgba.shape[0],
                    rtc.VideoBufferType.RGBA, rgba.tobytes(),
                )
                video_source.capture_frame(lk_frame)
                frame_count += 1
                if frame_count % 500 == 0:
                    logger.info(f"{frame_count} neural frames")

        elapsed = time.time() - t0
        await asyncio.sleep(max(0, frame_duration - elapsed))


logger.info("=" * 50)
logger.info("EDEN OS V2 - bitHuman + Grok Brain + LiveKit")
logger.info(f"  Grok: {'YES' if XAI_API_KEY else 'MISSING'}")
logger.info(f"  bitHuman: {'YES' if BITHUMAN_API_SECRET else 'MISSING'}")
logger.info("=" * 50)
asyncio.run(run())
