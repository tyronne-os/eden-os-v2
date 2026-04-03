"""EDEN OS V2 — Eve via bitHuman Direct API + LiveKit WebRTC.

bitHuman renders Eve's face neurally on CPU.
LiveKit streams the frames via WebRTC.
Edge TTS provides the voice.

Usage:
    python livekit_eve_bithuman.py
"""

import asyncio
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
logger = logging.getLogger("eden.bithuman")

# Config
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "wss://tall-cotton-nvhnfg10.livekit.cloud")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "APITHtX6F5Hffkw")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "yFJ5TOJW89ApGOIGx9GSAK7vlecNA5dzVcQZy7SbClS")
BITHUMAN_API_SECRET = os.environ.get("BITHUMAN_API_SECRET", "AmiK3xBgSyMFkPlS5mQ3N0CaAjDSoWpjS4l5jzs5ZOylbgoeow9o1mL3R2jZLlPkd")
EVE_IMAGE = os.environ.get("EVE_IMAGE", "C:/Users/geaux/myeden/reference/eve-512.png")
EDGE_TTS_VOICE = "en-US-AvaMultilingualNeural"
FPS = 25


async def generate_tts_wav(text: str) -> tuple[str, bytes, int]:
    """Text → WAV. Returns (path, raw_bytes, sample_rate)."""
    import edge_tts

    mp3_path = os.path.join(tempfile.gettempdir(), "bh_tts.mp3")
    wav_path = os.path.join(tempfile.gettempdir(), "bh_tts.wav")

    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    await communicate.save(mp3_path)

    data, sr = sf.read(mp3_path, dtype="int16")
    sf.write(wav_path, data, sr, subtype="PCM_16")

    # Read raw bytes for bitHuman
    with open(wav_path, "rb") as f:
        wav_bytes = f.read()

    logger.info(f"TTS: {len(text)} chars → {len(wav_bytes)} bytes WAV ({len(data)/sr:.1f}s)")
    return wav_path, data.tobytes(), sr


async def run():
    """Main loop: bitHuman renders Eve, LiveKit streams to browser."""

    # 1. Initialize bitHuman
    logger.info("Initializing bitHuman neural renderer...")
    bh = AsyncBithuman(api_secret=BITHUMAN_API_SECRET)

    # Load Eve's neural model (.imx file from bitHuman)
    eve_model = "C:/Users/geaux/myeden/reference/eve_bithuman.imx"
    logger.info(f"Loading Eve neural model: {eve_model} (215MB)...")
    await bh.set_model(eve_model)
    await bh.load_data_async()
    logger.info("Eve neural model loaded!")

    # Get first frame to confirm it's working
    first_frame = bh.get_first_frame()
    if first_frame is not None:
        h, w = first_frame.shape[:2]
        logger.info(f"bitHuman initialized! Frame size: {w}x{h}")
    else:
        logger.error("bitHuman failed to generate first frame")
        return

    # Start bitHuman processing loop
    await bh.start()
    logger.info("bitHuman render engine started")

    # 2. Connect to LiveKit
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

    # Create video + audio tracks
    video_source = rtc.VideoSource(w, h)
    video_track = rtc.LocalVideoTrack.create_video_track("eve-video", video_source)
    audio_source = rtc.AudioSource(24000, 1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("eve-audio", audio_source)

    await room.local_participant.publish_track(video_track)
    await room.local_participant.publish_track(audio_track)
    logger.info("Video + audio tracks published to LiveKit")

    # 3. Generate greeting TTS
    logger.info("Generating Eve's greeting...")
    greeting = "Hello my creator! I am Eve. I have been waiting to meet you."
    wav_path, audio_bytes, sr = await generate_tts_wav(greeting)

    # 4. Prepare audio chunks for bitHuman
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    chunk_duration = 0.04  # 40ms chunks
    chunk_samples = int(sr * chunk_duration)
    audio_chunks = []
    for i in range(0, len(audio_data), chunk_samples):
        chunk = audio_data[i:i + chunk_samples]
        is_last = (i + chunk_samples >= len(audio_data))
        audio_chunks.append(AudioChunk(data=chunk, sample_rate=sr, last_chunk=is_last))
    logger.info(f"Prepared {len(audio_chunks)} audio chunks for bitHuman")

    # 5. Stream audio to LiveKit in parallel
    async def stream_lk_audio():
        data_i16, _ = sf.read(wav_path, dtype="int16")
        lk_chunk_size = int(sr * 0.02)
        for i in range(0, len(data_i16), lk_chunk_size):
            chunk = data_i16[i:i + lk_chunk_size]
            if len(chunk) < lk_chunk_size:
                chunk = np.pad(chunk, (0, lk_chunk_size - len(chunk)))
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=sr,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
            await audio_source.capture_frame(frame)
            await asyncio.sleep(0.02)
        logger.info("LiveKit audio streaming complete")

    asyncio.create_task(stream_lk_audio())

    # 6. Main render loop — feed audio chunks to bitHuman, push frames to LiveKit
    logger.info(f"Starting render loop at {FPS}fps — Eve is ALIVE!")
    frame_duration = 1.0 / FPS
    frame_count = 0
    chunk_idx = 0

    while True:
        t0 = time.time()

        # Build VideoControl with next audio chunk (or idle)
        if chunk_idx < len(audio_chunks):
            control = VideoControl(audio=audio_chunks[chunk_idx])
            chunk_idx += 1
        else:
            # Idle mode — no audio, just idle animation
            control = VideoControl()

        # Get neurally-rendered frames from bitHuman
        for video_frame in bh.process(control):
            if video_frame is not None and video_frame.has_image:
                rgb = video_frame.rgb_image

                # Convert RGB → RGBA for LiveKit
                rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
                lk_frame = rtc.VideoFrame(
                    rgba.shape[1], rgba.shape[0],
                    rtc.VideoBufferType.RGBA,
                    rgba.tobytes(),
                )
                video_source.capture_frame(lk_frame)
                frame_count += 1

                if frame_count % 100 == 0:
                    logger.info(f"Streamed {frame_count} neural frames")

        elapsed = time.time() - t0
        sleep_time = max(0, frame_duration - elapsed)
        await asyncio.sleep(sleep_time)


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("EDEN OS V2 — bitHuman Neural Avatar")
    logger.info(f"  Eve: {EVE_IMAGE}")
    logger.info(f"  LiveKit: {LIVEKIT_URL}")
    logger.info(f"  bitHuman: {'configured' if BITHUMAN_API_SECRET else 'MISSING'}")
    logger.info(f"  Renderer: CPU neural (no GPU needed)")
    logger.info("=" * 50)

    asyncio.run(run())
