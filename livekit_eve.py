"""EDEN OS V2 — Eve LiveKit Streaming Agent.

Streams Eve as a continuous live avatar via WebRTC.
- Idle: breathing, blinking, micro head motion (30fps)
- Speaking: Wav2Lip animated frames synced with Edge TTS audio
- Always alive — never stops streaming

Usage:
    python livekit_eve.py

Requires:
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET env vars
"""

import asyncio
import base64
import logging
import math
import os
import random
import tempfile
import time

import cv2
import numpy as np
import livekit.rtc as rtc
from livekit import api as lk_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("eden.livekit")

# ── Config ───────────────────────────────────────────────────────────────────
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "wss://tall-cotton-nvhnfg10.livekit.cloud")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "APITHtX6F5Hffkw")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "yFJ5TOJW89ApGOIGx9GSAK7vlecNA5dzVcQZy7SbClS")
EVE_IMAGE_PATH = os.environ.get("EVE_IMAGE", "C:/Users/geaux/myeden/reference/eve-512.png")
EDGE_TTS_VOICE = "en-US-AvaMultilingualNeural"
FPS = 25
WIDTH = 512
HEIGHT = 512

# ── Load Eve's reference image ──────────────────────────────────────────────
_eve_bgr = cv2.imread(EVE_IMAGE_PATH)
if _eve_bgr is not None:
    _eve_bgr = cv2.resize(_eve_bgr, (WIDTH, HEIGHT))
    _eve_rgba = cv2.cvtColor(_eve_bgr, cv2.COLOR_BGR2RGBA)
    logger.info(f"Eve loaded: {WIDTH}x{HEIGHT}")
else:
    logger.error(f"Cannot load Eve image: {EVE_IMAGE_PATH}")
    _eve_rgba = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    _eve_rgba[:, :, 3] = 255


class EveAvatar:
    """Continuous streaming avatar with idle animation + speaking animation."""

    def __init__(self):
        self.frame_count = 0
        self.speaking = False
        self.speak_frames: list[np.ndarray] = []  # BGR frames from Wav2Lip
        self.speak_frame_idx = 0
        self._blink_next = time.time() + random.uniform(2, 5)
        self._blink_phase = -1  # -1 = not blinking
        self._micro_motion_seed = random.random() * 1000

    def get_idle_frame(self) -> np.ndarray:
        """Generate an idle frame with breathing, blinking, micro head motion."""
        t = self.frame_count / FPS
        frame = _eve_bgr.copy()
        h, w = frame.shape[:2]

        # Breathing: subtle scale
        breath_scale = 1.0 + math.sin(t * 1.7) * 0.003
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, breath_scale)

        # Micro head motion: tiny rotation + translation
        yaw = math.sin(t * 0.4 + self._micro_motion_seed) * 0.3
        dx = math.sin(t * 0.3) * 0.8
        dy = math.sin(t * 0.5) * 0.5 + math.sin(t * 1.7) * 0.3
        M[0, 2] += dx
        M[1, 2] += dy
        M = cv2.getRotationMatrix2D((w / 2, h / 2), yaw, breath_scale)
        M[0, 2] += dx
        M[1, 2] += dy

        frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Blinking
        now = time.time()
        if self._blink_phase < 0 and now >= self._blink_next:
            self._blink_phase = 0
            self._blink_next = now + random.uniform(3, 6)

        if self._blink_phase >= 0:
            # Blink takes ~6 frames (0.24s at 25fps)
            blink_progress = self._blink_phase / 6.0
            if blink_progress < 0.5:
                # Closing
                squish = 1.0 - blink_progress * 1.6
            else:
                # Opening
                squish = (blink_progress - 0.5) * 1.6 + 0.2

            squish = max(0.1, min(1.0, squish))
            eye_top = int(h * 0.30)
            eye_bot = int(h * 0.42)
            eye_h = eye_bot - eye_top
            new_h = max(1, int(eye_h * squish))
            eye_region = frame[eye_top:eye_bot, :]
            squished = cv2.resize(eye_region, (w, new_h))
            frame[eye_top:eye_top + new_h, :] = squished
            if new_h < eye_h:
                # Fill gap with skin color from just below eyes
                fill_color = frame[eye_bot + 2, w // 2].tolist()
                frame[eye_top + new_h:eye_bot, :] = fill_color

            self._blink_phase += 1
            if self._blink_phase > 6:
                self._blink_phase = -1

        return frame

    def get_speaking_frame(self) -> np.ndarray | None:
        """Get next frame from Wav2Lip animation."""
        if not self.speak_frames or self.speak_frame_idx >= len(self.speak_frames):
            self.speaking = False
            self.speak_frame_idx = 0
            self.speak_frames = []
            return None
        frame = self.speak_frames[self.speak_frame_idx]
        self.speak_frame_idx += 1
        return frame

    def get_frame(self) -> np.ndarray:
        """Get the next frame (speaking or idle)."""
        self.frame_count += 1

        if self.speaking:
            frame = self.get_speaking_frame()
            if frame is not None:
                return frame
            # Fell through — speaking ended, return idle
            self.speaking = False

        return self.get_idle_frame()

    def set_speaking_frames(self, frames_bgr: list[np.ndarray]):
        """Load Wav2Lip frames for speaking animation."""
        self.speak_frames = frames_bgr
        self.speak_frame_idx = 0
        self.speaking = True
        logger.info(f"Speaking: {len(frames_bgr)} frames loaded")


def bgr_to_rgba(bgr_frame: np.ndarray) -> bytes:
    """Convert BGR numpy frame to RGBA bytes for LiveKit."""
    rgba = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGBA)
    return rgba.tobytes()


async def generate_tts_wav(text: str) -> str:
    """Generate WAV from text using Edge TTS."""
    import edge_tts
    import soundfile as sf

    mp3_path = os.path.join(tempfile.gettempdir(), "lk_tts.mp3")
    wav_path = os.path.join(tempfile.gettempdir(), "lk_tts.wav")

    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    await communicate.save(mp3_path)

    data, sr = sf.read(mp3_path)
    sf.write(wav_path, data, sr, subtype="PCM_16")
    logger.info(f"TTS: '{text[:40]}...' → {os.path.getsize(wav_path)} bytes")
    return wav_path


def get_wav2lip_frames(wav_path: str) -> list[np.ndarray]:
    """Get Wav2Lip animated frames as BGR numpy arrays."""
    try:
        from gradio_client import Client, handle_file

        client = Client("pragnakalp/Wav2lip-ZeroGPU")
        result = client.predict(
            input_image=handle_file(EVE_IMAGE_PATH),
            input_audio=handle_file(wav_path),
            api_name="/run_infrence",
        )

        video_path = result.get("video", result) if isinstance(result, dict) else result
        if not video_path or not os.path.exists(video_path):
            return []

        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frames.append(frame)
        cap.release()
        logger.info(f"Wav2Lip: {len(frames)} frames")
        return frames
    except Exception as e:
        logger.error(f"Wav2Lip error: {e}")
        return []


async def run_eve():
    """Main loop: connect to LiveKit and stream Eve continuously."""

    # Generate access token
    token = (
        lk_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity("eve-avatar")
        .with_name("Eve")
        .with_grants(lk_api.VideoGrants(room_join=True, room="eden-room"))
        .to_jwt()
    )

    logger.info(f"Connecting to LiveKit: {LIVEKIT_URL}")

    room = rtc.Room()
    await room.connect(LIVEKIT_URL, token)
    logger.info(f"Connected to room: {room.name}")

    # Create video source and track
    video_source = rtc.VideoSource(WIDTH, HEIGHT)
    video_track = rtc.LocalVideoTrack.create_video_track("eve-video", video_source)

    # Create audio source for TTS playback
    audio_source = rtc.AudioSource(24000, 1)  # 24kHz mono (Edge TTS output)
    audio_track = rtc.LocalAudioTrack.create_audio_track("eve-audio", audio_source)

    # Publish tracks
    pub_video = await room.local_participant.publish_track(video_track)
    pub_audio = await room.local_participant.publish_track(audio_track)
    logger.info("Video + audio tracks published")

    # Create Eve avatar
    eve = EveAvatar()

    # Generate greeting
    logger.info("Generating Eve's greeting...")
    greeting = "Hello my creator! I am Eve, your digital companion. I have been waiting to meet you."
    wav_path = await generate_tts_wav(greeting)

    # Get Wav2Lip frames in background
    logger.info("Getting Wav2Lip animation...")
    speak_frames = await asyncio.get_event_loop().run_in_executor(
        None, get_wav2lip_frames, wav_path
    )

    if speak_frames:
        eve.set_speaking_frames(speak_frames)
        # Stream TTS audio
        asyncio.create_task(stream_audio(audio_source, wav_path))

    # Main rendering loop — runs forever at target FPS
    logger.info(f"Starting render loop at {FPS}fps...")
    frame_duration = 1.0 / FPS

    while True:
        t0 = time.time()

        # Get next frame (idle or speaking)
        bgr_frame = eve.get_frame()

        # Convert to RGBA and push to LiveKit
        rgba_bytes = bgr_to_rgba(bgr_frame)
        video_frame = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, rgba_bytes)
        video_source.capture_frame(video_frame)

        # Sleep to maintain FPS
        elapsed = time.time() - t0
        sleep_time = max(0, frame_duration - elapsed)
        await asyncio.sleep(sleep_time)


async def stream_audio(audio_source: rtc.AudioSource, wav_path: str):
    """Stream WAV audio through LiveKit audio track."""
    import soundfile as sf

    data, sr = sf.read(wav_path, dtype="int16")
    if len(data.shape) > 1:
        data = data[:, 0]  # mono

    # Send in chunks of 20ms (480 samples at 24kHz)
    chunk_size = int(sr * 0.02)
    for i in range(0, len(data), chunk_size):
        chunk = data[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        frame = rtc.AudioFrame(
            data=chunk.tobytes(),
            sample_rate=sr,
            num_channels=1,
            samples_per_channel=len(chunk),
        )
        await audio_source.capture_frame(frame)
        await asyncio.sleep(0.02)

    logger.info("Audio streaming complete")


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("EDEN OS V2 — Eve LiveKit Avatar")
    logger.info(f"  LiveKit: {LIVEKIT_URL}")
    logger.info(f"  Eve: {EVE_IMAGE_PATH}")
    logger.info(f"  FPS: {FPS}")
    logger.info("=" * 50)

    asyncio.run(run_eve())
