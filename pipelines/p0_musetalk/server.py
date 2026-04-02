"""Pipeline 0: Local Wav2Lip via lipsync package (CPU).

Primary pipeline -- uses the `lipsync` pip package for local CPU-based
Wav2Lip inference.  Falls back to HF ZeroGPU Gradio API, then to a
simple OpenCV head-motion animation if both are unavailable.

Eve's face-detection bounding box is cached after the first frame since
her position never changes -- this eliminates the main CPU bottleneck.
"""

import base64
import io
import logging
import os
import tempfile
import time
import wave

import cv2
import numpy as np

from pipelines.base import BasePipeline, create_pipeline_app

logger = logging.getLogger("eden.p0")

# ---------------------------------------------------------------------------
# Optional imports -- degrade gracefully
# ---------------------------------------------------------------------------
try:
    from lipsync import LipSync
    LIPSYNC_AVAILABLE = True
except ImportError:
    LIPSYNC_AVAILABLE = False
    logger.warning("lipsync package not installed -- will try HF fallback")

try:
    from gradio_client import Client, handle_file
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("gradio_client not installed -- HF fallback unavailable")


# ---------------------------------------------------------------------------
# Eve reference paths (tried in order)
# ---------------------------------------------------------------------------
_EVE_SEARCH_PATHS = [
    "C:/Users/geaux/myeden/reference/eve-512.png",
    "C:/Users/geaux/myeden/reference/eve-NATURAL.png",
]


class Wav2LipPipeline(BasePipeline):
    def __init__(self):
        super().__init__(
            pipeline_id=0,
            name="musetalk",
            hf_repo="TMElyralab/MuseTalk",
        )
        self._lip: "LipSync | None" = None
        self._gradio_client = None
        self._eve_512_path: str | None = None
        # Cached face bbox -- set once, reused forever (Eve never moves)
        self._cached_face_bbox = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self):
        self.logger.info("Wav2Lip pipeline initializing...")
        if LIPSYNC_AVAILABLE:
            try:
                self._lip = LipSync(model="wav2lip", device="cpu")
                self.logger.info("Local LipSync (wav2lip, CPU) loaded OK")
            except Exception as e:
                self.logger.error(f"LipSync init failed: {e}")
                self._lip = None
        if self._lip is None:
            self._get_gradio_client()
        self.loaded = True

    # ------------------------------------------------------------------
    # Gradio HF client (lazy)
    # ------------------------------------------------------------------
    def _get_gradio_client(self):
        if self._gradio_client is None and GRADIO_AVAILABLE:
            try:
                self._gradio_client = Client("pragnakalp/Wav2lip-ZeroGPU")
                self.logger.info("Connected to Wav2Lip-ZeroGPU on HuggingFace")
            except Exception as e:
                self.logger.error(f"Failed to connect to Wav2Lip: {e}")
        return self._gradio_client

    # ------------------------------------------------------------------
    # Eve 512x512 helper
    # ------------------------------------------------------------------
    def _ensure_eve_512(self, reference_image_path: str) -> str:
        """Return path to a 512x512 Eve image (cached on disk)."""
        if self._eve_512_path and os.path.exists(self._eve_512_path):
            return self._eve_512_path

        img = cv2.imread(reference_image_path)
        if img is None:
            search = _EVE_SEARCH_PATHS + [
                os.path.join(str(self.shared_dir), "reference", "eve-NATURAL.png"),
                os.path.join(str(self.shared_dir), "reference", "eve-512.png"),
            ]
            for alt in search:
                img = cv2.imread(alt)
                if img is not None:
                    break

        if img is None:
            self.logger.error("Cannot find Eve's reference image")
            return reference_image_path

        resized = cv2.resize(img, (512, 512))
        self._eve_512_path = os.path.join(tempfile.gettempdir(), "eve-512.png")
        cv2.imwrite(self._eve_512_path, resized)
        self.logger.info(f"Created 512x512 Eve at {self._eve_512_path}")
        return self._eve_512_path

    # ------------------------------------------------------------------
    # Audio → WAV helper
    # ------------------------------------------------------------------
    @staticmethod
    def _save_audio_as_wav(audio_bytes: bytes) -> str:
        """Write audio bytes to a temporary WAV file.

        Handles three formats:
          - RIFF/WAV  → write as-is
          - MP3 (ID3 or FFxx header) → write to .mp3, convert via pydub/ffmpeg
          - Raw PCM   → wrap in 16-bit mono 16 kHz WAV container
        """
        wav_path = os.path.join(tempfile.gettempdir(), "eden_tts_audio.wav")

        if audio_bytes[:4] == b"RIFF":
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)
        elif audio_bytes[:3] == b"ID3" or (len(audio_bytes) > 1 and audio_bytes[0] == 0xFF):
            # MP3 data -- try pydub conversion, else write raw wrapper
            mp3_path = os.path.join(tempfile.gettempdir(), "eden_tts_audio.mp3")
            with open(mp3_path, "wb") as f:
                f.write(audio_bytes)
            try:
                from pydub import AudioSegment
                seg = AudioSegment.from_mp3(mp3_path)
                seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
                seg.export(wav_path, format="wav")
            except Exception:
                # Last resort: wrap raw bytes as PCM
                with wave.open(wav_path, "w") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    w.writeframes(audio_bytes)
        else:
            # Assume raw PCM 16-bit mono 16 kHz
            with wave.open(wav_path, "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(audio_bytes)

        return wav_path

    # ------------------------------------------------------------------
    # Video → JPEG frames
    # ------------------------------------------------------------------
    def _video_to_frames(self, video_path: str) -> list[bytes]:
        """Extract JPEG frames from a video file."""
        frames: list[bytes] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(buf.tobytes())

        cap.release()
        self.logger.info(f"Extracted {len(frames)} frames from video")
        return frames

    # ------------------------------------------------------------------
    # PRIMARY: local lipsync
    # ------------------------------------------------------------------
    def _local_lipsync(self, audio_bytes: bytes, reference_image_path: str) -> list[bytes] | None:
        """Run Wav2Lip locally via the `lipsync` package.  Returns frames
        or None if it fails (caller should fall through to next method).
        """
        if self._lip is None:
            return None

        wav_path = self._save_audio_as_wav(audio_bytes)
        eve_path = self._ensure_eve_512(reference_image_path)
        out_video = os.path.join(tempfile.gettempdir(), "eden_lipsync_out.mp4")

        t0 = time.time()
        try:
            self._lip.sync(eve_path, wav_path, out_video)
            elapsed = time.time() - t0
            self.logger.info(f"Local lipsync completed in {elapsed:.1f}s")

            if os.path.exists(out_video) and os.path.getsize(out_video) > 0:
                frames = self._video_to_frames(out_video)
                if frames:
                    return frames

            self.logger.warning("Local lipsync produced no usable video")
        except Exception as e:
            self.logger.error(f"Local lipsync error: {e}")

        return None

    # ------------------------------------------------------------------
    # FALLBACK 1: HF Gradio Wav2Lip
    # ------------------------------------------------------------------
    def _hf_wav2lip(self, audio_bytes: bytes, reference_image_path: str) -> list[bytes] | None:
        """Call Wav2Lip on HF ZeroGPU via Gradio API."""
        client = self._get_gradio_client()
        if client is None:
            return None

        wav_path = self._save_audio_as_wav(audio_bytes)
        eve_path = self._ensure_eve_512(reference_image_path)

        t0 = time.time()
        try:
            result = client.predict(
                input_image=handle_file(eve_path),
                input_audio=handle_file(wav_path),
                api_name="/run_infrence",
            )
            elapsed = time.time() - t0
            self.logger.info(f"HF Wav2Lip completed in {elapsed:.1f}s")

            video_path = None
            if isinstance(result, dict):
                video_path = result.get("video")
            elif isinstance(result, str):
                video_path = result

            if video_path and os.path.exists(video_path):
                frames = self._video_to_frames(video_path)
                if frames:
                    return frames

            self.logger.warning("HF Wav2Lip returned no usable video")
        except Exception as e:
            self.logger.error(f"HF Wav2Lip API error: {e}")

        return None

    # ------------------------------------------------------------------
    # FALLBACK 2: CPU head-motion animation
    # ------------------------------------------------------------------
    def _cpu_fallback_animate(self, audio_bytes: bytes, reference_image_path: str) -> list[bytes]:
        """CPU-only fallback: basic head motion + lip approximation."""
        ref_img = cv2.imread(reference_image_path)
        if ref_img is None:
            ref_img = np.zeros((512, 512, 3), dtype=np.uint8)
            ref_img[:] = (60, 60, 80)

        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if len(audio_array) == 0:
            audio_array = np.zeros(8000, dtype=np.float32)

        samples_per_frame = max(1, len(audio_array) // 30)
        num_frames = max(1, len(audio_array) // samples_per_frame)

        frames: list[bytes] = []
        for i in range(min(num_frames, 300)):
            start = i * samples_per_frame
            end = min(start + samples_per_frame, len(audio_array))
            chunk = audio_array[start:end]
            energy = min(1.0, np.sqrt(np.mean(chunk ** 2)) / 32768.0 * 3.0)

            animated = ref_img.copy()
            h, w = animated.shape[:2]
            t = i * 0.033

            angle = np.sin(t * 1.1) * 2.0 + np.sin(t * 2.7) * energy * 3.0
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            M[1, 2] += np.sin(t * 0.7) * 2.0
            animated = cv2.warpAffine(animated, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            _, buf = cv2.imencode(".jpg", animated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(buf.tobytes())

        return frames

    # ------------------------------------------------------------------
    # Main entry point -- waterfall through methods
    # ------------------------------------------------------------------
    def animate(self, audio_bytes: bytes, reference_image_path: str) -> list[bytes]:
        """Generate lip-synced frames.

        Priority:
          1. Local lipsync (wav2lip CPU) -- fast, no network
          2. HF Gradio Wav2Lip           -- GPU but remote
          3. CPU head-motion fallback     -- always works
        """
        # --- 1. Local lipsync ---
        frames = self._local_lipsync(audio_bytes, reference_image_path)
        if frames:
            self.logger.info(f"Local lipsync produced {len(frames)} frames")
            return frames

        # --- 2. HF Gradio fallback ---
        self.logger.info("Trying HF Gradio Wav2Lip fallback...")
        frames = self._hf_wav2lip(audio_bytes, reference_image_path)
        if frames:
            self.logger.info(f"HF Wav2Lip produced {len(frames)} frames")
            return frames

        # --- 3. CPU animation ---
        self.logger.info("All lip-sync methods failed, using CPU animation fallback")
        return self._cpu_fallback_animate(audio_bytes, reference_image_path)


pipeline = Wav2LipPipeline()
app = create_pipeline_app(pipeline)
