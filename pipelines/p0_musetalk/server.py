"""Pipeline 0: Wav2Lip via HuggingFace ZeroGPU.

Primary pipeline — uses HF's free GPU via Gradio API.
Takes Eve's face + audio → returns lip-synced video frames.
Falls back to CPU OpenCV animation if HF is unavailable.
"""

import base64
import io
import logging
import os
import tempfile
import time

import cv2
import numpy as np

from pipelines.base import BasePipeline, create_pipeline_app

logger = logging.getLogger("eden.p0")

try:
    from gradio_client import Client, handle_file
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("gradio_client not installed — falling back to CPU animation")


class Wav2LipPipeline(BasePipeline):
    def __init__(self):
        super().__init__(
            pipeline_id=0,
            name="musetalk",
            hf_repo="TMElyralab/MuseTalk",
        )
        self._gradio_client = None
        self._eve_512_path = None

    def _get_gradio_client(self):
        if self._gradio_client is None and GRADIO_AVAILABLE:
            try:
                self._gradio_client = Client("pragnakalp/Wav2lip-ZeroGPU")
                self.logger.info("Connected to Wav2Lip-ZeroGPU on HuggingFace")
            except Exception as e:
                self.logger.error(f"Failed to connect to Wav2Lip: {e}")
        return self._gradio_client

    def _ensure_eve_512(self, reference_image_path: str) -> str:
        """Ensure we have a 512x512 version of Eve for Wav2Lip."""
        if self._eve_512_path and os.path.exists(self._eve_512_path):
            return self._eve_512_path

        img = cv2.imread(reference_image_path)
        if img is None:
            # Try alternate paths
            for alt in [
                os.path.join(self.shared_dir, "reference", "eve-NATURAL.png"),
                os.path.join(self.shared_dir, "eve-NATURAL.png"),
                "C:/Users/geaux/myeden/reference/eve-NATURAL.png",
                "C:/Users/geaux/myeden/reference/eve-512.png",
            ]:
                img = cv2.imread(alt)
                if img is not None:
                    break

        if img is None:
            self.logger.error("Cannot find Eve's reference image")
            return reference_image_path

        # Resize to 512x512 for Wav2Lip
        resized = cv2.resize(img, (512, 512))
        self._eve_512_path = os.path.join(tempfile.gettempdir(), "eve-512.png")
        cv2.imwrite(self._eve_512_path, resized)
        self.logger.info(f"Created 512x512 Eve at {self._eve_512_path}")
        return self._eve_512_path

    def load_model(self):
        self.logger.info("Wav2Lip pipeline initializing...")
        # Pre-connect to HF
        self._get_gradio_client()
        self.loaded = True

    def _video_to_frames(self, video_path: str) -> list[bytes]:
        """Extract JPEG frames from a video file."""
        frames = []
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

        frames = []
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

    def animate(self, audio_bytes: bytes, reference_image_path: str) -> list[bytes]:
        """Generate lip-synced frames via Wav2Lip HF API."""
        client = self._get_gradio_client()

        if client is None:
            self.logger.warning("Wav2Lip unavailable, using CPU fallback")
            return self._cpu_fallback_animate(audio_bytes, reference_image_path)

        # Save audio to temp WAV file
        import wave
        wav_path = os.path.join(tempfile.gettempdir(), "eden_tts_audio.wav")

        # Detect if audio is already WAV or raw PCM
        if audio_bytes[:4] == b'RIFF':
            # Already a WAV file
            with open(wav_path, 'wb') as f:
                f.write(audio_bytes)
        else:
            # Raw PCM — wrap in WAV
            with wave.open(wav_path, 'w') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(audio_bytes)

        # Get 512x512 Eve
        eve_path = self._ensure_eve_512(reference_image_path)

        # Call Wav2Lip on HF ZeroGPU
        t0 = time.time()
        try:
            result = client.predict(
                input_image=handle_file(eve_path),
                input_audio=handle_file(wav_path),
                api_name="/run_infrence",
            )
            elapsed = time.time() - t0
            self.logger.info(f"Wav2Lip completed in {elapsed:.1f}s")

            # Extract frames from returned video
            video_path = None
            if isinstance(result, dict):
                video_path = result.get("video")
            elif isinstance(result, str):
                video_path = result

            if video_path and os.path.exists(video_path):
                frames = self._video_to_frames(video_path)
                if frames:
                    return frames

            self.logger.warning("Wav2Lip returned no usable video")
        except Exception as e:
            self.logger.error(f"Wav2Lip API error: {e}")

        # Fall back to CPU animation
        self.logger.info("Falling back to CPU animation")
        return self._cpu_fallback_animate(audio_bytes, reference_image_path)


pipeline = Wav2LipPipeline()
app = create_pipeline_app(pipeline)
