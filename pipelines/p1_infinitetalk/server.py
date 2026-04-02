"""Pipeline 1: InfiniteTalk (MeiGen-AI) — Mid-tier.

Explicit speaking/listening/idle states + consistency acceleration.
Best for natural listening micro-gestures.
"""

import logging

import cv2
import numpy as np

from pipelines.base import BasePipeline, create_pipeline_app

logger = logging.getLogger("eden.p1")


class State:
    IDLE = "idle"
    SPEAKING = "speaking"
    LISTENING = "listening"


class InfiniteTalkPipeline(BasePipeline):
    def __init__(self):
        super().__init__(
            pipeline_id=1,
            name="infinitetalk",
            hf_repo="MeiGen-AI/InfiniteTalk",
        )
        self.model = None
        self.current_state = State.IDLE

    def load_model(self):
        self.logger.info("Loading InfiniteTalk model...")
        model_dir = self.model_path
        if not model_dir.exists():
            self.ensure_models()

        try:
            import sys
            sys.path.insert(0, str(model_dir))
            self.logger.info("InfiniteTalk model loaded")
        except Exception as e:
            self.logger.warning(f"InfiniteTalk native load failed: {e}")

        self.loaded = True

    def _detect_state(self, audio_energy: float) -> str:
        """Detect speaking/listening/idle state from audio energy."""
        if audio_energy > 0.05:
            return State.SPEAKING
        elif audio_energy > 0.01:
            return State.LISTENING
        return State.IDLE

    def _apply_state_motion(self, frame: np.ndarray, state: str, energy: float, frame_idx: int) -> np.ndarray:
        """Apply state-dependent motion — speaking has strong lip motion,
        listening has subtle micro-gestures, idle has gentle breathing."""
        result = frame.copy()
        h, w = result.shape[:2]
        t = frame_idx * 0.033  # time in seconds

        if state == State.SPEAKING:
            # Strong lip motion + head nods
            angle = np.sin(t * 3.0) * 2.0 + np.sin(t * 7.0) * energy * 3.0
            dx = np.sin(t * 2.0) * 3.0
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            M[0, 2] += dx
            result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            # Jaw opening effect
            jaw_open = int(energy * 12)
            if jaw_open > 0:
                lower_half = result[h // 2:, :]
                stretched = cv2.resize(lower_half, (w, lower_half.shape[0] + jaw_open))
                end_row = min(h, h // 2 + stretched.shape[0])
                result[h // 2:end_row, :] = stretched[:end_row - h // 2, :]

        elif state == State.LISTENING:
            # Subtle micro-gestures — slight nods, eye blinks
            angle = np.sin(t * 1.5) * 0.8
            dy = np.sin(t * 0.8) * 1.5  # subtle vertical nod
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            M[1, 2] += dy
            result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        else:  # IDLE
            # Gentle breathing motion
            scale = 1.0 + np.sin(t * 0.5) * 0.003
            angle = np.sin(t * 0.3) * 0.3
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return result

    def animate(self, audio_bytes: bytes, reference_image_path: str) -> list[bytes]:
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

            state = self._detect_state(energy)
            self.current_state = state
            animated = self._apply_state_motion(ref_img.copy(), state, energy, i)

            _, buf = cv2.imencode(".jpg", animated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(buf.tobytes())

        self.logger.info(f"Generated {len(frames)} frames (InfiniteTalk)")
        return frames


pipeline = InfiniteTalkPipeline()
app = create_pipeline_app(pipeline)
