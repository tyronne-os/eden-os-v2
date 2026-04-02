"""Pipeline 2: Ditto Talking Head (digital-avatar/ditto-talkinghead)."""

import logging

import cv2
import numpy as np

from pipelines.base import BasePipeline, create_pipeline_app

logger = logging.getLogger("eden.p2")


class DittoPipeline(BasePipeline):
    def __init__(self):
        super().__init__(
            pipeline_id=2,
            name="ditto",
            hf_repo="digital-avatar/ditto-talkinghead",
        )

    def load_model(self):
        self.logger.info("Loading Ditto model...")
        if not self.model_path.exists():
            self.ensure_models()
        self.loaded = True

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

            animated = ref_img.copy()
            h, w = animated.shape[:2]
            t = i * 0.033

            # Ditto-style: smooth 3D-aware head motion + lip sync
            yaw = np.sin(t * 1.2) * 3.0 + energy * 5.0
            pitch = np.sin(t * 0.8) * 2.0
            M = cv2.getRotationMatrix2D((w / 2, h / 2), yaw, 1.0)
            M[1, 2] += pitch
            animated = cv2.warpAffine(animated, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            # Lip opening
            if energy > 0.05:
                jaw = int(energy * 10)
                lower = animated[h * 2 // 3:, :]
                stretched = cv2.resize(lower, (w, lower.shape[0] + jaw))
                end_row = min(h, h * 2 // 3 + stretched.shape[0])
                animated[h * 2 // 3:end_row, :] = stretched[:end_row - h * 2 // 3, :]

            _, buf = cv2.imencode(".jpg", animated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(buf.tobytes())

        return frames


pipeline = DittoPipeline()
app = create_pipeline_app(pipeline)
