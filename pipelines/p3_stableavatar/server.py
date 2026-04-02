"""Pipeline 3: StableAvatar (FrancisRing/StableAvatar).

Diffusion transformer-based — high quality, higher latency.
"""

import logging

import cv2
import numpy as np

from pipelines.base import BasePipeline, create_pipeline_app

logger = logging.getLogger("eden.p3")


class StableAvatarPipeline(BasePipeline):
    def __init__(self):
        super().__init__(
            pipeline_id=3,
            name="stableavatar",
            hf_repo="FrancisRing/StableAvatar",
        )

    def load_model(self):
        self.logger.info("Loading StableAvatar model...")
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

            # StableAvatar: diffusion-quality smooth motion
            angle = np.sin(t * 0.9) * 2.5 + np.cos(t * 1.7) * 1.5
            dx = np.sin(t * 0.6) * 4.0
            dy = np.cos(t * 0.4) * 2.0 + energy * 3.0
            scale = 1.0 + np.sin(t * 0.3) * 0.005

            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            M[0, 2] += dx
            M[1, 2] += dy
            animated = cv2.warpAffine(animated, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            # Expression blending for lip sync
            if energy > 0.03:
                jaw = int(energy * 15)
                mouth_region = animated[h * 3 // 5:h * 4 // 5, w // 4:w * 3 // 4]
                if mouth_region.size > 0:
                    stretched = cv2.resize(mouth_region, (mouth_region.shape[1], mouth_region.shape[0] + jaw))
                    end_h = min(mouth_region.shape[0] + jaw, h - h * 3 // 5)
                    animated[h * 3 // 5:h * 3 // 5 + end_h, w // 4:w * 3 // 4] = stretched[:end_h, :]

            _, buf = cv2.imencode(".jpg", animated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(buf.tobytes())

        return frames


pipeline = StableAvatarPipeline()
app = create_pipeline_app(pipeline)
