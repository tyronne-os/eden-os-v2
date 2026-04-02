"""Base class for all EDEN animation pipelines.

Each pipeline implements:
  - load_model()  → lazy-load weights from HF Hub
  - animate()     → audio bytes + reference image → list of frame bytes
  - warmup()      → silent test for pre-warm scoring
"""

import abc
import base64
import logging
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("eden.pipeline")


class AnimateRequest(BaseModel):
    audio_b64: str
    reference_image: str = "eve-NATURAL.png"
    request_id: str = ""


class WarmupRequest(BaseModel):
    reference_image: str = "eve-NATURAL.png"


class BasePipeline(abc.ABC):
    """Abstract base for all EDEN pipelines."""

    def __init__(self, pipeline_id: int, name: str, hf_repo: str):
        self.pipeline_id = pipeline_id
        self.name = name
        self.hf_repo = hf_repo
        self.models_dir = Path(os.environ.get("MODELS_DIR", "/models"))
        self.shared_dir = Path(os.environ.get("SHARED_DIR", "/shared"))
        self.model_path = self.models_dir / name
        self.loaded = False
        self.logger = logging.getLogger(f"eden.pipeline.{name}")

    def ensure_models(self):
        """Download models from HF Hub if not present."""
        if self.model_path.exists() and any(self.model_path.iterdir()):
            self.logger.info(f"Models found at {self.model_path}")
            return
        self.logger.info(f"Downloading {self.hf_repo} to {self.model_path}...")
        token = os.environ.get("HF_TOKEN", None)
        snapshot_download(
            repo_id=self.hf_repo,
            local_dir=str(self.model_path),
            token=token,
            max_workers=4,
        )
        self.logger.info(f"Download complete: {self.hf_repo}")

    @abc.abstractmethod
    def load_model(self):
        """Load model weights into GPU memory."""
        ...

    @abc.abstractmethod
    def animate(self, audio_bytes: bytes, reference_image_path: str) -> list[bytes]:
        """Generate animated frames from audio + reference image.
        Returns list of JPEG-encoded frame bytes.
        """
        ...

    def warmup(self, reference_image_path: str) -> float:
        """Run a quick test animation and return motion score."""
        # Generate 0.5s of silence for warmup
        import numpy as np
        silence = np.zeros(8000, dtype=np.int16).tobytes()  # 0.5s at 16kHz
        try:
            frames = self.animate(silence, reference_image_path)
            if len(frames) < 2:
                return 0.0
            # Quick motion check
            arrays = [np.frombuffer(f, dtype=np.uint8) for f in frames[:5]]
            diffs = []
            for i in range(1, len(arrays)):
                min_len = min(len(arrays[i - 1]), len(arrays[i]))
                diff = np.mean(np.abs(arrays[i][:min_len].astype(float) - arrays[i - 1][:min_len].astype(float)))
                diffs.append(diff)
            return min(1.0, float(np.mean(diffs) / 25.0)) if diffs else 0.0
        except Exception as e:
            self.logger.warning(f"Warmup failed: {e}")
            return 0.0


def create_pipeline_app(pipeline: BasePipeline) -> FastAPI:
    """Create a FastAPI app wrapping a pipeline instance."""
    app = FastAPI(title=f"EDEN Pipeline {pipeline.pipeline_id}: {pipeline.name}")

    @app.post("/animate")
    async def animate(request: AnimateRequest):
        if not pipeline.loaded:
            try:
                pipeline.ensure_models()
                pipeline.load_model()
                pipeline.loaded = True
            except Exception as e:
                return {"frames": [], "error": str(e)}

        t0 = time.time()
        audio_bytes = base64.b64decode(request.audio_b64)
        ref_path = str(pipeline.shared_dir / "reference" / request.reference_image)

        try:
            frames = pipeline.animate(audio_bytes, ref_path)
            frames_b64 = [base64.b64encode(f).decode() for f in frames]
        except Exception as e:
            pipeline.logger.error(f"Animation error: {e}")
            frames_b64 = []

        elapsed = time.time() - t0
        return {
            "frames": frames_b64,
            "pipeline_id": pipeline.pipeline_id,
            "pipeline_name": pipeline.name,
            "frame_count": len(frames_b64),
            "elapsed_s": round(elapsed, 2),
        }

    @app.post("/warmup")
    async def warmup_endpoint(request: WarmupRequest):
        if not pipeline.loaded:
            try:
                pipeline.ensure_models()
                pipeline.load_model()
                pipeline.loaded = True
            except Exception as e:
                return {"motion_score": 0.0, "error": str(e)}

        ref_path = str(pipeline.shared_dir / "reference" / request.reference_image)
        score = pipeline.warmup(ref_path)
        return {"motion_score": score, "pipeline": pipeline.name}

    @app.get("/health")
    async def health():
        return {
            "pipeline_id": pipeline.pipeline_id,
            "name": pipeline.name,
            "loaded": pipeline.loaded,
            "hf_repo": pipeline.hf_repo,
        }

    return app
