"""EDEN OS V2 — Pipeline Performance Metrics.

Tracks per-pipeline response times, motion scores, success rates,
and load times in a thread-safe sliding window.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field


# Pipeline sizes in GB — determines failover order (smallest = fastest to load)
PIPELINE_SIZES_GB: dict[int, float] = {
    4: 1.26,   # LiveAvatar FP8
    0: 6.37,   # MuseTalk
    2: 6.45,   # Ditto
    3: 18.49,  # StableAvatar
    1: 85.02,  # InfiniteTalk
}

# Canonical failover order: smallest → largest
SIZE_ORDER: list[int] = [4, 0, 2, 3, 1]

PIPELINE_NAMES: dict[int, str] = {
    0: "musetalk",
    1: "infinitetalk",
    2: "ditto",
    3: "stableavatar",
    4: "liveavatar",
}


@dataclass
class PipelineMetrics:
    """Performance metrics for a single pipeline."""

    pipeline_id: int
    name: str = ""
    model_size_gb: float = 0.0
    response_times_ms: deque = field(default_factory=lambda: deque(maxlen=50))
    motion_scores: deque = field(default_factory=lambda: deque(maxlen=50))
    success_count: int = 0
    failure_count: int = 0
    total_calls: int = 0
    last_load_time_ms: float = 0.0
    estimated_memory_mb: float = 0.0
    last_updated: float = 0.0

    @property
    def avg_response_time_ms(self) -> float:
        if not self.response_times_ms:
            return 0.0
        return sum(self.response_times_ms) / len(self.response_times_ms)

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.success_count / self.total_calls

    @property
    def avg_motion_score(self) -> float:
        if not self.motion_scores:
            return 0.0
        return sum(self.motion_scores) / len(self.motion_scores)

    @property
    def trend(self) -> str:
        """Compare last 10 motion scores to previous 10."""
        scores = list(self.motion_scores)
        if len(scores) < 10:
            return "insufficient_data"
        recent = scores[-10:]
        previous = scores[-20:-10] if len(scores) >= 20 else scores[:10]
        recent_avg = sum(recent) / len(recent)
        previous_avg = sum(previous) / len(previous)
        diff = recent_avg - previous_avg
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "degrading"
        return "stable"

    def to_dict(self) -> dict:
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "model_size_gb": self.model_size_gb,
            "avg_response_time_ms": round(self.avg_response_time_ms, 1),
            "avg_motion_score": round(self.avg_motion_score, 3),
            "success_rate": round(self.success_rate, 3),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_calls": self.total_calls,
            "last_load_time_ms": round(self.last_load_time_ms, 1),
            "trend": self.trend,
            "last_updated": self.last_updated,
        }


class MetricsStore:
    """Thread-safe metrics storage for all pipelines."""

    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: dict[int, PipelineMetrics] = {}
        for pid in range(5):
            self._metrics[pid] = PipelineMetrics(
                pipeline_id=pid,
                name=PIPELINE_NAMES.get(pid, f"pipeline{pid}"),
                model_size_gb=PIPELINE_SIZES_GB.get(pid, 0.0),
            )

    def record_call(self, pid: int, response_time_ms: float, motion_score: float, success: bool):
        with self._lock:
            m = self._metrics.get(pid)
            if not m:
                return
            m.response_times_ms.append(response_time_ms)
            m.motion_scores.append(motion_score)
            m.total_calls += 1
            if success:
                m.success_count += 1
            else:
                m.failure_count += 1
            m.last_updated = time.time()

    def record_load_time(self, pid: int, ms: float):
        with self._lock:
            m = self._metrics.get(pid)
            if m:
                m.last_load_time_ms = ms
                m.estimated_memory_mb = PIPELINE_SIZES_GB.get(pid, 0) * 1024 * 0.6

    def get_metrics(self, pid: int) -> PipelineMetrics | None:
        with self._lock:
            return self._metrics.get(pid)

    def get_all_metrics(self) -> dict[int, PipelineMetrics]:
        with self._lock:
            return dict(self._metrics)

    def get_summary(self) -> dict:
        with self._lock:
            return {
                "pipelines": {pid: m.to_dict() for pid, m in self._metrics.items()},
                "size_order": SIZE_ORDER,
                "total_calls": sum(m.total_calls for m in self._metrics.values()),
            }
