"""EDEN OS V2 — Grok-Powered Intelligent Pipeline Agents.

Four specialized agents + AgentManager coordinator.
Primary: xAI Grok API for fast intelligent pipeline switching.
Fallback: Static SIZE_ORDER routing with zero API calls.

All agents are optional — if XAI_API_KEY not set, falls back to static routing.
Uses Grok-4-fast (non-reasoning) for sub-second decisions with 2s timeout, 10s caching.
"""

import asyncio
import json
import logging
import os
import time

from openai import AsyncOpenAI

from .metrics import MetricsStore, SIZE_ORDER, PIPELINE_NAMES, PIPELINE_SIZES_GB

logger = logging.getLogger("eden.agents")

_xai_key = os.environ.get("XAI_API_KEY", "")
AGENTS_ENABLED = bool(_xai_key)


class BaseAgent:
    """Base class for all Grok-powered agents."""

    def __init__(self, name: str):
        self.name = name
        self.enabled = AGENTS_ENABLED
        self._client: AsyncOpenAI | None = None
        self._cache: dict[str, tuple[float, any]] = {}
        self._cache_ttl = 10.0  # seconds

    def _get_client(self) -> AsyncOpenAI | None:
        if self._client is None and self.enabled:
            self._client = AsyncOpenAI(
                api_key=_xai_key,
                base_url="https://api.x.ai/v1",
            )
        return self._client

    def _get_cached(self, key: str):
        if key in self._cache:
            ts, val = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return val
        return None

    def _set_cached(self, key: str, val):
        self._cache[key] = (time.time(), val)

    async def _ask_grok(self, system: str, prompt: str, max_tokens: int = 200) -> str:
        """Send a message to Grok-4-fast with 2s timeout. Returns empty string on failure."""
        client = self._get_client()
        if not client:
            return ""
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model="grok-4-fast-non-reasoning",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,
                ),
                timeout=2.0,
            )
            return resp.choices[0].message.content
        except asyncio.TimeoutError:
            logger.warning(f"Agent {self.name}: Grok timeout (2s)")
            return ""
        except Exception as e:
            logger.warning(f"Agent {self.name}: Grok error: {e}")
            return ""

    def _parse_json(self, text: str) -> dict | list | None:
        """Safely parse JSON from Grok response, handling markdown fences."""
        if not text:
            return None
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return None


class PipelineSpeedAgent(BaseAgent):
    """Monitors response times, dynamically reranks pipelines for fastest switching."""

    def __init__(self):
        super().__init__("speed")

    async def evaluate(self, metrics_store: MetricsStore) -> list[int]:
        """Return recommended pipeline order based on speed + success rate."""
        cached = self._get_cached("order")
        if cached is not None:
            return cached

        if not self.enabled:
            return list(SIZE_ORDER)

        all_metrics = metrics_store.get_all_metrics()

        # Don't call Grok if we have insufficient data
        total_calls = sum(all_metrics[pid].total_calls for pid in range(5))
        if total_calls < 5:
            return list(SIZE_ORDER)

        summary_lines = []
        for pid in SIZE_ORDER:
            m = all_metrics[pid]
            summary_lines.append(
                f"P{pid} ({m.name}): size={m.model_size_gb}GB, "
                f"avg_response={m.avg_response_time_ms:.0f}ms, "
                f"success_rate={m.success_rate:.1%}, "
                f"avg_motion={m.avg_motion_score:.3f}, "
                f"calls={m.total_calls}"
            )

        result = await self._ask_grok(
            system="You are an AI pipeline optimizer. Return ONLY a JSON array of integer pipeline IDs. No explanation.",
            prompt=(
                "Given these pipeline performance metrics:\n"
                + "\n".join(summary_lines)
                + "\n\nReturn the optimal pipeline order as a JSON array. "
                "Prioritize: 1) fastest response time, 2) highest success rate, "
                "3) smallest model size as tiebreaker."
            ),
        )

        parsed = self._parse_json(result)
        if isinstance(parsed, list) and len(parsed) == 5 and all(isinstance(x, int) for x in parsed):
            self._set_cached("order", parsed)
            logger.info(f"Speed agent recommended order: {parsed}")
            return parsed

        return list(SIZE_ORDER)


class QualityIntelligenceAgent(BaseAgent):
    """Analyzes motion score trends, predicts best pipeline for current conditions."""

    def __init__(self):
        super().__init__("quality")

    async def evaluate(self, metrics_store: MetricsStore) -> dict:
        """Return best pipeline recommendation."""
        cached = self._get_cached("best")
        if cached is not None:
            return cached

        all_metrics = metrics_store.get_all_metrics()

        if not self.enabled:
            best_pid = max(
                SIZE_ORDER,
                key=lambda pid: all_metrics[pid].avg_motion_score if all_metrics[pid].total_calls > 0 else 0.0,
            )
            return {"best_pipeline": best_pid, "confidence": 0.5, "reason": "static_fallback"}

        total_calls = sum(all_metrics[pid].total_calls for pid in range(5))
        if total_calls < 5:
            return {"best_pipeline": SIZE_ORDER[0], "confidence": 0.3, "reason": "insufficient_data"}

        summary_lines = []
        for pid in range(5):
            m = all_metrics[pid]
            summary_lines.append(
                f"P{pid} ({m.name}): avg_motion={m.avg_motion_score:.3f}, "
                f"trend={m.trend}, calls={m.total_calls}, "
                f"success_rate={m.success_rate:.1%}"
            )

        result = await self._ask_grok(
            system="You are an animation quality analyst. Return ONLY JSON, no explanation.",
            prompt=(
                "Pipeline motion quality metrics:\n"
                + "\n".join(summary_lines)
                + '\n\nWhich pipeline produces the best animation? '
                'Return: {"best_pipeline": <int>, "confidence": <0-1>, "reason": "<brief>"}'
            ),
        )

        parsed = self._parse_json(result)
        if isinstance(parsed, dict) and "best_pipeline" in parsed:
            self._set_cached("best", parsed)
            logger.info(f"Quality agent: best=P{parsed['best_pipeline']}, reason={parsed.get('reason')}")
            return parsed

        return {"best_pipeline": SIZE_ORDER[0], "confidence": 0.3, "reason": "agent_error"}


class FailoverDecisionAgent(BaseAgent):
    """Makes intelligent failover decisions instead of hard thresholds."""

    def __init__(self):
        super().__init__("failover")

    async def evaluate(
        self,
        motion_score: float,
        pipeline_id: int,
        consecutive_bad: int,
        metrics_store: MetricsStore,
    ) -> dict:
        """Decide whether to failover — smarter than a dumb threshold."""
        # Hard failover: always trigger if completely static for 2+ checks
        if motion_score == 0.0 and consecutive_bad >= 2:
            return {"should_failover": True, "reason": "completely_static"}

        if not self.enabled:
            return {"should_failover": motion_score < 0.05, "reason": "threshold_fallback"}

        m = metrics_store.get_metrics(pipeline_id)
        if not m:
            return {"should_failover": motion_score < 0.05, "reason": "no_metrics"}

        result = await self._ask_grok(
            system="You are a real-time systems reliability engineer. Minimize unnecessary failovers. Return ONLY JSON.",
            prompt=(
                f"Pipeline P{pipeline_id} ({PIPELINE_NAMES.get(pipeline_id, '?')}) status:\n"
                f"- Current motion score: {motion_score:.3f} (threshold: 0.05)\n"
                f"- Consecutive bad checks: {consecutive_bad}\n"
                f"- Average motion score: {m.avg_motion_score:.3f}\n"
                f"- Trend: {m.trend}\n"
                f"- Success rate: {m.success_rate:.1%}\n"
                f"- Total calls: {m.total_calls}\n\n"
                "Should we failover? Consider: is it warming up? Is the trend improving? "
                "Is unnecessary switching costly?\n"
                'Return: {"should_failover": true/false, "reason": "<brief>"}'
            ),
        )

        parsed = self._parse_json(result)
        if isinstance(parsed, dict) and "should_failover" in parsed:
            logger.info(
                f"Failover agent: P{pipeline_id} motion={motion_score:.3f} → "
                f"failover={parsed['should_failover']}, reason={parsed.get('reason')}"
            )
            return parsed

        # Fallback to threshold
        return {"should_failover": motion_score < 0.05, "reason": "agent_fallback"}


class PreemptiveWarmupAgent(BaseAgent):
    """Predicts which pipeline will be needed next and recommends pre-warming."""

    def __init__(self):
        super().__init__("warmup")

    async def evaluate(self, current_pipeline_id: int, metrics_store: MetricsStore) -> dict:
        """Recommend which pipeline to pre-warm."""
        idx = SIZE_ORDER.index(current_pipeline_id) if current_pipeline_id in SIZE_ORDER else 0
        next_pid = SIZE_ORDER[idx + 1] if idx + 1 < len(SIZE_ORDER) else None

        if not self.enabled:
            return {"warmup_pipeline": next_pid, "reason": "static_next_in_order"}

        m = metrics_store.get_metrics(current_pipeline_id)
        if not m:
            return {"warmup_pipeline": next_pid, "reason": "no_metrics"}

        result = await self._ask_grok(
            system="You are a predictive infrastructure manager. Return ONLY JSON.",
            prompt=(
                f"Current pipeline: P{current_pipeline_id} ({PIPELINE_NAMES.get(current_pipeline_id)})\n"
                f"- Trend: {m.trend}, Success rate: {m.success_rate:.1%}, Avg motion: {m.avg_motion_score:.3f}\n"
                f"Available (by size): {[f'P{pid} ({PIPELINE_SIZES_GB[pid]}GB)' for pid in SIZE_ORDER]}\n"
                'Which to pre-warm? Return: {"warmup_pipeline": <int or null>, "reason": "<brief>"}'
            ),
        )

        parsed = self._parse_json(result)
        if isinstance(parsed, dict) and "warmup_pipeline" in parsed:
            logger.info(f"Warmup agent: recommend P{parsed['warmup_pipeline']}, reason={parsed.get('reason')}")
            return parsed

        return {"warmup_pipeline": next_pid, "reason": "agent_fallback"}


class AgentManager:
    """Coordinator for all pipeline intelligence agents.

    Primary: Grok-4-fast via xAI API (fast, cheap, already paid for)
    Fallback: Static SIZE_ORDER routing (zero API calls)
    """

    def __init__(self, metrics_store: MetricsStore):
        self.metrics_store = metrics_store
        self.speed_agent = PipelineSpeedAgent()
        self.quality_agent = QualityIntelligenceAgent()
        self.failover_agent = FailoverDecisionAgent()
        self.warmup_agent = PreemptiveWarmupAgent()
        self.enabled = AGENTS_ENABLED
        logger.info(f"AgentManager initialized. Grok agents enabled: {self.enabled}")

    async def get_routing_order(self, force_strong: bool = False) -> list[int]:
        """Get agent-recommended pipeline order."""
        if force_strong:
            return list(reversed(SIZE_ORDER))
        return await self.speed_agent.evaluate(self.metrics_store)

    async def should_failover(self, motion_score: float, pipeline_id: int, consecutive_bad: int = 0) -> bool:
        """Agent-enhanced failover decision."""
        result = await self.failover_agent.evaluate(
            motion_score=motion_score,
            pipeline_id=pipeline_id,
            consecutive_bad=consecutive_bad,
            metrics_store=self.metrics_store,
        )
        return result.get("should_failover", motion_score < 0.05)

    async def recommend_warmup(self, current_pipeline_id: int) -> int | None:
        """Get warmup recommendation."""
        result = await self.warmup_agent.evaluate(current_pipeline_id, self.metrics_store)
        return result.get("warmup_pipeline")

    async def get_quality_recommendation(self) -> dict:
        """Get quality agent's pipeline recommendation."""
        return await self.quality_agent.evaluate(self.metrics_store)

    def status(self) -> dict:
        return {
            "agents_enabled": self.enabled,
            "engine": "grok-4-fast" if self.enabled else "static",
            "xai_key_set": bool(_xai_key),
            "agents": {
                "speed": self.speed_agent.enabled,
                "quality": self.quality_agent.enabled,
                "failover": self.failover_agent.enabled,
                "warmup": self.warmup_agent.enabled,
            },
        }
