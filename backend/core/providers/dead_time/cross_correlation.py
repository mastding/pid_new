"""Cross-correlation dead-time provider."""
from __future__ import annotations

from core.algorithms.system_id import _estimate_dead_time
from core.providers.dead_time.base import BaseDeadTimeProvider
from core.shared import register_provider


@register_provider("dead_time")
class CrossCorrelationDeadTimeProvider(BaseDeadTimeProvider):
    name = "cross_correlation"

    def estimate(self, *, mv, pv, dt: float, loop_type: str, context=None) -> dict[str, object]:
        dead_time = float(_estimate_dead_time(mv, pv, dt))
        return {
            "provider": self.name,
            "L": dead_time,
            "confidence": 1.0 if dead_time > 0 else 0.4,
            "meta": {"loop_type": loop_type},
        }
