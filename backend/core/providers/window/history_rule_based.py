"""History-rule-based window detection provider."""
from __future__ import annotations

from core.algorithms.data_analysis import build_candidate_windows
from core.providers.window.base import BaseWindowDetectionProvider
from core.shared import register_provider


@register_provider("window_detection")
class HistoryRuleBasedWindowProvider(BaseWindowDetectionProvider):
    name = "history_rule_based"

    def detect(self, *, df, dt: float, loop_type: str, context=None) -> dict[str, object]:
        candidate_windows, step_events = build_candidate_windows(df, dt)
        return {
            "provider": self.name,
            "candidate_windows": candidate_windows,
            "step_events": step_events,
            "meta": {
                "loop_type": loop_type,
                "candidate_count": len(candidate_windows),
                "step_event_count": len(step_events),
            },
        }
