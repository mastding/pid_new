"""Quality-score-based deterministic window selector."""
from __future__ import annotations

from typing import Any

from core.shared import register_provider


@register_provider("window_selection")
class QualityScoreWindowSelectionProvider:
    name = "quality_score_selector"

    def select(
        self,
        *,
        candidate_windows: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not candidate_windows:
            return {
                "provider": self.name,
                "chosen_index": -1,
                "score": 0.0,
                "reasoning": "无候选窗口",
                "meta": {},
            }

        chosen_index = max(
            range(len(candidate_windows)),
            key=lambda i: float(candidate_windows[i].get("window_quality_score", 0.0)),
        )
        chosen = candidate_windows[chosen_index]
        return {
            "provider": self.name,
            "chosen_index": chosen_index,
            "score": float(chosen.get("window_quality_score", 0.0)),
            "reasoning": "按 window_quality_score 最高选窗",
            "meta": {
                "source": chosen.get("window_source", ""),
                "n_points": int(chosen.get("window_end_idx", 0)) - int(chosen.get("window_start_idx", 0)),
            },
        }
