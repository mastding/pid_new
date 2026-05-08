"""History-rule-based window detection provider."""
from __future__ import annotations

from core.algorithms.data_analysis import build_candidate_windows
from core.pipeline.window_algorithm_family import window_algorithm_family
from core.providers.window.base import BaseWindowDetectionProvider
from core.shared import register_provider


def _apply_algorithm_plan(candidate_windows: list[dict], policy: dict | None) -> list[dict]:
    if not policy:
        return candidate_windows
    plan = {
        str(item.get("family")): item
        for item in policy.get("algorithm_plan", [])
        if isinstance(item, dict) and item.get("family")
    }
    if not plan:
        return candidate_windows

    annotated: list[dict] = []
    for window in candidate_windows:
        updated = dict(window)
        family = window_algorithm_family(updated)
        item = plan.get(family)
        updated["window_algorithm_family"] = family
        if item:
            updated["window_algorithm_plan_state"] = item.get("state", "available")
            updated["window_algorithm_plan_reason"] = item.get("reason", "")
        else:
            updated["window_algorithm_plan_state"] = "available"
            updated["window_algorithm_plan_reason"] = "允许参与候选，但不作为本体策略优先推荐"
        annotated.append(updated)
    return annotated


@register_provider("window_detection")
class HistoryRuleBasedWindowProvider(BaseWindowDetectionProvider):
    name = "history_rule_based"

    def detect(self, *, df, dt: float, loop_type: str, context=None) -> dict[str, object]:
        policy = context.get("policy") if isinstance(context, dict) else None
        candidate_windows, step_events = build_candidate_windows(
            df,
            dt,
            loop_type=loop_type,
            policy=policy if isinstance(policy, dict) else None,
        )
        candidate_windows = _apply_algorithm_plan(candidate_windows, policy if isinstance(policy, dict) else None)
        return {
            "provider": self.name,
            "candidate_windows": candidate_windows,
            "step_events": step_events,
            "meta": {
                "loop_type": loop_type,
                "candidate_count": len(candidate_windows),
                "step_event_count": len(step_events),
                "policy_applied": bool(policy),
            },
        }
