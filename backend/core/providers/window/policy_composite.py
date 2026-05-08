"""Policy-aware composite window detection provider."""
from __future__ import annotations

from typing import Any

from core.algorithms.data_analysis import _merge_events
from core.pipeline.window_algorithm_family import window_algorithm_family
from core.providers.window.event_window_builder import build_windows_from_events
from core.providers.window.history_rule_based import _apply_algorithm_plan
from core.providers.window.base import BaseWindowDetectionProvider
from core.shared import provider_registry, register_provider


_FAMILY_ORDER = ["sp_step", "mv_step", "mv_ramp", "steady_disturbance", "rolling_scan"]


def _policy_family_sets(policy: dict[str, Any] | None) -> tuple[set[str], set[str]]:
    if not policy:
        return set(), set()
    disabled = {str(item) for item in policy.get("disabled_algorithm_families") or []}
    runnable = {
        str(item.get("family"))
        for item in policy.get("algorithm_plan", [])
        if isinstance(item, dict) and item.get("state") != "disabled" and item.get("family")
    }
    return runnable, disabled


def _policy_plan_by_family(policy: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not policy:
        return {}
    return {
        str(item.get("family")): item
        for item in policy.get("algorithm_plan", [])
        if isinstance(item, dict) and item.get("family")
    }


def _policy_merge_points(policy: dict[str, Any] | None, dt: float) -> int:
    if isinstance(policy, dict):
        try:
            merge_gap_s = float(policy.get("merge_gap_s") or 180.0)
        except (TypeError, ValueError):
            merge_gap_s = 180.0
    else:
        merge_gap_s = 180.0
    return max(40, int(merge_gap_s / max(dt, 1e-6)))


def _summarize_families(
    *,
    all_families: list[str],
    family_events: dict[str, list[dict[str, Any]]],
    candidate_windows: list[dict[str, Any]],
    runnable: set[str],
    disabled: set[str],
    policy: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    plan_by_family = _policy_plan_by_family(policy)
    summaries: list[dict[str, Any]] = []
    for family in all_families:
        windows = [w for w in candidate_windows if window_algorithm_family(w) == family]
        plan = plan_by_family.get(family, {})
        policy_state = str(plan.get("state") or ("disabled" if family in disabled else "available"))
        if family in disabled:
            run_state = "disabled"
        elif family in runnable:
            run_state = "ran"
        else:
            run_state = "skipped"
        summaries.append({
            "family": family,
            "provider": family,
            "run_state": run_state,
            "policy_state": policy_state,
            "policy_reason": plan.get("reason", ""),
            "event_count": len(family_events.get(family, [])),
            "window_count": len(windows),
            "usable_count": sum(1 for w in windows if w.get("window_usable_for_id")),
            "best_score": round(max([float(w.get("window_quality_score", 0.0)) for w in windows] or [0.0]), 4),
        })
    return summaries


@register_provider("window_detection")
class PolicyCompositeWindowProvider(BaseWindowDetectionProvider):
    """Run existing window detection, then expose policy-driven family gating.

    This is intentionally a thin Phase-3 bridge: algorithm math still lives in
    the mature historical implementation, while orchestration can now reason in
    terms of algorithm families. Later each family can become its own provider.
    """

    name = "policy_composite"

    def detect(self, *, df, dt: float, loop_type: str, context=None) -> dict[str, Any]:
        context = context or {}
        policy = context.get("policy") if isinstance(context, dict) else None
        runnable, disabled = _policy_family_sets(policy if isinstance(policy, dict) else None)
        if not runnable:
            runnable = {"sp_step", "mv_step", "mv_ramp", "steady_disturbance", "rolling_scan"}

        family_results: dict[str, dict[str, Any]] = {}
        family_events: dict[str, list[dict[str, Any]]] = {}
        for family in _FAMILY_ORDER:
            if family in disabled or family not in runnable:
                continue
            provider = provider_registry.get("window_algorithm_family", family)
            if provider is None:
                continue
            result = provider.detect(df=df, dt=dt, loop_type=loop_type, context=context)
            events = list(result.get("step_events", []))
            family_results[family] = result
            family_events[family] = events

        sp_events = family_events.get("sp_step", [])
        mv_step_events = family_events.get("mv_step", [])
        mv_ramp_events = family_events.get("mv_ramp", [])
        steady_events = family_events.get("steady_disturbance", [])

        merge_pts = _policy_merge_points(policy if isinstance(policy, dict) else None, dt)
        all_events = _merge_events(sp_events, mv_step_events, proximity=40)
        all_events = _merge_events(all_events, mv_ramp_events, proximity=merge_pts)
        all_events = _merge_events(all_events, steady_events, proximity=merge_pts)
        if not all_events:
            all_events = family_events.get("rolling_scan", [])

        candidate_windows = build_windows_from_events(
            df=df,
            dt=dt,
            loop_type=loop_type,
            events=all_events,
            policy=policy if isinstance(policy, dict) else None,
        )

        candidate_windows = _apply_algorithm_plan(candidate_windows, policy if isinstance(policy, dict) else None)
        if disabled:
            candidate_windows = [
                window for window in candidate_windows
                if window_algorithm_family(window) not in disabled
            ]

        family_summaries = _summarize_families(
            all_families=_FAMILY_ORDER,
            family_events=family_events,
            candidate_windows=candidate_windows,
            runnable=runnable,
            disabled=disabled,
            policy=policy if isinstance(policy, dict) else None,
        )
        meta = {
            "loop_type": loop_type,
            "step_event_count": len(all_events),
            "policy_applied": bool(policy),
            "provider_chain": list(_FAMILY_ORDER),
            "family_event_counts": {family: len(events) for family, events in family_events.items()},
            "family_summaries": family_summaries,
            "runnable_algorithm_families": sorted(runnable),
            "disabled_algorithm_families": sorted(disabled),
            "candidate_count": len(candidate_windows),
        }
        return {
            "provider": self.name,
            "candidate_windows": candidate_windows,
            "step_events": all_events,
            "family_results": family_results,
            "meta": meta,
        }
