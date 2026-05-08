"""Apply WindowSelectionPolicy to candidate-window quality scores."""
from __future__ import annotations

from typing import Any

from core.pipeline.window_algorithm_family import window_algorithm_family


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def apply_window_policy_to_candidates(
    candidate_windows: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Annotate candidates with policy score and return display summaries.

    Phase 2 uses policy as a soft/hard gate over the existing deterministic
    windows. It intentionally preserves `window_original_quality_score` so we
    can compare old ranking vs policy-adjusted ranking in the UI and events.
    """
    if not candidate_windows:
        return [], []

    preferred = set(policy.get("preferred_algorithm_families") or [])
    deprioritized = set(policy.get("deprioritized_algorithm_families") or [])
    disabled = set(policy.get("disabled_algorithm_families") or [])
    min_points = _as_int(policy.get("min_window_points"), 30)
    min_mv = policy.get("min_mv_excitation")
    min_mv_value = _as_float(min_mv) if min_mv is not None else None
    max_sat = policy.get("max_mv_saturation_ratio")
    max_sat_value = _as_float(max_sat) if max_sat is not None else None
    min_pv = policy.get("min_pv_response")
    min_pv_value = _as_float(min_pv) if min_pv is not None else None
    max_drift = policy.get("max_drift_ratio")
    max_drift_value = _as_float(max_drift) if max_drift is not None else None
    max_window_points = policy.get("max_window_points")
    max_window_points_value = _as_int(max_window_points) if max_window_points is not None else None
    policy_confidence = max(0.0, min(_as_float(policy.get("confidence"), 0.0), 1.0))
    policy_weight = 0.25 + 0.25 * policy_confidence

    annotated: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for index, window in enumerate(candidate_windows):
        updated = dict(window)
        original_score = max(0.0, min(_as_float(window.get("window_quality_score")), 1.0))
        family = window_algorithm_family(window)
        n_points = _as_int(window.get("window_end_idx")) - _as_int(window.get("window_start_idx"))
        mv_span = _as_float(window.get("window_mv_span"))
        pv_span = _as_float(window.get("window_pv_span"))
        metrics = window.get("window_quality_metrics") if isinstance(window.get("window_quality_metrics"), dict) else {}
        sat_ratio = _as_float(metrics.get("saturation_ratio") if metrics else window.get("saturation_ratio"))
        drift_ratio = _as_float(metrics.get("drift_ratio") if metrics else window.get("drift_ratio"))

        violations: list[dict[str, Any]] = []
        hard_block = False
        consistency = 1.0

        if family in disabled:
            hard_block = True
            consistency -= 0.6
            violations.append({
                "level": "hard",
                "code": "disabled_algorithm_family",
                "message": f"算法族 {family} 被本体策略禁用",
            })
        elif family in deprioritized:
            consistency -= 0.08
            violations.append({
                "level": "soft",
                "code": "deprioritized_algorithm_family",
                "message": f"算法族 {family} 被本体策略降级",
            })
        elif preferred and family not in preferred:
            consistency -= 0.05
            violations.append({
                "level": "soft",
                "code": "not_preferred_algorithm_family",
                "message": f"算法族 {family} 不在本体策略优先列表",
            })

        if n_points < min_points:
            hard_block = True
            consistency -= 0.35
            violations.append({
                "level": "hard",
                "code": "too_few_points",
                "message": f"窗口点数 {n_points} 小于最低要求 {min_points}",
            })

        if max_window_points_value is not None and n_points > max_window_points_value:
            consistency -= 0.12
            violations.append({
                "level": "soft",
                "code": "too_many_points",
                "message": f"窗口点数 {n_points} 高于策略建议上限 {max_window_points_value}",
            })

        if min_mv_value is not None and mv_span < min_mv_value:
            hard_block = True
            ratio = mv_span / max(min_mv_value, 1e-9)
            consistency -= min(0.4, 0.25 + 0.15 * (1.0 - max(0.0, ratio)))
            violations.append({
                "level": "hard",
                "code": "mv_excitation_below_policy",
                "message": f"MV 激励幅度 {mv_span:.3g} 小于策略要求 {min_mv_value:.3g}",
            })

        if min_pv_value is not None and pv_span < min_pv_value:
            hard_block = True
            ratio = pv_span / max(min_pv_value, 1e-9)
            consistency -= min(0.35, 0.2 + 0.15 * (1.0 - max(0.0, ratio)))
            violations.append({
                "level": "hard",
                "code": "pv_response_below_policy",
                "message": f"PV 响应幅度 {pv_span:.3g} 小于策略要求 {min_pv_value:.3g}",
            })

        if max_sat_value is not None and sat_ratio > max_sat_value:
            consistency -= min(0.25, (sat_ratio - max_sat_value) * 0.5)
            violations.append({
                "level": "soft",
                "code": "mv_saturation_above_policy",
                "message": f"MV 饱和/贴边比例 {sat_ratio:.2%} 高于策略建议 {max_sat_value:.2%}",
            })

        if max_drift_value is not None and drift_ratio > max_drift_value:
            consistency -= min(0.25, (drift_ratio - max_drift_value) * 0.25)
            violations.append({
                "level": "soft",
                "code": "pv_drift_above_policy",
                "message": f"PV 漂移比例 {drift_ratio:.2f} 高于策略建议 {max_drift_value:.2f}",
            })

        consistency = max(0.0, min(consistency, 1.0))
        policy_score = max(0.0, min(original_score * (1.0 - policy_weight) + consistency * policy_weight, 1.0))
        original_usable = bool(window.get("window_usable_for_id"))
        adjusted_usable = original_usable and not hard_block and consistency >= 0.45

        reasons = list(window.get("window_quality_reasons") or [])
        if not adjusted_usable:
            reasons.extend(v["message"] for v in violations if v.get("level") == "hard")

        updated.update({
            "window_algorithm_family": family,
            "window_original_quality_score": original_score,
            "window_quality_score": policy_score,
            "window_policy_score": policy_score,
            "window_ontology_consistency_score": consistency,
            "window_policy_violations": violations,
            "window_policy_adjusted": True,
            "window_policy_hard_blocked": hard_block,
            "window_usable_for_id": adjusted_usable,
            "window_quality_reasons": reasons,
        })
        annotated.append(updated)
        summaries.append({
            "index": index,
            "window_source": updated.get("window_source", ""),
            "algorithm_family": family,
            "original_score": round(original_score, 4),
            "policy_score": round(policy_score, 4),
            "ontology_consistency_score": round(consistency, 4),
            "usable_before_policy": original_usable,
            "usable_after_policy": adjusted_usable,
            "policy_violations": violations,
        })

    return annotated, summaries
