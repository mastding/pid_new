"""Deterministic fallback policy for identification refinement.

The LLM refinement advisor is useful, but the pipeline should still make a
bounded, explainable retry decision when the advisor is unavailable. This module
uses the previous round's window-algorithm comparison to pick a diverse fallback
candidate instead of repeating the exact downgraded model/window pair.
"""
from __future__ import annotations

from typing import Any

from core.policies.refinement import (
    refinement_fallback_rule,
    refinement_model_fallbacks_for_loop,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _window_index_by_source(windows_summary: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(item.get("source", "")): int(item.get("index", idx))
        for idx, item in enumerate(windows_summary)
    }


def _is_candidate_usable(item: dict[str, Any]) -> bool:
    # R2/confidence are post-fit reliability signals; window score is only a data
    # quality proxy. Keep the threshold modest so fallback can still explore.
    rule = refinement_fallback_rule()
    confidence = _safe_float(item.get("confidence"))
    r2 = _safe_float(item.get("r2_score"))
    window_score = _safe_float(item.get("window_quality_score"))
    return (
        confidence >= rule.min_confidence
        or r2 >= rule.min_r2
        or window_score >= rule.min_window_quality
    )


def recommend_refinement_from_algorithm_comparison(
    *,
    loop_type: str,
    windows_summary: list[dict[str, Any]],
    algorithm_comparison: list[dict[str, Any]],
    last_best: dict[str, Any],
    last_review: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a deterministic retry instruction, or None if retry is not useful.

    The policy deliberately chooses a *different* window/model pair from the
    downgraded best attempt. Repeating the same candidate would only consume
    time without adding evidence.
    """
    if not algorithm_comparison:
        return None

    source_to_index = _window_index_by_source(windows_summary)
    last_source = str(last_best.get("window_source", ""))
    last_model = str(last_best.get("model_type", "")).upper()

    ranked = sorted(
        [item for item in algorithm_comparison if _is_candidate_usable(item)],
        key=lambda item: (
            _safe_float(item.get("confidence")),
            _safe_float(item.get("r2_score")),
            _safe_float(item.get("fit_score"), -1e12),
            _safe_float(item.get("window_quality_score")),
        ),
        reverse=True,
    )
    if not ranked:
        return None

    chosen: dict[str, Any] | None = None
    for item in ranked:
        source = str(item.get("window_source", ""))
        model = str(item.get("model_type", "")).upper()
        if source != last_source or model != last_model:
            chosen = item
            break
    if chosen is None:
        return None

    source = str(chosen.get("window_source", ""))
    force_window_index = source_to_index.get(source)
    if force_window_index is None:
        return None

    chosen_model = str(chosen.get("model_type", "")).upper()
    model_pool = [chosen_model] if chosen_model else []
    rule = refinement_fallback_rule()
    for model_type in refinement_model_fallbacks_for_loop(loop_type):
        if model_type not in model_pool:
            model_pool.append(model_type)
    model_pool = model_pool[:rule.max_model_pool_size]

    reason = str(last_review.get("reason", "")).strip()
    rationale = (
        f"确定性精修：上一轮被降级后，改用算法族 "
        f"{chosen.get('algorithm_label') or chosen.get('algorithm') or '-'} 的备选窗口 {source}；"
        f"该族最佳 R²={_safe_float(chosen.get('r2_score')):.3f}、"
        f"置信度={_safe_float(chosen.get('confidence')):.0%}。"
    )
    if reason:
        rationale += f" 降级原因：{reason[:80]}"

    return {
        "retry": True,
        "source": "deterministic_algorithm_policy",
        "rationale": rationale,
        "force_window_index": force_window_index,
        "force_model_types": model_pool,
        "hint_L": None,
        "recommended_algorithm": chosen.get("algorithm", ""),
        "recommended_algorithm_label": chosen.get("algorithm_label") or chosen.get("algorithm", ""),
        "recommended_window_source": source,
        "evidence": chosen,
        "policy": {
            "min_confidence": rule.min_confidence,
            "min_r2": rule.min_r2,
            "min_window_quality": rule.min_window_quality,
            "max_model_pool_size": rule.max_model_pool_size,
        },
    }
