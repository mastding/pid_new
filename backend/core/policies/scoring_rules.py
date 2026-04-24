"""Shared scoring and capping rules for evaluation."""
from __future__ import annotations

import numpy as np

from core.policies.loop_priors import normalize_loop_type, reality_t_range_for_loop


def stability_limits(loop_type: str | None, t1: float) -> dict[str, float]:
    loop = normalize_loop_type(loop_type)
    t1 = max(float(t1), 1e-6)
    if loop == "level":
        settling_limit = min(3600.0, max(900.0, 6.0 * t1))
    elif loop == "temperature":
        settling_limit = min(2400.0, max(600.0, 8.0 * t1))
    elif loop == "pressure":
        settling_limit = min(900.0, max(240.0, 10.0 * t1))
    else:
        settling_limit = min(600.0, max(120.0, 12.0 * t1))
    return {
        "settling_time_limit": settling_limit,
        "steady_state_error_limit": 8.0,
        "decay_ratio_limit": 0.8,
    }


def adaptive_reality_check_t(loop_type: str | None, identified_t: float, confidence: float) -> float:
    lo, hi = reality_t_range_for_loop(loop_type)
    identified_t = max(float(identified_t or 0.0), 1e-6)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if identified_t < lo:
        return lo
    if identified_t > hi:
        return hi

    slow_factor = 1.15 + 0.55 * (1.0 - confidence)
    return max(lo, min(hi, identified_t * slow_factor))


def final_rating(perf_score: float, confidence: float) -> float:
    raw = 0.7 * perf_score + 0.3 * confidence * 10.0
    if perf_score <= 1.0:
        raw = min(raw, 3.0)
    elif perf_score <= 3.0:
        raw = min(raw, 5.0)
    if confidence < 0.2:
        raw = min(raw, 6.0)
    return round(min(10.0, max(0.0, raw)), 2)


def apply_score_caps(
    *,
    perf_score: float,
    final_rating_score: float,
    readiness_score: float,
    confidence: float,
    reality_diverged: bool,
    reality_score: float,
    loop_type: str,
    tuning_unreliable: bool,
    tuning_unreliable_reason: str,
    passed: bool,
) -> tuple[float, float, float, bool, list[str]]:
    cap_reasons: list[str] = []
    perf = perf_score
    final = final_rating_score
    readiness = readiness_score
    accepted = passed

    if confidence < 0.5:
        perf = min(perf, 5.0)
        final = min(final, 5.0)
        readiness = min(readiness, 5.0)
        accepted = False
        cap_reasons.append(f"模型置信度 {confidence:.2f} < 0.5，评分封顶 5 分")

    if reality_diverged:
        perf = min(perf, 4.0)
        final = min(final, 4.0)
        readiness = min(readiness, 4.0)
        accepted = False
        cap_reasons.append(
            f"Reality check 发散：用 {loop_type} 典型时间常数仿真评分 {reality_score:.1f}，与名义模型评分差距过大，提示辨识模型可能塌缩"
        )

    if tuning_unreliable:
        perf = min(perf, 3.0)
        final = min(final, 3.0)
        readiness = min(readiness, 3.0)
        accepted = False
        cap_reasons.append(f"整定参数物理量级不合理：{tuning_unreliable_reason}")

    return perf, final, readiness, accepted, cap_reasons
