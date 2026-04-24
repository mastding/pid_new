"""Constraint and guardrail rules shared by tuning and evaluation."""
from __future__ import annotations

from typing import Any

from core.policies.loop_priors import normalize_loop_type

TI_MIN_BY_LOOP: dict[str, float] = {
    "flow": 2.0,
    "pressure": 10.0,
    "temperature": 60.0,
    "level": 60.0,
}

PB_MIN = 5.0
PB_MAX = 1000.0


def tuning_ti_min(loop_type: str | None) -> float:
    return TI_MIN_BY_LOOP.get(normalize_loop_type(loop_type), 2.0)


def validate_pid_candidate(candidate: dict[str, Any], loop_type: str | None) -> list[str]:
    kp = float(candidate.get("Kp", 0.0))
    ti = float(candidate.get("Ti", 0.0))
    pb = (100.0 / kp) if kp > 1e-9 else float("inf")
    reasons: list[str] = []
    ti_min = tuning_ti_min(loop_type)
    if ti > 0 and ti < ti_min:
        reasons.append(f"TI={ti:.1f}s 低于{loop_type}最小合理值 {ti_min:.0f}s")
    if not (PB_MIN <= pb <= PB_MAX):
        reasons.append(f"PB={pb:.2f}% 越界 [{PB_MIN},{PB_MAX}]")
    return reasons


def tuning_unreliable_summary(loop_type: str | None) -> str:
    ti_min = tuning_ti_min(loop_type)
    return (
        f"所有候选策略 PID 参数物理量级不合理（最小 TI={ti_min:.0f}s，PB ∈ [{PB_MIN},{PB_MAX}]%），"
        f"通常意味着辨识模型时间常数塌缩，建议沿用现场参数或重做手动阶跃测试"
    )
