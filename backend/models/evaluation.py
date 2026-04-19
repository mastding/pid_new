"""Evaluation result models."""
from __future__ import annotations

from pydantic import BaseModel


class EvaluationResult(BaseModel):
    """PID parameter evaluation result."""

    passed: bool = False
    performance_score: float = 0.0
    final_rating: float = 0.0
    method_confidence: float = 0.0
    robustness_score: float = 0.0
    is_stable: bool = False
    overshoot_percent: float = 0.0
    settling_time: float = 0.0
    failure_reason: str = ""
    feedback_target: str = ""
    feedback_action: str = ""
