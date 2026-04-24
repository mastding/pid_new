"""Method-level provider for step-response performance scoring."""
from __future__ import annotations

from core.algorithms.pid_evaluation import _performance_score
from core.providers.evaluation.base import BasePerformanceScoringProvider
from core.shared import register_provider


@register_provider("evaluation_scoring")
class StepResponseScoringProvider(BasePerformanceScoringProvider):
    name = "step_response_scoring"

    def score(self, *, simulation, context=None) -> dict[str, object]:
        score, details = _performance_score(simulation)
        return {
            "provider": self.name,
            "score": score,
            "details": details,
        }
