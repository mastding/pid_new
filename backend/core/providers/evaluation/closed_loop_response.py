"""Method-level provider for closed-loop response simulation."""
from __future__ import annotations

from core.algorithms.pid_evaluation import _simulate
from core.providers.evaluation.base import BaseSimulationProvider
from core.shared import register_provider


@register_provider("evaluation_simulation")
class ClosedLoopResponseProvider(BaseSimulationProvider):
    name = "closed_loop_response"

    def simulate(
        self,
        *,
        model_params,
        pid_params,
        sp_initial: float,
        sp_final: float,
        n_steps: int,
        dt: float,
        loop_type: str,
        context=None,
    ) -> dict[str, object]:
        result = _simulate(
            model_params,
            pid_params,
            sp_initial=sp_initial,
            sp_final=sp_final,
            n_steps=n_steps,
            dt=dt,
            loop_type=loop_type,
        )
        result["provider"] = self.name
        return result
