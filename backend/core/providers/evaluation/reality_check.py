"""Method-level provider for adaptive reality-check evaluation."""
from __future__ import annotations

from core.algorithms.pid_evaluation import _adaptive_reality_check_t
from core.shared import provider_registry, register_provider
from core.providers.evaluation.base import BaseRealityCheckProvider


@register_provider("evaluation_reality_check")
class AdaptiveRealityCheckProvider(BaseRealityCheckProvider):
    name = "adaptive_typical_t"

    def check(
        self,
        *,
        model_params,
        pid_params,
        perf_score: float,
        confidence: float,
        loop_type: str,
        n_steps: int,
        dt: float,
        context=None,
    ) -> dict[str, object]:
        scenario = None
        if isinstance(context, dict):
            scenario = context.get("evaluation_primary_scenario")
        mt = str(model_params.get("model_type", "FOPDT")).upper()
        typical_t = _adaptive_reality_check_t(loop_type, float(model_params.get("T1", model_params.get("T", 0.0))), confidence)
        reality_score = perf_score
        diverged = False
        reality_sim = None

        if typical_t > 0:
            sim_provider = provider_registry.get("evaluation_simulation", "closed_loop_response")
            score_provider = provider_registry.get("evaluation_scoring", "step_response_scoring")
            if sim_provider is None or score_provider is None:
                raise RuntimeError("reality check requires simulation and scoring providers")

            reality_mp = dict(model_params)
            reality_mp["T1"] = typical_t
            reality_mp["T2"] = typical_t * 0.3 if mt == "SOPDT" else 0.0

            reality_sim = sim_provider.simulate(
                model_params=reality_mp,
                pid_params=pid_params,
                sp_initial=float((scenario or {}).get("sp_initial", 50.0)),
                sp_final=float((scenario or {}).get("sp_final", 60.0)),
                n_steps=n_steps,
                dt=dt,
                loop_type=loop_type,
                context=context,
            )
            scored = score_provider.score(simulation=reality_sim, context=context)
            reality_score = float(scored["score"])
            diverged = (perf_score - reality_score) > 3.0 or not bool(reality_sim.get("is_stable", True))

        return {
            "provider": self.name,
            "typical_t": typical_t,
            "reality_score": round(reality_score, 2),
            "diverged": diverged,
            "simulation": reality_sim,
        }
