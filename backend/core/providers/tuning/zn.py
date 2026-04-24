"""ZN tuning provider."""
from __future__ import annotations

from core.algorithms.pid_tuning import apply_tuning_rules
from core.providers.tuning.base import BaseSingleStrategyTuningProvider
from core.shared import register_provider


@register_provider("tuning_strategy")
class ZNTuningProvider(BaseSingleStrategyTuningProvider):
    name = "zn"
    strategy = "ZN"

    def tune(
        self,
        *,
        K: float,
        T: float,
        L: float,
        dt: float,
        loop_type: str,
        model_type: str,
        model_params=None,
        confidence: float = 1.0,
        nrmse: float = 0.0,
        r2: float = 1.0,
        context=None,
    ) -> dict[str, object]:
        result = apply_tuning_rules(
            K=K,
            T=T,
            L=L,
            strategy=self.strategy,
            model_type=model_type,
            model_params=model_params,
        )
        result["provider"] = self.name
        return result
