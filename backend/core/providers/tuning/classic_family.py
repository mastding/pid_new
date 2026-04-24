"""Aggregate tuning provider composed from method-level tuning providers."""
from __future__ import annotations

from core.algorithms.pid_tuning import _heuristic_strategy
from core.policies.constraints import tuning_unreliable_summary, validate_pid_candidate
from core.providers.tuning.base import BaseTuningProvider
from core.shared import provider_registry, register_provider

_ALL_STRATEGY_PROVIDERS = {
    "IMC": "imc",
    "LAMBDA": "lambda",
    "ZN": "zn",
    "CHR": "chr",
}


def _preferred_strategy(
    *,
    loop_type: str,
    model_type: str,
    K: float,
    T: float,
    L: float,
    model_params: dict,
    confidence: float,
    nrmse: float,
    r2: float,
) -> dict[str, str]:
    mt = (model_type or "FOPDT").strip().upper()
    heuristic = _heuristic_strategy(
        loop_type=loop_type,
        model_type=mt,
        K=K,
        T=T,
        L=L,
        model_params=model_params,
        confidence=confidence,
        nrmse=nrmse,
        r2=r2,
    )
    preferred = heuristic["strategy"]
    if mt == "IPDT" and preferred in {"ZN", "CHR"}:
        preferred = "LAMBDA"
    if mt == "SOPDT" and preferred == "ZN":
        preferred = "LAMBDA"
    if mt == "SOPDT_UNDER":
        preferred = "LAMBDA"
    if mt == "IFOPDT" and preferred in {"ZN", "CHR"}:
        preferred = "LAMBDA"
    return {"strategy": preferred, "reason": heuristic["reason"]}


@register_provider("tuning")
class ClassicTuningFamilyProvider(BaseTuningProvider):
    name = "classic_family"

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
        mp = model_params or {}
        preferred_info = _preferred_strategy(
            loop_type=loop_type,
            model_type=model_type,
            K=K,
            T=T,
            L=L,
            model_params=mp,
            confidence=confidence,
            nrmse=nrmse,
            r2=r2,
        )
        preferred = preferred_info["strategy"]

        candidates: list[dict[str, object]] = []
        for strategy, provider_name in _ALL_STRATEGY_PROVIDERS.items():
            provider = provider_registry.get("tuning_strategy", provider_name)
            if provider is None:
                continue
            try:
                result = provider.tune(
                    K=K,
                    T=T,
                    L=L,
                    dt=dt,
                    loop_type=loop_type,
                    model_type=model_type,
                    model_params=mp,
                    confidence=confidence,
                    nrmse=nrmse,
                    r2=r2,
                    context=context,
                )
            except Exception:
                continue
            result["is_recommended"] = strategy == preferred
            reasons = validate_pid_candidate(result, loop_type)
            if reasons:
                result["unreliable"] = True
                result["unreliable_reason"] = "; ".join(reasons)
            candidates.append(result)

        reliable = [c for c in candidates if not c.get("unreliable")]
        if reliable:
            best = next((c for c in reliable if c["strategy"] == preferred), reliable[0])
            tuning_unreliable = False
            unreliable_reason = ""
        else:
            best = next((c for c in candidates if c["strategy"] == preferred), candidates[0] if candidates else None)
            tuning_unreliable = True
            unreliable_reason = tuning_unreliable_summary(loop_type)

        return {
            "provider": self.name,
            "best": best,
            "heuristic_strategy": preferred,
            "heuristic_reason": preferred_info["reason"],
            "all_candidates": candidates,
            "tuning_unreliable": tuning_unreliable,
            "tuning_unreliable_reason": unreliable_reason,
        }
