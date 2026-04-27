"""Configurable rules for identification refinement fallback."""
from __future__ import annotations

from dataclasses import dataclass

from core.policies.loop_priors import normalize_loop_type


@dataclass(frozen=True)
class RefinementFallbackRule:
    min_confidence: float = 0.25
    min_r2: float = 0.20
    min_window_quality: float = 0.65
    max_model_pool_size: int = 3


REFINEMENT_FALLBACK_RULE = RefinementFallbackRule()


REFINEMENT_MODEL_FALLBACKS: dict[str, list[str]] = {
    "flow": ["FO", "FOPDT", "SOPDT_UNDER"],
    "temperature": ["FOPDT", "SOPDT", "SOPDT_UNDER"],
    "level": ["IPDT", "IFOPDT", "FOPDT"],
    "pressure": ["FOPDT", "FO", "SOPDT_UNDER"],
}


def refinement_model_fallbacks_for_loop(loop_type: str | None) -> list[str]:
    loop = normalize_loop_type(loop_type)
    return list(REFINEMENT_MODEL_FALLBACKS.get(loop, ["FOPDT", "FO", "SOPDT_UNDER"]))


def refinement_fallback_rule() -> RefinementFallbackRule:
    return REFINEMENT_FALLBACK_RULE
