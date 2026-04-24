"""Loop-specific priors and preferred model families."""
from __future__ import annotations

from typing import Any

MODEL_ORDER: dict[str, list[str]] = {
    "flow": ["FO", "FOPDT", "SOPDT", "SOPDT_UNDER", "IPDT"],
    "pressure": ["FO", "FOPDT", "SOPDT", "SOPDT_UNDER", "IPDT"],
    "temperature": ["SOPDT", "FOPDT", "FO", "IPDT"],
    "level": ["IFOPDT", "IPDT", "FOPDT", "FO", "SOPDT"],
}

MIN_REASONABLE_T: dict[str, float] = {
    "flow": 1.0,
    "pressure": 5.0,
    "temperature": 30.0,
    "level": 60.0,
}

REALITY_T_RANGES: dict[str, tuple[float, float]] = {
    "flow": (3.0, 20.0),
    "pressure": (15.0, 120.0),
    "temperature": (120.0, 1200.0),
    "level": (300.0, 1800.0),
}


def normalize_loop_type(loop_type: str | None, default: str = "flow") -> str:
    return (loop_type or default).lower().strip()


def model_order_for_loop(loop_type: str | None) -> list[str]:
    loop = normalize_loop_type(loop_type)
    return list(MODEL_ORDER.get(loop, ["FOPDT", "FO", "SOPDT", "IPDT"]))


def min_reasonable_t(loop_type: str | None, dt: float) -> float:
    loop = normalize_loop_type(loop_type)
    return max(3.0 * dt, MIN_REASONABLE_T.get(loop, 1.0))


def reality_t_range_for_loop(loop_type: str | None) -> tuple[float, float]:
    loop = normalize_loop_type(loop_type)
    return REALITY_T_RANGES.get(loop, (5.0, 30.0))
