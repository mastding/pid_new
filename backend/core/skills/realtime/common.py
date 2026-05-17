"""Shared helpers for realtime assessment skills."""
from __future__ import annotations

from typing import Any


LEVEL_RANK = {"normal": 0, "potential": 1, "low": 2, "medium": 3, "high": 4}


def metric_value(metric: dict[str, Any], *path: str) -> Any:
    cur: Any = metric
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def level_from_score(score: float | None) -> str:
    if score is None:
        return "potential"
    if score < 0.45:
        return "high"
    if score < 0.65:
        return "medium"
    if score < 0.82:
        return "low"
    return "normal"


def metric_level_to_risk(name: str, level: str | None, value: float | None = None) -> str:
    level = str(level or "").lower()
    if name == "harris":
        if value is not None:
            if value < 0.45:
                return "medium"
            if value < 0.6:
                return "low"
        if level in {"poor", "weak", "alarm"}:
            return "medium"
    if name == "cpk":
        if value is not None:
            if value < 1.0:
                return "medium"
            if value < 1.33:
                return "low"
        if level in {"poor", "weak"}:
            return "medium"
    return "normal"


def max_level(*levels: str) -> str:
    return max((level or "normal" for level in levels), key=lambda item: LEVEL_RANK.get(item, 0))
