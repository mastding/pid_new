"""Helpers for normalizing window algorithm family names."""
from __future__ import annotations

from typing import Any


_ALIASES = {
    "step_up": "sp_step",
    "step_down": "sp_step",
    "sv_step": "sp_step",
    "sp_step": "sp_step",
    "mv_step": "mv_step",
    "mv_ramp": "mv_ramp",
    "steady_disturbance": "steady_disturbance",
    "steady_disturbance_scan": "steady_disturbance",
    "mv_fallback": "rolling_scan",
    "largest_mv_change": "rolling_scan",
    "rolling_scan": "rolling_scan",
}


def normalize_window_algorithm_family(value: Any) -> str:
    """Return the canonical algorithm family used by policies and UI."""
    raw = str(value or "").strip()
    if not raw:
        return "unknown"
    return _ALIASES.get(raw, raw)


def window_algorithm_family(window: dict[str, Any]) -> str:
    """Infer canonical family from a candidate window dict."""
    return normalize_window_algorithm_family(
        window.get("window_algorithm")
        or window.get("window_algorithm_label")
        or window.get("type")
    )
