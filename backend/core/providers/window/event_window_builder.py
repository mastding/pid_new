"""Build scored candidate windows from window-detection events."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.algorithms.data_analysis import _adaptive_padding, score_window


def build_windows_from_events(
    *,
    df: pd.DataFrame,
    dt: float,
    loop_type: str,
    events: list[dict[str, Any]],
    policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert detection events into scored identification-window candidates."""
    n = len(df)
    windows: list[dict[str, Any]] = []
    for ev in events:
        center = int(ev["start_idx"])
        amp = float(ev.get("amplitude", 1.0))
        if ev.get("type") == "steady_disturbance":
            w_start = max(0, int(ev.get("start_idx", 0)))
            w_end = min(n, int(ev.get("end_idx", n)))
        else:
            pre, post = _adaptive_padding(amp, dt, n, loop_type, policy=policy)
            w_start = max(0, center - pre)
            w_end = min(n, center + post)
        if w_end - w_start < 20:
            continue

        seg = df.iloc[w_start:w_end]
        quality = (
            ev.get("pre_scored_quality")
            if isinstance(ev.get("pre_scored_quality"), dict)
            else score_window(seg, policy=policy)
        )
        algorithm = "sv_step" if ev.get("type") in {"step_up", "step_down"} else str(ev.get("type", "unknown"))
        if algorithm == "mv_fallback":
            algorithm_label = "largest_mv_change"
        elif algorithm == "steady_disturbance":
            algorithm_label = "steady_disturbance_scan"
        else:
            algorithm_label = algorithm
        windows.append({
            **ev,
            "window_algorithm": algorithm,
            "window_algorithm_label": algorithm_label,
            "window_selection_basis": "score_window: mv_excitation + pv_response + lag_correlation - saturation/drift",
            "window_start_idx": w_start,
            "window_end_idx": w_end,
            "window_usable_for_id": quality["passed"],
            "window_quality_score": quality["score"],
            "window_quality_reasons": quality["reasons"],
            "window_score_breakdown": quality["score_breakdown"],
            "window_quality_metrics": quality["raw_metrics"],
            "window_operating_state": quality.get("operating_state"),
            "window_mv_span": quality["mv_span"],
            "window_pv_span": quality["pv_span"],
            "window_corr": quality["corr"],
        })

    windows.sort(key=lambda w: (
        int(bool(w.get("window_usable_for_id"))),
        float(w.get("window_quality_score", 0.0)),
    ), reverse=True)
    return label_window_sources(windows)


def label_window_sources(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    type_counter: dict[str, int] = {}
    for w in windows:
        base = str(w.get("window_algorithm") or "")
        if base in {"step_up", "step_down", "sv_step"}:
            base = "sp_step"
        if base not in {"sp_step", "mv_step", "mv_ramp", "steady_disturbance", "mv_fallback"}:
            base = "mv_change"
        if base == "mv_fallback":
            base = "rolling_scan"
        type_counter[base] = type_counter.get(base, 0) + 1
        w["window_source"] = f"{base}_{type_counter[base]}"
    return windows
