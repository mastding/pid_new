"""Window algorithm-family providers.

Each provider emits raw events for one detection family. The composite provider
keeps the previous merge/scoring semantics while making the families pluggable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.algorithms.data_analysis import (
    _detect_mv_activity_segments,
    _detect_mv_steps,
    _detect_steady_disturbance_segments,
    _detect_sv_steps,
)
from core.providers.window.base import BaseWindowDetectionProvider
from core.shared import register_provider


class BaseWindowAlgorithmFamilyProvider(BaseWindowDetectionProvider):
    category = "window_algorithm_family"

    def detect(self, *, df: pd.DataFrame, dt: float, loop_type: str, context=None) -> dict[str, Any]:
        events = self.detect_events(df=df, dt=dt, loop_type=loop_type, context=context or {})
        return {
            "provider": self.name,
            "step_events": events,
            "candidate_windows": [],
            "meta": {
                "loop_type": loop_type,
                "event_count": len(events),
            },
        }

    def detect_events(self, *, df: pd.DataFrame, dt: float, loop_type: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError


@register_provider("window_algorithm_family")
class SpStepWindowFamilyProvider(BaseWindowAlgorithmFamilyProvider):
    name = "sp_step"

    def detect_events(self, *, df: pd.DataFrame, dt: float, loop_type: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        if "SV" not in df.columns or df["SV"].nunique(dropna=True) <= 1:
            return []
        sv_thr = max(0.5, float(df["SV"].std(ddof=0) * 0.2))
        policy = context.get("policy") if isinstance(context, dict) else None
        if isinstance(policy, dict) and policy.get("min_sp_excitation") is not None:
            try:
                sv_thr = max(sv_thr, float(policy.get("min_sp_excitation")))
            except (TypeError, ValueError):
                pass
        return _detect_sv_steps(df, threshold=sv_thr)


@register_provider("window_algorithm_family")
class MvStepWindowFamilyProvider(BaseWindowAlgorithmFamilyProvider):
    name = "mv_step"

    def detect_events(self, *, df: pd.DataFrame, dt: float, loop_type: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        policy = context.get("policy") if isinstance(context, dict) else None
        return _detect_mv_steps(df, policy=policy if isinstance(policy, dict) else None)


@register_provider("window_algorithm_family")
class MvRampWindowFamilyProvider(BaseWindowAlgorithmFamilyProvider):
    name = "mv_ramp"

    def detect_events(self, *, df: pd.DataFrame, dt: float, loop_type: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        policy = context.get("policy") if isinstance(context, dict) else None
        return _detect_mv_activity_segments(df, dt, policy=policy if isinstance(policy, dict) else None)


@register_provider("window_algorithm_family")
class SteadyDisturbanceWindowFamilyProvider(BaseWindowAlgorithmFamilyProvider):
    name = "steady_disturbance"

    def detect_events(self, *, df: pd.DataFrame, dt: float, loop_type: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        policy = context.get("policy") if isinstance(context, dict) else None
        return _detect_steady_disturbance_segments(df, dt, loop_type, policy=policy if isinstance(policy, dict) else None)


@register_provider("window_algorithm_family")
class RollingScanWindowFamilyProvider(BaseWindowAlgorithmFamilyProvider):
    name = "rolling_scan"

    def detect_events(self, *, df: pd.DataFrame, dt: float, loop_type: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        if "MV" not in df.columns:
            return []
        mv = df["MV"].to_numpy(dtype=float)
        if mv.size < 2:
            return []
        mv_diff = np.abs(np.diff(mv))
        center = int(np.argmax(mv_diff))
        return [{
            "start_idx": center,
            "end_idx": min(len(df), center + 2),
            "amplitude": float(np.max(mv_diff)),
            "type": "mv_fallback",
        }]
