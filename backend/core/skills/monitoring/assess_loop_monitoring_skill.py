"""Loop monitoring skill based on raw LoopFeatures."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.shared.loop_features import extract_loop_features
from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class AssessLoopMonitoringInputs(BaseModel):
    loop_id: str | None = Field(None, description="回路位号，默认使用上下文文件名")
    loop_type: str | None = Field(None, description="回路类型覆盖值")


def _score_to_status(score: float) -> str:
    if score >= 0.8:
        return "normal"
    if score >= 0.55:
        return "warning"
    return "alarm"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return round(v, digits)


def _activity_score(level: str) -> float:
    return {
        "flat": 0.15,
        "low": 0.35,
        "medium": 0.75,
        "high": 0.9,
        "unavailable": 0.0,
    }.get(level, 0.4)


def assess_loop_monitoring_from_features(features: dict[str, Any]) -> dict[str, Any]:
    """Create a monitoring snapshot from raw LoopFeatures.

    This function makes monitoring-level statements, but it does not decide
    tuning readiness or root cause. Those belong to later assessment/diagnosis
    skills.
    """
    data_quality = features.get("data_quality_raw", {})
    pv_stats = features.get("pv_stats", {})
    mv_stats = features.get("mv_stats", {})
    sp_tracking = features.get("sp_tracking_raw", {})
    event_raw = features.get("event_raw", {})
    constraint_raw = features.get("constraint_raw", {})
    frequency_raw = features.get("frequency_raw", {})
    stationarity_raw = features.get("stationarity_raw", {})
    relation_raw = features.get("pv_mv_relation_raw", {})
    operating = features.get("operating_summary_raw", {})

    missing = float(data_quality.get("missing_ratio_total") or 0.0)
    irregular = float(data_quality.get("irregular_sample_ratio") or 0.0)
    quality_score = _clamp01(1.0 - 0.65 * missing - 0.35 * irregular)

    mv_sat = float(constraint_raw.get("mv_saturation_ratio") or 0.0)
    mv_high = float(constraint_raw.get("mv_high_saturation_ratio") or 0.0)
    mv_low = float(constraint_raw.get("mv_low_saturation_ratio") or 0.0)
    constraint_score = _clamp01(1.0 - min(1.0, mv_sat * 3.0))

    pv_power = float(frequency_raw.get("pv_dominant_power_ratio") or 0.0)
    pv_zc_per_h = float(frequency_raw.get("pv_zero_crossing_per_hour") or 0.0)
    rolling_std_p95 = float(stationarity_raw.get("rolling_pv_std_p95") or 0.0)
    pv_std = float(pv_stats.get("std") or 0.0)
    rolling_ratio = rolling_std_p95 / pv_std if pv_std > 0 else 0.0
    stability_score = _clamp01(1.0 - min(0.45, pv_power) - min(0.35, pv_zc_per_h / 180.0) - min(0.2, rolling_ratio / 5.0))

    mv_reversal_per_h = float(mv_stats.get("direction_reversal_per_hour") or 0.0)
    mv_travel_per_h = float(mv_stats.get("travel_per_hour") or 0.0)
    mv_behavior_score = _clamp01(
        1.0
        - min(0.35, mv_reversal_per_h / 120.0)
        - min(0.35, mv_travel_per_h / 180.0)
        - min(0.3, mv_sat * 2.0)
    )

    if sp_tracking.get("sp_available"):
        err_ratio = float(sp_tracking.get("error_within_5pct_span_ratio") or 0.0)
        tracking_score = _clamp01(0.35 + 0.65 * err_ratio)
    else:
        tracking_score = None

    excitation_score = _activity_score(str(operating.get("mv_activity_level_raw") or "unavailable"))
    relation_strength = abs(float(relation_raw.get("cross_correlation_peak_abs") or 0.0))
    response_observability_score = _clamp01(0.55 * excitation_score + 0.45 * min(1.0, relation_strength / 0.5))

    score_parts = [quality_score, constraint_score, stability_score, mv_behavior_score, response_observability_score]
    if tracking_score is not None:
        score_parts.append(tracking_score)
    overall_score = _clamp01(sum(score_parts) / len(score_parts))

    alerts: list[dict[str, Any]] = []
    if missing > 0.05:
        alerts.append({"type": "data_missing", "severity": "warning", "message": f"数据缺失比例 {missing:.1%}"})
    if irregular > 0.05:
        alerts.append({"type": "sample_irregular", "severity": "warning", "message": f"采样不规则比例 {irregular:.1%}"})
    if mv_sat > 0.05:
        alerts.append({"type": "mv_saturation", "severity": "warning", "message": f"MV 贴边/饱和比例 {mv_sat:.1%}"})
    if pv_power > 0.25 and pv_zc_per_h > 10:
        alerts.append({
            "type": "periodic_component",
            "severity": "notice",
            "message": "PV 存在较强周期成分，是否为振荡需交给振荡识别 skill 判断。",
        })
    if mv_reversal_per_h > 60:
        alerts.append({"type": "mv_chatter", "severity": "notice", "message": "MV 方向反复切换较频繁。"})

    return {
        "status": _score_to_status(overall_score),
        "overall_score": round(overall_score, 4),
        "data_health": {
            "score": round(quality_score, 4),
            "status": _score_to_status(quality_score),
            "missing_ratio": _round(missing),
            "irregular_sample_ratio": _round(irregular),
            "long_gap_count": data_quality.get("long_gap_count", 0),
        },
        "stability": {
            "score": round(stability_score, 4),
            "status": _score_to_status(stability_score),
            "pv_dominant_period_s": frequency_raw.get("pv_dominant_period_s"),
            "pv_dominant_power_ratio": frequency_raw.get("pv_dominant_power_ratio"),
            "pv_zero_crossing_per_hour": frequency_raw.get("pv_zero_crossing_per_hour"),
            "rolling_pv_std_p95": stationarity_raw.get("rolling_pv_std_p95"),
        },
        "pv_mv_behavior": {
            "score": round(mv_behavior_score, 4),
            "status": _score_to_status(mv_behavior_score),
            "pv_activity_level_raw": operating.get("pv_activity_level_raw"),
            "mv_activity_level_raw": operating.get("mv_activity_level_raw"),
            "mv_move_count_per_hour": mv_stats.get("move_count_per_hour"),
            "mv_direction_reversal_per_hour": mv_stats.get("direction_reversal_per_hour"),
            "mv_travel_per_hour": mv_stats.get("travel_per_hour"),
            "mv_adjacent_change_per_hour": event_raw.get("mv_adjacent_change_per_hour"),
        },
        "constraints": {
            "score": round(constraint_score, 4),
            "status": _score_to_status(constraint_score),
            "mv_saturation_ratio": _round(mv_sat),
            "mv_high_saturation_ratio": _round(mv_high),
            "mv_low_saturation_ratio": _round(mv_low),
            "longest_mv_saturation_duration_s": constraint_raw.get("longest_mv_saturation_duration_s"),
        },
        "tracking": {
            "sp_available": bool(sp_tracking.get("sp_available")),
            "score": round(tracking_score, 4) if tracking_score is not None else None,
            "status": _score_to_status(tracking_score) if tracking_score is not None else "unavailable",
            "error_abs_mean": sp_tracking.get("error_abs_mean"),
            "error_abs_p95": sp_tracking.get("error_abs_p95"),
            "reason": sp_tracking.get("reason"),
        },
        "response_observability": {
            "score": round(response_observability_score, 4),
            "status": _score_to_status(response_observability_score),
            "estimated_direction_raw": relation_raw.get("estimated_direction_raw"),
            "cross_correlation_peak_abs": relation_raw.get("cross_correlation_peak_abs"),
            "best_lag_s_dpv_dmv": relation_raw.get("best_lag_s_dpv_dmv"),
        },
        "alerts": alerts,
    }


@register
class AssessLoopMonitoringSkill(BaseSkill):
    name = "assess_loop_monitoring"
    description = "基于原始 LoopFeatures 生成回路监控快照，包括数据健康、稳定性、MV 行为、约束和跟踪状态。"
    input_model = AssessLoopMonitoringInputs

    def run(self, inputs: AssessLoopMonitoringInputs, ctx: LoopContext) -> SkillResult:
        if ctx.cleaned_df is None or ctx.dt is None:
            return SkillResult(success=False, reasoning="未检测到已加载的数据集，请先调用 load_dataset。")

        loop_id = inputs.loop_id or Path(ctx.csv_path).stem
        features = extract_loop_features(
            ctx.cleaned_df,
            loop_id=loop_id,
            loop_type=inputs.loop_type or ctx.loop_type,
            source_file=Path(ctx.csv_path).name,
            sample_time_s=float(ctx.dt),
            tag_prefix=ctx.loop_prefix,
        )
        monitoring = assess_loop_monitoring_from_features(features)
        ctx.data_profile["loop_features"] = features
        ctx.data_profile["loop_monitoring"] = monitoring
        return SkillResult(
            success=True,
            data={
                "provider": "raw_feature_rules",
                "features": features,
                "monitoring": monitoring,
            },
            warnings=[alert["message"] for alert in monitoring.get("alerts", [])],
            reasoning=f"已基于原始历史特征生成监控快照，状态 {monitoring['status']}，评分 {monitoring['overall_score']:.2f}。",
        )
