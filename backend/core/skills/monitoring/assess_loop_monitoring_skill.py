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


def _severity_rank(severity: str) -> int:
    return {
        "notice": 1,
        "low": 1,
        "warning": 2,
        "medium": 2,
        "alarm": 3,
        "high": 3,
        "critical": 4,
    }.get(severity, 1)


def _event(
    *,
    event_type: str,
    severity: str,
    name: str,
    message: str,
    evidence: dict[str, Any] | None = None,
    recommendation: str = "",
) -> dict[str, Any]:
    return {
        "type": event_type,
        "severity": severity,
        "name": name,
        "message": message,
        "evidence": evidence or {},
        "recommendation": recommendation,
        "status": "new",
        "source": "assess_loop_monitoring",
    }


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
    noise_raw = features.get("noise_raw", {})
    oscillation_raw = features.get("oscillation_raw", {})
    stationarity_raw = features.get("stationarity_raw", {})
    relation_raw = features.get("pv_mv_relation_raw", {})
    operating = features.get("operating_summary_raw", {})
    operating_condition = features.get("operating_condition_profile", {})

    missing = float(data_quality.get("missing_ratio_total") or 0.0)
    irregular = float(data_quality.get("irregular_sample_ratio") or 0.0)
    duplicate_ratio = float(data_quality.get("duplicate_timestamp_ratio") or 0.0)
    long_gap_count = int(data_quality.get("long_gap_count") or 0)
    pv_outlier_count = int(data_quality.get("pv_outlier_count") or 0)
    mv_outlier_count = int(data_quality.get("mv_outlier_count") or 0)
    pv_noise_ratio = float(noise_raw.get("pv_noise_ratio") or 0.0)
    pv_spike_ratio = float(noise_raw.get("pv_spike_ratio") or 0.0)
    mv_spike_ratio = float(noise_raw.get("mv_spike_ratio") or 0.0)
    quality_score = _clamp01(
        1.0
        - 0.45 * missing
        - 0.2 * irregular
        - 0.15 * duplicate_ratio
        - min(0.12, long_gap_count * 0.02)
        - min(0.2, pv_noise_ratio * 4.0)
        - min(0.12, (pv_spike_ratio + mv_spike_ratio) * 8.0)
    )

    mv_sat = float(constraint_raw.get("mv_saturation_ratio") or 0.0)
    mv_high = float(constraint_raw.get("mv_high_saturation_ratio") or 0.0)
    mv_low = float(constraint_raw.get("mv_low_saturation_ratio") or 0.0)
    constraint_score = _clamp01(1.0 - min(1.0, mv_sat * 3.0))

    pv_power = float(frequency_raw.get("pv_dominant_power_ratio") or 0.0)
    pv_zc_per_h = float(frequency_raw.get("pv_zero_crossing_per_hour") or 0.0)
    rolling_std_p95 = float(stationarity_raw.get("rolling_pv_std_p95") or 0.0)
    pv_std = float(pv_stats.get("std") or 0.0)
    rolling_ratio = rolling_std_p95 / pv_std if pv_std > 0 else 0.0
    oscillation_detected = bool(oscillation_raw.get("detected"))
    oscillation_conf = float(oscillation_raw.get("confidence") or 0.0)
    stability_score = _clamp01(
        1.0
        - min(0.35, pv_power)
        - min(0.25, pv_zc_per_h / 180.0)
        - min(0.2, rolling_ratio / 5.0)
        - (0.25 * oscillation_conf if oscillation_detected else 0.0)
    )

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

    events: list[dict[str, Any]] = []
    if missing > 0.05:
        events.append(_event(
            event_type="data_quality",
            severity="warning",
            name="数据缺失",
            message=f"数据缺失比例 {missing:.1%}",
            evidence={"missing_ratio": _round(missing)},
            recommendation="检查历史库采集链路，必要时剔除缺失片段。",
        ))
    if irregular > 0.05:
        events.append(_event(
            event_type="data_quality",
            severity="warning",
            name="采样不规则",
            message=f"采样不规则比例 {irregular:.1%}",
            evidence={"irregular_sample_ratio": _round(irregular), "long_gap_count": long_gap_count},
            recommendation="确认采样周期和时间戳，避免影响频谱、调节时间和窗口识别。",
        ))
    if pv_noise_ratio > 0.02 or pv_spike_ratio > 0.01:
        events.append(_event(
            event_type="noise",
            severity="warning" if pv_noise_ratio <= 0.05 else "alarm",
            name="PV噪声/尖峰",
            message=f"PV高频噪声比 {pv_noise_ratio:.2%}，尖峰比例 {pv_spike_ratio:.2%}",
            evidence={
                "pv_noise_ratio": _round(pv_noise_ratio),
                "pv_snr_db": noise_raw.get("pv_snr_db"),
                "pv_spike_count": noise_raw.get("pv_spike_count"),
                "pv_outlier_count": pv_outlier_count,
            },
            recommendation="先确认测量链路和滤波配置，再使用该数据做辨识或性能评价。",
        ))
    if mv_sat > 0.05:
        events.append(_event(
            event_type="constraint",
            severity="warning" if mv_sat <= 0.2 else "alarm",
            name="MV贴边/饱和",
            message=f"MV 贴边/饱和比例 {mv_sat:.1%}",
            evidence={
                "mv_saturation_ratio": _round(mv_sat),
                "mv_high_saturation_ratio": _round(mv_high),
                "mv_low_saturation_ratio": _round(mv_low),
                "longest_mv_saturation_duration_s": constraint_raw.get("longest_mv_saturation_duration_s"),
                "segment_count": constraint_raw.get("mv_saturation_segment_count"),
            },
            recommendation="饱和片段会污染辨识和性能评价，应先确认工况/阀位能力。",
        ))
    if oscillation_detected:
        events.append(_event(
            event_type="oscillation",
            severity=str(oscillation_raw.get("severity") or "warning"),
            name="疑似振荡",
            message=f"PV 存在周期成分，主周期 {oscillation_raw.get('pv_dominant_period_s') or '-'}s",
            evidence={
                "confidence": oscillation_raw.get("confidence"),
                "pv_dominant_power_ratio": oscillation_raw.get("pv_dominant_power_ratio"),
                "pv_zero_crossing_per_hour": oscillation_raw.get("pv_zero_crossing_per_hour"),
                "phase_hint": oscillation_raw.get("phase_hint"),
            },
            recommendation="进入振荡监测查看频谱证据；若PV/MV同频，再进入根因诊断区分PID、阀门或外扰。",
        ))
    if mv_reversal_per_h > 60:
        events.append(_event(
            event_type="mv_behavior",
            severity="notice",
            name="MV频繁换向",
            message="MV 方向反复切换较频繁。",
            evidence={
                "mv_direction_reversal_per_hour": _round(mv_reversal_per_h),
                "mv_spike_count": noise_raw.get("mv_spike_count"),
                "mv_outlier_count": mv_outlier_count,
            },
            recommendation="结合阀门/执行机构诊断检查卡滞、死区或控制器过激。",
        ))

    suitability = str(operating_condition.get("tuning_suitability") or "")
    if suitability in {"cautious", "not_recommended"}:
        condition_label = str(operating_condition.get("condition_label") or "unknown")
        events.append(_event(
            event_type="operating_condition",
            severity="warning" if suitability == "cautious" else "alarm",
            name="运行工况",
            message=f"当前工况 {condition_label}，整定建议 {suitability}",
            evidence={
                "condition_label": condition_label,
                "confidence": operating_condition.get("confidence"),
                "tuning_suitability": suitability,
            },
            recommendation="进入运行工况页面查看证据，并优先选择稳定、非饱和、激励充分的片段。",
        ))

    events.sort(key=lambda item: _severity_rank(str(item.get("severity", ""))), reverse=True)
    alerts = [
        {"type": item["type"], "severity": item["severity"], "message": item["message"]}
        for item in events
    ]

    return {
        "status": _score_to_status(overall_score),
        "overall_score": round(overall_score, 4),
        "data_health": {
            "score": round(quality_score, 4),
            "status": _score_to_status(quality_score),
            "missing_ratio": _round(missing),
            "irregular_sample_ratio": _round(irregular),
            "duplicate_timestamp_ratio": _round(duplicate_ratio),
            "long_gap_count": long_gap_count,
            "pv_outlier_count": pv_outlier_count,
            "mv_outlier_count": mv_outlier_count,
            "pv_noise_ratio": _round(pv_noise_ratio),
            "pv_snr_db": noise_raw.get("pv_snr_db"),
            "pv_spike_count": noise_raw.get("pv_spike_count"),
        },
        "stability": {
            "score": round(stability_score, 4),
            "status": _score_to_status(stability_score),
            "oscillation_detected": oscillation_detected,
            "oscillation_severity": oscillation_raw.get("severity"),
            "oscillation_confidence": oscillation_raw.get("confidence"),
            "pv_dominant_period_s": frequency_raw.get("pv_dominant_period_s"),
            "pv_dominant_power_ratio": frequency_raw.get("pv_dominant_power_ratio"),
            "pv_zero_crossing_per_hour": frequency_raw.get("pv_zero_crossing_per_hour"),
            "phase_hint": oscillation_raw.get("phase_hint"),
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
            "mv_spike_count": noise_raw.get("mv_spike_count"),
            "mv_spike_ratio": noise_raw.get("mv_spike_ratio"),
        },
        "constraints": {
            "score": round(constraint_score, 4),
            "status": _score_to_status(constraint_score),
            "mv_saturation_ratio": _round(mv_sat),
            "mv_high_saturation_ratio": _round(mv_high),
            "mv_low_saturation_ratio": _round(mv_low),
            "longest_mv_saturation_duration_s": constraint_raw.get("longest_mv_saturation_duration_s"),
            "mv_saturation_segment_count": constraint_raw.get("mv_saturation_segment_count"),
            "reason": "MV存在明显贴边/饱和" if mv_sat > 0.05 else "暂无明显约束风险",
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
            "process_direction": relation_raw.get("process_direction"),
            "process_direction_confidence": relation_raw.get("process_direction_confidence"),
            "process_direction_basis": relation_raw.get("process_direction_basis"),
            "cross_correlation_peak_abs": relation_raw.get("cross_correlation_peak_abs"),
            "best_lag_s_dpv_dmv": relation_raw.get("best_lag_s_dpv_dmv"),
        },
        "operating_condition": {
            "condition_label": operating_condition.get("condition_label"),
            "confidence": operating_condition.get("confidence"),
            "tuning_suitability": operating_condition.get("tuning_suitability"),
            "evidence": operating_condition.get("evidence", []),
            "recommendations": operating_condition.get("recommendations", []),
        },
        "noise": {
            "score": round(_clamp01(1.0 - min(0.7, pv_noise_ratio * 8.0) - min(0.3, pv_spike_ratio * 10.0)), 4),
            "status": _score_to_status(_clamp01(1.0 - min(0.7, pv_noise_ratio * 8.0) - min(0.3, pv_spike_ratio * 10.0))),
            **noise_raw,
        },
        "oscillation": oscillation_raw,
        "events": events,
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
