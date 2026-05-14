"""Loop assessment skill built on monitoring snapshots and raw LoopFeatures."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.shared.loop_features import extract_loop_features
from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.monitoring.assess_loop_monitoring_skill import assess_loop_monitoring_from_features
from core.skills.registry import register


class AssessLoopAssessmentInputs(BaseModel):
    loop_id: str | None = Field(None, description="回路位号，默认使用上下文文件名")
    loop_type: str | None = Field(None, description="回路类型覆盖值")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _level(score: float) -> str:
    if score >= 0.8:
        return "excellent"
    if score >= 0.6:
        return "good"
    if score >= 0.4:
        return "fair"
    return "weak"


def _status_to_score(status: str | None) -> float:
    if status == "normal":
        return 1.0
    if status == "warning":
        return 0.6
    if status == "alarm":
        return 0.25
    return 0.5


def _gate(name: str, passed: bool, severity: str, message: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "severity": "ok" if passed else severity,
        "message": message,
        "evidence": evidence or {},
    }


def _recommended_action(blocking_reasons: list[dict[str, Any]], readiness_score: float, identification_score: float) -> str:
    if not blocking_reasons and readiness_score >= 0.75:
        return "start_tuning"
    if any(item["type"] == "data_quality" for item in blocking_reasons):
        return "fix_data_quality"
    if any(item["type"] == "constraint" for item in blocking_reasons):
        return "check_constraint_or_valve_capacity"
    if any(item["type"] == "oscillation" for item in blocking_reasons):
        return "run_oscillation_diagnosis"
    if any(item["type"] == "operating_condition" for item in blocking_reasons):
        return "select_steady_operating_segment"
    if identification_score < 0.55:
        return "improve_identification_excitation"
    return "manual_review_before_tuning"


def _recommendation_text(code: str) -> str:
    mapping = {
        "start_tuning": "当前数据和工况满足初步条件，可以发起整定任务，并保留模型评审兜底。",
        "fix_data_quality": "先处理数据质量问题：缺失、采样异常、噪声尖峰或数据断点。",
        "check_constraint_or_valve_capacity": "先确认 MV 饱和/约束是否来自阀位能力、工况边界或执行机构问题。",
        "run_oscillation_diagnosis": "先进入振荡诊断，区分 PID 过激、阀门问题和外部周期扰动。",
        "select_steady_operating_segment": "优先选择稳定、非饱和、激励充分的历史片段，再进入辨识整定。",
        "improve_identification_excitation": "当前可辨识性不足，建议寻找更明显 MV 激励片段或补充受控小阶跃。",
        "manual_review_before_tuning": "建议人工复核关键证据后再决定是否整定。",
    }
    return mapping.get(code, code)


def assess_loop_assessment_from_features(
    features: dict[str, Any],
    monitoring: dict[str, Any] | None = None,
    *,
    window_summary: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a standardized loop assessment result.

    The assessment layer intentionally consumes already-computed observable facts.
    Window detection remains a separate skill; when available, its summary is used
    as one evidence source rather than being embedded in raw LoopFeatures.
    """
    monitoring = monitoring or assess_loop_monitoring_from_features(features)
    window_summary = window_summary or {}
    diagnostics = diagnostics or {}

    data_health = monitoring.get("data_health", {})
    stability = monitoring.get("stability", {})
    constraints = monitoring.get("constraints", {})
    response = monitoring.get("response_observability", {})
    noise = monitoring.get("noise", {})
    operating_condition = features.get("operating_condition_profile", {})
    excitation = features.get("excitation_profile", {})
    actuator = features.get("actuator_profile", {})
    process_prior = features.get("process_prior", {})
    quality_raw = features.get("data_quality_raw", {})

    data_quality_score = _clamp01(float(data_health.get("score") or 0.0))
    performance_score = _clamp01(
        0.32 * float(monitoring.get("overall_score") or 0.0)
        + 0.24 * float(stability.get("score") or 0.0)
        + 0.22 * float(constraints.get("score") or 0.0)
        + 0.22 * float(monitoring.get("pv_mv_behavior", {}).get("score") or 0.0)
    )

    # 准入校验本身只需要回答"这条回路是否值得进入辨识"——这个判断完全可以
    # 来自画像层（激励充分性 / 过程方向置信度 / 响应可观测性）。窗口检测的产物
    # 只在用户真的发起整定时再算。
    has_window_summary = bool(window_summary)
    if has_window_summary:
        usable_ratio = float(window_summary.get("usable_window_count") or 0.0) / max(float(window_summary.get("window_count") or 0.0), 1.0)
        best_window_score = float(window_summary.get("best_window_score") or 0.0)
    else:
        usable_ratio = 0.0
        best_window_score = 0.0
    excitation_score = float(excitation.get("usable_excitation_ratio") or 0.0)
    direction_score = float(process_prior.get("process_direction_confidence") or 0.0)
    response_score = float(response.get("score") or 0.0)
    if has_window_summary:
        identification_score = _clamp01(
            0.35 * excitation_score
            + 0.25 * response_score
            + 0.2 * direction_score
            + 0.2 * max(best_window_score, usable_ratio)
        )
    else:
        # 没有窗口信息时把权重重新分配到三项画像指标上；激励 + 响应 + 方向
        # 三项加权和已经能覆盖"可辨识性"的核心要素。
        identification_score = _clamp01(
            0.45 * excitation_score
            + 0.30 * response_score
            + 0.25 * direction_score
        )

    condition_suitability = str(operating_condition.get("tuning_suitability") or "unknown")
    condition_score = {"suitable": 1.0, "cautious": 0.62, "not_recommended": 0.2}.get(condition_suitability, 0.5)
    actuator_score = _clamp01(
        1.0
        - min(0.4, float(actuator.get("mv_saturation_ratio") or 0.0) * 1.6)
        - (0.25 if actuator.get("mv_rate_limit_hint") else 0.0)
        - (0.25 if actuator.get("mv_stiction_hint") else 0.0)
    )
    readiness_score = _clamp01(
        0.26 * data_quality_score
        + 0.24 * performance_score
        + 0.22 * identification_score
        + 0.18 * condition_score
        + 0.10 * actuator_score
    )

    blocking_reasons: list[dict[str, Any]] = []
    if data_quality_score < 0.65:
        blocking_reasons.append({
            "type": "data_quality",
            "severity": "high" if data_quality_score < 0.4 else "medium",
            "message": "数据质量不足，可能影响监控、辨识和整定判断。",
            "evidence": {
                "missing_ratio": quality_raw.get("missing_ratio_total"),
                "irregular_sample_ratio": quality_raw.get("irregular_sample_ratio"),
                "score": round(data_quality_score, 4),
            },
        })
    mv_saturation = float(constraints.get("mv_saturation_ratio") or actuator.get("mv_saturation_ratio") or 0.0)
    if mv_saturation >= 0.05:
        blocking_reasons.append({
            "type": "constraint",
            "severity": "high" if mv_saturation >= 0.2 else "medium",
            "message": "MV 存在饱和/贴边，辨识和整定前应确认约束来源。",
            "evidence": {"mv_saturation_ratio": round(mv_saturation, 6)},
        })
    if bool(stability.get("oscillation_detected")):
        blocking_reasons.append({
            "type": "oscillation",
            "severity": "medium",
            "message": "检测到振荡迹象，建议先区分 PID、阀门或外扰来源。",
            "evidence": {
                "confidence": stability.get("oscillation_confidence"),
                "period_s": stability.get("pv_dominant_period_s"),
            },
        })
    if condition_suitability == "not_recommended":
        blocking_reasons.append({
            "type": "operating_condition",
            "severity": "high",
            "message": "当前工况不建议直接整定。",
            "evidence": {"condition_label": operating_condition.get("condition_label")},
        })
    elif condition_suitability == "cautious":
        blocking_reasons.append({
            "type": "operating_condition",
            "severity": "medium",
            "message": "当前识别为负荷变化/过渡工况，非硬阻断；建议优先选择稳定片段后再整定。",
            "evidence": {"condition_label": operating_condition.get("condition_label")},
        })
    if identification_score < 0.45:
        evidence_block: dict[str, Any] = {
            "excitation_score": round(excitation_score, 4),
            "response_observability_score": round(response_score, 4),
            "direction_confidence": round(direction_score, 4),
        }
        # 仅在外部确实传了窗口摘要时，把窗口指标作为补充证据；准入校验本身
        # 不依赖它们。
        if has_window_summary:
            evidence_block["best_window_score"] = round(best_window_score, 4)
            evidence_block["usable_window_ratio"] = round(usable_ratio, 4)
        blocking_reasons.append({
            "type": "identification",
            "severity": "medium",
            "message": "可辨识性偏弱，当前历史片段可能不足以稳定拟合模型。",
            "evidence": evidence_block,
        })

    gate_checks = [
        _gate("data_quality", data_quality_score >= 0.65, "high", "数据质量需达标", {"score": round(data_quality_score, 4)}),
        _gate("operating_condition", condition_suitability != "not_recommended", "high", "未命中明显禁整工况；如为谨慎工况，仅作为软提醒。", {"suitability": condition_suitability}),
        _gate("constraints", mv_saturation < 0.2, "high", "MV 不应长时间饱和或贴边", {"mv_saturation_ratio": round(mv_saturation, 6)}),
        _gate("oscillation", not bool(stability.get("oscillation_detected")), "medium", "整定前应先处理明显振荡", {"detected": bool(stability.get("oscillation_detected"))}),
        _gate("identification", identification_score >= 0.45, "medium", "历史激励需支持模型辨识", {"score": round(identification_score, 4)}),
    ]

    next_action = _recommended_action(blocking_reasons, readiness_score, identification_score)
    decision = "ready" if not blocking_reasons and readiness_score >= 0.75 else "caution" if readiness_score >= 0.45 else "blocked"
    recommendations = [_recommendation_text(next_action)]
    if condition_suitability == "cautious":
        recommendations.append("运行工况为谨慎整定：建议人工确认当前是否代表目标负荷区间。")
    if actuator.get("mv_stiction_hint"):
        recommendations.append("执行机构存在疑似长时间不动作/黏滞迹象，必要时先检查阀门。")

    legacy_flags = list((diagnostics.get("flags") or []))
    for item in blocking_reasons:
        legacy_flags.append({"type": item["type"], "severity": item["severity"], "message": item["message"]})

    return {
        "summary": {
            "decision": decision,
            "decision_text": {"ready": "适合整定", "caution": "谨慎整定", "blocked": "暂不建议整定"}[decision],
            "recommended_next_action": next_action,
            "recommended_next_action_text": _recommendation_text(next_action),
        },
        "performance": {
            "score": round(performance_score, 4),
            "level": _level(performance_score),
            "monitoring_score": monitoring.get("overall_score"),
            "stability_score": stability.get("score"),
            "constraint_score": constraints.get("score"),
            "pv_mv_behavior_score": monitoring.get("pv_mv_behavior", {}).get("score"),
        },
        "tuning_readiness": {
            "score": round(readiness_score, 4),
            "level": _level(readiness_score),
            "decision": decision,
            "blocking_reasons": blocking_reasons,
            "gate_checks": gate_checks,
            "recommended_next_action": next_action,
        },
        "identification_suitability": {
            "score": round(identification_score, 4),
            "level": _level(identification_score),
            "excitation_score": round(excitation_score, 4),
            "response_observability_score": response.get("score"),
            "direction_confidence": process_prior.get("process_direction_confidence"),
            # window_summary 现在默认空 dict，下面这些字段会是 0 / None；
            # 前端应该忽略它们，等用户主动启动整定后再看 window_selection 阶段。
            "window_count": int(window_summary.get("window_count") or 0),
            "usable_window_count": int(window_summary.get("usable_window_count") or 0),
            "best_window_score": round(best_window_score, 4) if window_summary.get("window_count") else None,
            "best_window_source": window_summary.get("best_window_source"),
            "windows_evaluated": has_window_summary,
        },
        "operating_condition": operating_condition,
        "data_quality": {
            "score": round(data_quality_score, 4),
            "level": _level(data_quality_score),
            "missing_ratio": round(float(quality_raw.get("missing_ratio_total") or 0.0), 5),
            "continuity_score": round(_status_to_score(data_health.get("status")), 4),
            "noise_score": float(noise.get("score") or 0.0),
            "saturation_score": float(constraints.get("score") or 0.0),
        },
        "identifiability": {
            "score": round(identification_score, 4),
            "level": _level(identification_score),
            "window_count": int(window_summary.get("window_count") or 0),
            "usable_window_count": int(window_summary.get("usable_window_count") or 0),
            "best_window_score": round(best_window_score, 4) if window_summary.get("window_count") else None,
            "best_window_source": window_summary.get("best_window_source") or "",
            "best_window_reasons": window_summary.get("best_window_reasons") or [],
            "windows_evaluated": has_window_summary,
        },
        "diagnostics": {
            **diagnostics,
            "flags": legacy_flags,
        },
        "readiness": {
            "score": round(readiness_score, 4),
            "level": _level(readiness_score),
            "recommendations": recommendations,
        },
    }


@register
class AssessLoopAssessmentSkill(BaseSkill):
    name = "assess_loop_assessment"
    description = "基于 LoopFeatures 和监控快照评估回路是否适合辨识整定，输出性能、可辨识性、整定准备度和下一步动作。"
    input_model = AssessLoopAssessmentInputs

    def run(self, inputs: AssessLoopAssessmentInputs, ctx: LoopContext) -> SkillResult:
        if ctx.cleaned_df is None or ctx.dt is None:
            return SkillResult(success=False, reasoning="未检测到已加载的数据集，请先调用 load_dataset。")
        loop_id = inputs.loop_id or Path(ctx.csv_path).stem
        features = extract_loop_features(
            ctx.cleaned_df,
            loop_id=loop_id,
            loop_type=inputs.loop_type or ctx.loop_type,
            source_file=Path(ctx.csv_path).name,
            sample_time_s=float(ctx.dt),
            loop_name=loop_id,
            tag_prefix=ctx.loop_prefix,
        )
        monitoring = assess_loop_monitoring_from_features(features)
        assessment = assess_loop_assessment_from_features(features, monitoring)
        ctx.data_profile["loop_assessment"] = assessment
        return SkillResult(
            success=True,
            data=assessment,
            warnings=[item["message"] for item in assessment["tuning_readiness"]["blocking_reasons"]],
            reasoning=f"已完成回路评估，结论 {assessment['summary']['decision_text']}，准备度 {assessment['tuning_readiness']['score']:.2f}。",
        )
