"""Skill: decide realtime tuning action from assessment evidence."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register
from core.skills.realtime.common import level_from_score, max_level, metric_level_to_risk


class DecideRealtimeTuningActionInputs(BaseModel):
    monitoring: dict[str, Any] = Field(default_factory=dict)
    assessment: dict[str, Any] = Field(default_factory=dict)
    diagnosis: list[dict[str, Any]] = Field(default_factory=list)
    harris_metric: dict[str, Any] | None = None
    cpk_metric: dict[str, Any] | None = None


def decide_realtime_tuning_action(
    *,
    monitoring: dict[str, Any],
    assessment: dict[str, Any],
    diagnosis: list[dict[str, Any]],
    harris_metric: dict[str, Any] | None,
    cpk_metric: dict[str, Any] | None,
) -> dict[str, Any]:
    mon = monitoring.get("monitoring") or {}
    assessment_decision = (assessment.get("summary") or {}).get("decision") or "unknown"
    readiness = assessment.get("tuning_readiness") or {}
    score = mon.get("overall_score") or (assessment.get("performance") or {}).get("score")
    metric_risks = [
        metric_level_to_risk("harris", harris_metric.get("level"), harris_metric.get("value")) if harris_metric else "normal",
        metric_level_to_risk("cpk", cpk_metric.get("level"), cpk_metric.get("value")) if cpk_metric else "normal",
    ]
    risk_level = max_level(
        level_from_score(float(score)) if isinstance(score, (int, float)) else "potential",
        *(str(item.get("severity") or "normal") for item in diagnosis[:2]),
        *metric_risks,
    )
    blocked = assessment_decision == "blocked" or bool(readiness.get("blocking_reasons") and risk_level == "high")
    need_tuning = (not blocked) and any(
        item.get("root_cause") == "pid_parameters" and item.get("confidence", 0) >= 0.55
        for item in diagnosis
    )
    decision = {
        "decision": "blocked" if blocked else "tuning_recommended" if need_tuning else "observe",
        "need_tuning": need_tuning,
        "blocked": blocked,
        "priority": "high" if risk_level == "high" else "medium" if risk_level == "medium" else "low",
        "reason_codes": [item.get("root_cause") for item in diagnosis[:4]],
        "required_confirmations": ["engineer_review"] if need_tuning else [],
        "summary": (
            "建议进入整定准备流程，需工程师确认后创建整定任务。"
            if need_tuning else
            "当前存在阻断项，先处理数据质量、工况或执行机构问题。"
            if blocked else
            "当前以持续监控和补充证据为主。"
        ),
    }
    return {"risk_level": risk_level, "score": score, "decision": decision}


@register
class DecideRealtimeTuningActionSkill(BaseSkill):
    name = "decide_realtime_tuning_action"
    description = "基于实时评估根因、Harris/CPK 和准入结果，判断是否建议进入整定流程。"
    input_model = DecideRealtimeTuningActionInputs
    risk_level = "high"
    preconditions = ["assessment", "monitoring", "diagnosis"]
    effects = [{"key": "decision", "description": "实时评估后的整定动作建议"}]
    stage = "decision"
    deterministic_gate = True

    def run(self, inputs: DecideRealtimeTuningActionInputs, ctx: LoopContext) -> SkillResult:
        action = decide_realtime_tuning_action(
            monitoring=inputs.monitoring,
            assessment=inputs.assessment,
            diagnosis=inputs.diagnosis,
            harris_metric=inputs.harris_metric,
            cpk_metric=inputs.cpk_metric,
        )
        return SkillResult(
            success=True,
            data=action,
            reasoning=f"实时评估动作结论: {action['decision']['decision']}。",
        )
