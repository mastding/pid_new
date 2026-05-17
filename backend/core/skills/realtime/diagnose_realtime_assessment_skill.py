"""Skill: diagnose realtime loop assessment root causes."""
from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register
from core.skills.realtime.common import level_from_score, metric_value


class DiagnoseRealtimeAssessmentInputs(BaseModel):
    assessment: dict[str, Any] = Field(default_factory=dict)
    monitoring: dict[str, Any] = Field(default_factory=dict)
    harris_metric: dict[str, Any] | None = None
    cpk_metric: dict[str, Any] | None = None
    ontology: dict[str, Any] = Field(default_factory=dict)


def diagnose_realtime_assessment(
    *,
    assessment: dict[str, Any],
    monitoring: dict[str, Any],
    harris_metric: dict[str, Any] | None,
    cpk_metric: dict[str, Any] | None,
    ontology: dict[str, Any],
) -> list[dict[str, Any]]:
    snapshot = monitoring.get("monitoring") or {}
    readiness = assessment.get("tuning_readiness") or {}
    diagnostics = assessment.get("diagnostics") or {}
    flags = list(diagnostics.get("flags") or [])
    results: list[dict[str, Any]] = []

    def add(root: str, confidence: float, severity: str, evidence: list[Any], action: str) -> None:
        results.append({
            "diagnosis_id": f"diag_{root}_{uuid.uuid4().hex[:8]}",
            "root_cause": root,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "severity": severity,
            "evidence": evidence,
            "action": action,
        })

    data_score = metric_value(snapshot, "data_health", "score")
    if isinstance(data_score, (int, float)) and data_score < 0.68:
        add(
            "data_quality",
            1.0 - float(data_score),
            level_from_score(float(data_score)),
            [snapshot.get("data_health")],
            "先处理缺失、采样异常、尖峰或测量噪声，再解释整定指标。",
        )

    constraint_score = metric_value(snapshot, "constraints", "score")
    mv_sat = metric_value(snapshot, "constraints", "mv_saturation_ratio")
    if isinstance(constraint_score, (int, float)) and constraint_score < 0.75:
        add(
            "actuator_or_constraint",
            1.0 - float(constraint_score),
            level_from_score(float(constraint_score)),
            [snapshot.get("constraints")],
            "优先确认 MV 饱和、阀位能力和执行机构余量，避免把约束问题误判为 PID 参数问题。",
        )

    if bool(metric_value(snapshot, "stability", "oscillation_detected")):
        add(
            "oscillation_or_disturbance",
            float(metric_value(snapshot, "stability", "oscillation_confidence") or 0.65),
            "medium",
            [snapshot.get("stability")],
            "进入振荡诊断，区分 PID 过激、阀门问题和外部周期扰动。",
        )

    harris_value = harris_metric.get("value") if harris_metric else None
    blocking_types = {str(item.get("type")) for item in readiness.get("blocking_reasons") or []}
    if (
        ("data_quality" not in blocking_types)
        and (not isinstance(mv_sat, (int, float)) or float(mv_sat) < 0.1)
        and ((isinstance(harris_value, (int, float)) and harris_value < 0.6) or readiness.get("decision") == "ready")
    ):
        add(
            "pid_parameters",
            0.72 if isinstance(harris_value, (int, float)) and harris_value < 0.6 else 0.55,
            "medium",
            [
                {"harris": harris_metric},
                {"cpk": cpk_metric},
                {"ontology_case_id": ontology.get("case_id")},
                {"assessment_summary": assessment.get("summary")},
            ],
            "满足数据和约束准入后，可进入保守整定流程并通过仿真评估推荐参数。",
        )

    for flag in flags[:3]:
        root = str(flag.get("type") or "assessment_flag")
        if any(item["root_cause"] == root for item in results):
            continue
        add(
            root,
            0.5,
            str(flag.get("severity") or "medium"),
            [flag],
            str(flag.get("message") or "按评估标记继续人工复核。"),
        )

    if not results:
        add(
            "no_dominant_fault",
            0.45,
            "low",
            [{"assessment": assessment.get("summary")}, {"ontology_missing": ontology.get("missing_fields")}],
            "当前没有明确单一根因，建议持续监控或补充本体/规格限后复评。",
        )

    return sorted(results, key=lambda item: (item["severity"] != "high", -float(item["confidence"])))


@register
class DiagnoseRealtimeAssessmentSkill(BaseSkill):
    name = "diagnose_realtime_assessment"
    description = "结合监控快照、Harris/CPK 指标和本体上下文，诊断实时评估中的主要根因。"
    input_model = DiagnoseRealtimeAssessmentInputs
    risk_level = "medium"
    preconditions = ["assessment", "monitoring", "ontology"]
    effects = [{"key": "diagnosis", "description": "实时评估根因诊断结果"}]
    stage = "diagnosis"
    deterministic_gate = True

    def run(self, inputs: DiagnoseRealtimeAssessmentInputs, ctx: LoopContext) -> SkillResult:
        diagnosis = diagnose_realtime_assessment(
            assessment=inputs.assessment,
            monitoring=inputs.monitoring,
            harris_metric=inputs.harris_metric,
            cpk_metric=inputs.cpk_metric,
            ontology=inputs.ontology,
        )
        return SkillResult(
            success=True,
            data={"diagnosis": diagnosis},
            reasoning=f"已生成 {len(diagnosis)} 条实时评估根因诊断。",
        )
