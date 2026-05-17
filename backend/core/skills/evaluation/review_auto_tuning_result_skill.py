"""Skill: review an auto tuning result before engineer adoption."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class ReviewAutoTuningResultInputs(BaseModel):
    evaluation: dict[str, Any] = Field(default_factory=dict)
    pid_params: dict[str, Any] = Field(default_factory=dict)
    ontology_context: dict[str, Any] = Field(default_factory=dict)


def review_auto_tuning_result(
    *,
    evaluation: dict[str, Any],
    pid_params: dict[str, Any] | None = None,
    ontology_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_rating = evaluation.get("final_rating")
    performance_score = evaluation.get("performance_score")
    robustness_score = evaluation.get("robustness_score")
    mv_saturation_pct = evaluation.get("mv_saturation_pct")
    overshoot_percent = evaluation.get("overshoot_percent")
    passed = bool(evaluation.get("passed"))
    warnings: list[str] = []
    required_confirmations = ["engineer_review", "simulation_curve_review"]

    if not passed:
        warnings.append("仿真评估未通过，不能直接采纳推荐参数。")
    if isinstance(final_rating, (int, float)) and final_rating < 7.0:
        warnings.append("综合评分低于 7.0，需要重新选窗、辨识或降低整定激进度。")
    if isinstance(robustness_score, (int, float)) and robustness_score < 0.65:
        warnings.append("鲁棒性评分偏低，需要查看最差场景包络线。")
    if isinstance(mv_saturation_pct, (int, float)) and mv_saturation_pct > 5:
        warnings.append("MV 饱和比例偏高，需要核对执行机构余量和输出限幅。")
    if isinstance(overshoot_percent, (int, float)) and overshoot_percent > 20:
        warnings.append("超调偏大，需要保守化 PID 或重新生成仿真场景。")
    missing_fields = list((ontology_context or {}).get("missing_fields") or [])
    if missing_fields:
        warnings.append("本体上下文字段不完整，采纳前需要补齐关键工况/约束信息。")
        required_confirmations.append("ontology_context_review")

    can_adopt = passed and not warnings
    if passed and warnings:
        decision = "conditional_review"
    elif can_adopt:
        decision = "ready_for_engineer_confirmation"
    else:
        decision = "revise_required"

    return {
        "decision": decision,
        "can_adopt": can_adopt,
        "required_confirmations": required_confirmations,
        "warnings": warnings,
        "summary": (
            "仿真通过且未发现额外风险，可进入工程师确认。"
            if can_adopt else
            "仿真存在需复核项，建议先查看曲线、本体约束和最差场景，再决定是否重试整定。"
            if warnings else
            "仿真未提供足够证据，需要人工复核。"
        ),
        "scores": {
            "final_rating": final_rating,
            "performance_score": performance_score,
            "robustness_score": robustness_score,
        },
        "pid_params_present": bool(pid_params),
    }


@register
class ReviewAutoTuningResultSkill(BaseSkill):
    name = "review_auto_tuning_result"
    description = "对自动整定结果进行上线前复核，判断是否可进入工程师确认、是否需要重试或补充证据。"
    input_model = ReviewAutoTuningResultInputs
    risk_level = "high"
    preconditions = ["evaluation", "pid_params"]
    effects = [{"key": "auto_tuning_review", "description": "自动整定结果复核结论"}]
    stage = "result_review"
    deterministic_gate = True

    def run(self, inputs: ReviewAutoTuningResultInputs, ctx: LoopContext) -> SkillResult:
        review = review_auto_tuning_result(
            evaluation=inputs.evaluation,
            pid_params=inputs.pid_params,
            ontology_context=inputs.ontology_context,
        )
        return SkillResult(
            success=True,
            data={"auto_tuning_review": review},
            warnings=list(review.get("warnings") or []),
            reasoning=f"自动整定结果复核结论: {review['decision']}。",
        )
