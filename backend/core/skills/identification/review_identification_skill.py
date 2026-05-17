"""Skill: review identification model reliability."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.pipeline.identification_advisor import review_identification_via_llm
from core.pipeline.identification_refinement_advisor import ask_refinement_via_llm
from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class ReviewIdentificationInputs(BaseModel):
    chosen_window_summary: dict[str, Any] = Field(default_factory=dict)
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    windows_summary: list[dict[str, Any]] = Field(default_factory=list)
    algorithm_comparison: list[dict[str, Any]] = Field(default_factory=list)
    history_summary: list[dict[str, Any]] = Field(default_factory=list)
    round_idx: int = Field(0, ge=0)
    max_refinement_rounds: int = Field(2, ge=0, le=5)
    allow_retry_plan: bool = Field(True)
    use_llm_advisor: bool = Field(True)


@register
class ReviewIdentificationSkill(BaseSkill):
    name = "review_identification"
    description = "评审辨识模型是否可用于整定，并在降级时给出可选的下一轮辨识重试计划。"
    input_model = ReviewIdentificationInputs
    risk_level = "high"
    preconditions = ["model"]
    effects = [
        {"key": "data_profile.model_review", "description": "模型可靠性评审结论"},
        {"key": "data_profile.identification_refinement_plan", "description": "可选的辨识重试计划"},
    ]
    stage = "model_review"
    deterministic_gate = True

    def run(self, inputs: ReviewIdentificationInputs, ctx: LoopContext) -> SkillResult:
        model = dict(ctx.model or {})
        confidence = float(ctx.confidence if ctx.confidence is not None else model.get("confidence", 0.0) or 0.0)
        best_for_review = {
            "model_type": model.get("model_type", "FOPDT"),
            "K": model.get("K", 0.0),
            "T": model.get("T", model.get("T1", 0.0)),
            "T1": model.get("T1", model.get("T", 0.0)),
            "T2": model.get("T2", 0.0),
            "L": model.get("L", 0.0),
            "r2_score": model.get("r2_score", 0.0),
            "normalized_rmse": model.get("normalized_rmse", 0.0),
            "window_source": model.get("window_source", ""),
        }

        if not inputs.use_llm_advisor:
            verdict = "accept" if confidence >= 0.5 else "downgrade"
            review = {
                "available": False,
                "verdict": verdict,
                "reason": "LLM 评审关闭，按模型置信度执行确定性评审。",
                "concerns": [] if verdict == "accept" else [f"模型置信度 {confidence:.2f} 偏低"],
                "fallback": True,
            }
        else:
            review = review_identification_via_llm(
                loop_type=ctx.loop_type,
                data_profile=ctx.data_profile,
                chosen_window_summary=inputs.chosen_window_summary,
                best_model=best_for_review,
                attempts=inputs.attempts,
                confidence=confidence,
            )
            if not review.get("available", True):
                review = {
                    "available": False,
                    "verdict": "accept",
                    "reason": "LLM 评审不可用，默认采纳算法模型，但保留 fallback 标记。",
                    "concerns": [],
                    "fallback": True,
                    "error_type": review.get("error_type"),
                    "error_message": review.get("error_message"),
                    "raw_text": review.get("raw_text", ""),
                }

        retry_plan = None
        if (
            inputs.allow_retry_plan
            and inputs.use_llm_advisor
            and review.get("verdict") == "downgrade"
            and inputs.round_idx < inputs.max_refinement_rounds
        ):
            retry_plan = ask_refinement_via_llm(
                loop_type=ctx.loop_type,
                round_idx=inputs.round_idx + 1,
                max_rounds=inputs.max_refinement_rounds,
                data_profile=ctx.data_profile,
                windows_summary=inputs.windows_summary,
                algorithm_comparison=inputs.algorithm_comparison,
                last_best=best_for_review,
                last_attempts=inputs.attempts,
                last_review=review,
                history_summary=inputs.history_summary,
            )

        usable_for_tuning = review.get("verdict") == "accept" and confidence >= 0.5
        result = {
            **review,
            "usable_for_tuning": usable_for_tuning,
            "usable_for_diagnosis": True,
            "retry_plan": retry_plan,
            "score_caps": {
                "max_performance_score": 10 if usable_for_tuning else 5,
                "max_readiness_score": 10 if usable_for_tuning else 5,
            },
        }
        ctx.data_profile["model_review"] = result
        if retry_plan:
            ctx.data_profile["identification_refinement_plan"] = retry_plan
        return SkillResult(success=True, data=result, reasoning=str(result.get("reason") or "模型评审完成。"))
