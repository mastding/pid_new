"""Evaluation skill backed by pluggable providers."""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.providers import evaluation as _evaluation_providers  # noqa: F401
from core.shared import provider_registry
from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class EvaluateTuningInputs(BaseModel):
    provider: str = Field("closed_loop_sim", description="评估 provider 名称")
    tuning_unreliable: bool = Field(False, description="是否已标记整定不可靠")
    tuning_unreliable_reason: str = Field("", description="整定不可靠原因")


@register
class EvaluateTuningSkill(BaseSkill):
    name = "evaluate_tuning"
    description = "对推荐 PID 参数执行闭环评估，返回性能分、综合分、就绪分和约束说明。"
    input_model = EvaluateTuningInputs

    def run(self, inputs: EvaluateTuningInputs, ctx: LoopContext) -> SkillResult:
        if not ctx.model:
            return SkillResult(success=False, reasoning="未检测到辨识模型，请先调用 identify_model。")
        if not ctx.pid_params:
            return SkillResult(success=False, reasoning="未检测到 PID 参数，请先调用 generate_tuning_candidates。")
        if ctx.dt is None:
            return SkillResult(success=False, reasoning="未检测到采样周期，请先调用 load_dataset。")

        provider = provider_registry.get("evaluation", inputs.provider)
        if provider is None:
            return SkillResult(success=False, reasoning=f"未知评估 provider: {inputs.provider}")

        model_type = str(ctx.model.get("model_type", "FOPDT"))
        model_params = dict(ctx.model)
        T = float(ctx.model.get("T1", ctx.model.get("T", 0.0)) or 0.0)
        result = provider.evaluate(
            Kp=float(ctx.pid_params.get("Kp", 0.0)),
            Ki=float(ctx.pid_params.get("Ki", 0.0)),
            Kd=float(ctx.pid_params.get("Kd", 0.0)),
            model_type=model_type,
            model_params=model_params,
            K=float(ctx.model.get("K", 0.0)),
            T=T,
            L=float(ctx.model.get("L", 0.0)),
            dt=ctx.dt,
            loop_type=ctx.loop_type,
            confidence=float(ctx.confidence or 0.0),
            tuning_unreliable=inputs.tuning_unreliable,
            tuning_unreliable_reason=inputs.tuning_unreliable_reason,
            context={"ctx": ctx},
        )
        ctx.evaluation = dict(result)
        ctx.data_profile["evaluation"] = {
            "provider": result.get("provider", provider.name),
            "performance_score": result.get("performance_score", 0.0),
            "final_rating": result.get("final_rating", 0.0),
            "readiness_score": result.get("readiness_score", 0.0),
        }
        return SkillResult(
            success=True,
            data=result,
            reasoning=f"已用 {provider.name} 完成 PID 评估，综合分 {float(result.get('final_rating', 0.0)):.2f}。",
        )
