"""PID tuning candidate generation skill backed by pluggable providers."""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.providers import tuning as _tuning_providers  # noqa: F401
from core.shared import provider_registry
from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class GenerateTuningCandidatesInputs(BaseModel):
    provider: str = Field("classic_family", description="PID 整定 provider 名称")


@register
class GenerateTuningCandidatesSkill(BaseSkill):
    name = "generate_tuning_candidates"
    description = "根据辨识模型生成 PID 候选参数，返回推荐策略、所有候选以及整定可靠性标记。"
    input_model = GenerateTuningCandidatesInputs
    risk_level = "high"
    preconditions = ["model", "dt"]
    effects = [
        {"key": "pid_params", "description": "推荐 PID 参数"},
        {"key": "data_profile.tuning_candidates", "description": "PID 候选摘要"},
    ]
    stage = "tuning"
    deterministic_gate = True

    def run(self, inputs: GenerateTuningCandidatesInputs, ctx: LoopContext) -> SkillResult:
        if not ctx.model:
            return SkillResult(success=False, reasoning="未检测到辨识模型，请先调用 identify_model。")
        if ctx.dt is None:
            return SkillResult(success=False, reasoning="未检测到采样周期，请先调用 load_dataset。")

        provider = provider_registry.get("tuning", inputs.provider)
        if provider is None:
            return SkillResult(success=False, reasoning=f"未知 PID 整定 provider: {inputs.provider}")

        model_type = str(ctx.model.get("model_type", "FOPDT"))
        model_params = dict(ctx.model)
        T = float(ctx.model.get("T1", ctx.model.get("T", 0.0)) or 0.0)
        L = float(ctx.model.get("L", 0.0) or 0.0)
        result = provider.tune(
            K=float(ctx.model.get("K", 0.0)),
            T=T,
            L=L,
            dt=ctx.dt,
            loop_type=ctx.loop_type,
            model_type=model_type,
            model_params=model_params,
            confidence=float(ctx.confidence or 0.0),
            nrmse=float(ctx.model.get("normalized_rmse", 0.0) or 0.0),
            r2=float(ctx.model.get("r2_score", 0.0) or 0.0),
            context={"ctx": ctx},
        )
        best = dict(result.get("best") or {})
        if best:
            ctx.pid_params = best
        ctx.data_profile["tuning_candidates"] = {
            "provider": result.get("provider", provider.name),
            "candidate_count": len(result.get("all_candidates", [])),
            "heuristic_strategy": result.get("heuristic_strategy", ""),
        }
        return SkillResult(
            success=True,
            data={
                "provider": result.get("provider", provider.name),
                "recommended": best,
                "candidates": result.get("all_candidates", []),
                "heuristic_strategy": result.get("heuristic_strategy", ""),
                "heuristic_reason": result.get("heuristic_reason", ""),
                "tuning_unreliable": bool(result.get("tuning_unreliable", False)),
                "tuning_unreliable_reason": result.get("tuning_unreliable_reason", ""),
            },
            reasoning=f"已用 {provider.name} 生成 {len(result.get('all_candidates', []))} 个 PID 候选策略。",
        )
