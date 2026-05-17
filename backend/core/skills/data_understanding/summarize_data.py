"""Skill: build a reusable deterministic data profile via provider."""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.providers import common as _common_providers  # noqa: F401
from core.shared import provider_registry
from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class SummarizeDataInputs(BaseModel):
    provider: str = Field("deterministic_profile", description="数据画像 provider 名称")


@register
class SummarizeDataSkill(BaseSkill):
    name = "summarize_data"
    description = (
        "生成确定性数据画像，覆盖 PV/MV 量程、饱和、噪声、死区、振荡和扰动等维度，"
        "并把画像缓存到会话上下文供后续选窗、辨识和评审复用。"
    )
    input_model = SummarizeDataInputs
    risk_level = "low"
    preconditions = ["cleaned_df", "dt"]
    effects = [{"key": "data_profile", "description": "确定性数据画像"}]
    stage = "data_analysis"

    def run(self, inputs: SummarizeDataInputs, ctx: LoopContext) -> SkillResult:
        if ctx.cleaned_df is None or ctx.dt is None:
            return SkillResult(success=False, reasoning="未检测到已加载的数据集，请先调用 load_dataset。")

        provider = provider_registry.get("data_profile", inputs.provider)
        if provider is None:
            return SkillResult(success=False, reasoning=f"未知数据画像 provider: {inputs.provider}")

        result = provider.summarize(
            df=ctx.cleaned_df,
            dt=float(ctx.dt),
            loop_type=getattr(ctx, "loop_type", None),
            context={"ctx": ctx},
        )
        profile = dict(result["profile"])
        ctx.data_profile = profile

        return SkillResult(
            success=True,
            data=profile,
            warnings=list(result.get("warnings", [])),
            reasoning=str(result.get("reasoning", profile.get("text_summary", ""))),
        )
