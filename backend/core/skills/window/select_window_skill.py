"""Deterministic window selection skill backed by pluggable providers."""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.providers import window as _window_providers  # noqa: F401
from core.shared import provider_registry
from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class SelectWindowInputs(BaseModel):
    provider: str = Field("quality_score_selector", description="选窗 provider 名称")


def _window_summary(w: dict[str, object]) -> dict[str, object]:
    return {
        "source": w.get("window_source", ""),
        "score": float(w.get("window_quality_score", 0.0)),
        "n_points": int(w.get("window_end_idx", 0)) - int(w.get("window_start_idx", 0)),
    }


@register
class SelectWindowSkill(BaseSkill):
    name = "select_window"
    description = "对候选窗口执行确定性选窗，返回选中索引、评分、原因和窗口摘要。"
    input_model = SelectWindowInputs

    def run(self, inputs: SelectWindowInputs, ctx: LoopContext) -> SkillResult:
        if not ctx.candidate_windows:
            return SkillResult(success=False, reasoning="未检测到候选窗口，请先调用 detect_windows。")

        provider = provider_registry.get("window_selection", inputs.provider)
        if provider is None:
            return SkillResult(success=False, reasoning=f"未知选窗 provider: {inputs.provider}")

        result = provider.select(candidate_windows=list(ctx.candidate_windows), context={"ctx": ctx})
        chosen_index = int(result.get("chosen_index", -1))
        if chosen_index < 0 or chosen_index >= len(ctx.candidate_windows):
            return SkillResult(success=False, reasoning="选窗 provider 未返回合法索引。")

        ctx.selected_window_index = chosen_index
        chosen_window = ctx.candidate_windows[chosen_index]
        ctx.data_profile["window_selection"] = {
            "provider": result.get("provider", provider.name),
            "chosen_index": chosen_index,
            "reasoning": result.get("reasoning", ""),
        }
        return SkillResult(
            success=True,
            data={
                "provider": result.get("provider", provider.name),
                "chosen_index": chosen_index,
                "score": float(result.get("score", 0.0)),
                "reasoning": result.get("reasoning", ""),
                "chosen_window_summary": _window_summary(chosen_window),
                "meta": result.get("meta", {}),
            },
            reasoning=f"已用 {provider.name} 选中窗口 {chosen_index}。",
        )
