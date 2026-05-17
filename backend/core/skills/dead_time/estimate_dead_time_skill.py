"""Dead-time estimation skill backed by pluggable providers."""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.providers import dead_time as _dead_time_providers  # noqa: F401
from core.shared import provider_registry
from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class EstimateDeadTimeInputs(BaseModel):
    provider: str = Field("cross_correlation", description="死区估计 provider 名称")
    window_index: int | None = Field(None, description="候选窗口索引，默认使用已选窗口或第一个窗口")


@register
class EstimateDeadTimeSkill(BaseSkill):
    name = "estimate_dead_time"
    description = "对指定窗口估计死区时间 L，并返回估计值、置信度和使用的 provider。"
    input_model = EstimateDeadTimeInputs
    stage = "identification"
    risk_level = "medium"
    preconditions = ["cleaned_df", "dt", "candidate_windows"]
    effects = [
        {"key": "data_profile.dead_time", "description": "候选窗口死区时间估计"},
    ]

    def run(self, inputs: EstimateDeadTimeInputs, ctx: LoopContext) -> SkillResult:
        if ctx.cleaned_df is None or ctx.dt is None:
            return SkillResult(success=False, reasoning="未检测到已加载的数据集，请先调用 load_dataset。")
        if not ctx.candidate_windows:
            return SkillResult(success=False, reasoning="未检测到候选窗口，请先调用 detect_windows。")

        provider = provider_registry.get("dead_time", inputs.provider)
        if provider is None:
            return SkillResult(success=False, reasoning=f"未知死区估计 provider: {inputs.provider}")

        idx = inputs.window_index
        if idx is None:
            idx = ctx.selected_window_index if ctx.selected_window_index is not None else 0
        if idx < 0 or idx >= len(ctx.candidate_windows):
            return SkillResult(success=False, reasoning=f"窗口索引越界: {idx}")

        window = ctx.candidate_windows[idx]
        start = int(window.get("window_start_idx", 0))
        end = int(window.get("window_end_idx", len(ctx.cleaned_df)))
        seg = ctx.cleaned_df.iloc[start:end].reset_index(drop=True)
        result = provider.estimate(
            mv=seg["MV"].to_numpy(dtype=float),
            pv=seg["PV"].to_numpy(dtype=float),
            dt=ctx.dt,
            loop_type=ctx.loop_type,
            context={"ctx": ctx, "window": window},
        )
        ctx.data_profile["dead_time"] = {
            "provider": result.get("provider", provider.name),
            "window_index": idx,
            "L": result.get("L", 0.0),
            "confidence": result.get("confidence", 0.0),
        }
        return SkillResult(
            success=True,
            data={
                "provider": result.get("provider", provider.name),
                "window_index": idx,
                "L": float(result.get("L", 0.0)),
                "confidence": float(result.get("confidence", 0.0)),
                "meta": result.get("meta", {}),
            },
            reasoning=f"已用 {provider.name} 在窗口 {idx} 上估计死区 L={float(result.get('L', 0.0)):.2f}s。",
        )
