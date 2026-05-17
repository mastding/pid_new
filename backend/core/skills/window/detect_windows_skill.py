"""Window detection skill backed by pluggable providers."""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.providers import window as _window_providers  # noqa: F401
from core.shared import provider_registry
from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class DetectWindowsInputs(BaseModel):
    provider: str = Field("policy_composite", description="窗口检测 provider 名称")
    loop_type: str | None = Field(None, description="回路类型覆盖值")
    max_windows: int = Field(8, ge=1, le=50, description="最多返回多少个窗口摘要")
    include_unusable: bool = Field(True, description="是否保留不可用于辨识的窗口")
    policy: dict | None = Field(None, description="本体/LLM 生成的窗口算法族策略")


def _summarize_window(w: dict, idx: int) -> dict:
    return {
        "index": idx,
        "start": int(w.get("window_start_idx", 0)),
        "end": int(w.get("window_end_idx", 0)),
        "n_points": int(w.get("window_end_idx", 0)) - int(w.get("window_start_idx", 0)),
        "type": w.get("type", ""),
        "algorithm": w.get("window_algorithm", ""),
        "algorithm_label": w.get("window_algorithm_label", ""),
        "algorithm_family": w.get("window_algorithm_family", w.get("window_algorithm", "")),
        "algorithm_plan_state": w.get("window_algorithm_plan_state", ""),
        "algorithm_plan_reason": w.get("window_algorithm_plan_reason", ""),
        "score": round(float(w.get("window_quality_score", 0.0)), 4),
        "usable": bool(w.get("window_usable_for_id", False)),
        "source": w.get("window_source", ""),
        "selection_basis": w.get("window_selection_basis", ""),
        "score_breakdown": w.get("window_score_breakdown", {}),
        "mv_span": round(float(w.get("window_mv_span", 0.0)), 4),
        "pv_span": round(float(w.get("window_pv_span", 0.0)), 4),
        "corr": round(float(w.get("window_corr", 0.0)), 4),
    }


@register
class DetectWindowsSkill(BaseSkill):
    name = "detect_windows"
    description = "在已加载的清洗数据上检测候选辨识窗口，并返回可用窗口、质量分与窗口摘要。"
    input_model = DetectWindowsInputs
    risk_level = "medium"
    preconditions = ["cleaned_df", "dt"]
    effects = [
        {"key": "candidate_windows", "description": "候选辨识窗口"},
        {"key": "data_profile.window_detection", "description": "窗口检测摘要"},
    ]
    stage = "window_selection"
    deterministic_gate = True

    def run(self, inputs: DetectWindowsInputs, ctx: LoopContext) -> SkillResult:
        if ctx.cleaned_df is None or ctx.dt is None:
            return SkillResult(success=False, reasoning="未检测到已加载的数据集，请先调用 load_dataset。")

        provider = provider_registry.get("window_detection", inputs.provider)
        if provider is None:
            return SkillResult(success=False, reasoning=f"未知窗口检测 provider: {inputs.provider}")

        loop_type = inputs.loop_type or ctx.loop_type
        result = provider.detect(
            df=ctx.cleaned_df,
            dt=ctx.dt,
            loop_type=loop_type,
            context={"ctx": ctx, "policy": inputs.policy},
        )
        windows = list(result.get("candidate_windows", []))
        if not inputs.include_unusable:
            windows = [w for w in windows if w.get("window_usable_for_id")]

        ctx.candidate_windows = windows
        ctx.data_profile["window_detection"] = {
            "provider": result.get("provider", provider.name),
            "candidate_count": len(windows),
            "step_event_count": len(result.get("step_events", [])),
        }

        summaries = [_summarize_window(w, i) for i, w in enumerate(windows[: inputs.max_windows])]
        usable_count = sum(1 for w in windows if w.get("window_usable_for_id"))
        return SkillResult(
            success=True,
            data={
                "provider": result.get("provider", provider.name),
                "candidate_count": len(windows),
                "usable_count": usable_count,
                "step_event_count": len(result.get("step_events", [])),
                "windows": summaries,
                "meta": result.get("meta", {}),
            },
            reasoning=f"已用 {provider.name} 检测到 {len(windows)} 个候选窗口，其中 {usable_count} 个可用于辨识。",
        )
