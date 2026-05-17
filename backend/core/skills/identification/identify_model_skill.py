"""Identification skill backed by pluggable providers."""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.providers import identification as _identification_providers  # noqa: F401
from core.shared import provider_registry
from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class IdentifyModelInputs(BaseModel):
    provider: str = Field("transfer_function_fit", description="系统辨识 provider 名称")
    window_indices: list[int] | None = Field(None, description="指定参与辨识的窗口索引列表")
    use_usable_windows_only: bool = Field(True, description="是否仅使用可用于辨识的窗口")
    model_pool: list[str] | None = Field(None, description="强制模型白名单")
    hint_L: float | None = Field(None, description="强制死区初值提示")


@register
class IdentifyModelSkill(BaseSkill):
    name = "identify_model"
    description = "对一个或多个候选窗口执行系统辨识，输出最佳模型、attempts、拟合分和置信度。"
    input_model = IdentifyModelInputs
    risk_level = "high"
    preconditions = ["cleaned_df", "dt", "candidate_windows"]
    effects = [
        {"key": "model", "description": "辨识模型"},
        {"key": "confidence", "description": "模型置信度"},
        {"key": "data_profile.identification", "description": "辨识摘要"},
    ]
    stage = "identification"
    deterministic_gate = True

    def run(self, inputs: IdentifyModelInputs, ctx: LoopContext) -> SkillResult:
        if ctx.cleaned_df is None or ctx.dt is None:
            return SkillResult(success=False, reasoning="未检测到已加载的数据集，请先调用 load_dataset。")
        if not ctx.candidate_windows:
            return SkillResult(success=False, reasoning="未检测到候选窗口，请先调用 detect_windows。")

        provider = provider_registry.get("identification", inputs.provider)
        if provider is None:
            return SkillResult(success=False, reasoning=f"未知系统辨识 provider: {inputs.provider}")

        windows = list(ctx.candidate_windows)
        if inputs.use_usable_windows_only:
            usable = [w for w in windows if w.get("window_usable_for_id")]
            if usable:
                windows = usable

        if inputs.window_indices:
            selected = [ctx.candidate_windows[idx] for idx in inputs.window_indices if 0 <= idx < len(ctx.candidate_windows)]
            if selected:
                windows = selected

        result = provider.identify(
            cleaned_df=ctx.cleaned_df,
            candidate_windows=windows,
            actual_dt=ctx.dt,
            loop_type=ctx.loop_type,
            quality_metrics=ctx.data_profile,
            force_model_types=inputs.model_pool,
            force_l_hint=inputs.hint_L,
            context={"ctx": ctx},
        )
        best_model = dict(result.get("best_model", {}))
        if not best_model and result.get("model") is not None:
            model_obj = result.get("model")
            if hasattr(model_obj, "model_dump"):
                best_model = model_obj.model_dump()
            elif isinstance(model_obj, dict):
                best_model = dict(model_obj)
        if best_model and "confidence" not in best_model:
            conf_obj = result.get("confidence")
            if hasattr(conf_obj, "confidence"):
                best_model["confidence"] = float(conf_obj.confidence)
            elif isinstance(conf_obj, dict) and "confidence" in conf_obj:
                best_model["confidence"] = float(conf_obj["confidence"])
        if best_model:
            ctx.model = best_model
            ctx.confidence = float(best_model.get("confidence", 0.0))
        ctx.data_profile["identification"] = {
            "provider": result.get("provider", provider.name),
            "attempt_count": len(result.get("attempts", [])),
        }
        return SkillResult(
            success=True,
            data={
                "provider": result.get("provider", provider.name),
                "best_model": best_model,
                "attempts": result.get("attempts", []),
                "window_source": result.get("window_source", ""),
                "selection_reason": result.get("selection_reason", ""),
                "fit_preview": result.get("fit_preview", {}),
                "candidates": result.get("candidates", []),
                "meta": {
                    "window_count": len(windows),
                    "attempt_count": len(result.get("attempts", [])),
                },
            },
            reasoning=f"已用 {provider.name} 完成系统辨识，尝试 {len(result.get('attempts', []))} 次。",
        )
