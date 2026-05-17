"""Skill: load and clean a process dataset via pluggable providers."""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.providers import common as _common_providers  # noqa: F401
from core.shared import provider_registry
from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class LoadDatasetInputs(BaseModel):
    provider: str = Field("clean_csv_loader", description="数据加载 provider 名称")
    loop_prefix: str | None = Field(None, description="可选回路前缀")
    start_time: str | None = Field(None, description="可选起始时间")
    end_time: str | None = Field(None, description="可选结束时间")


@register
class LoadDatasetSkill(BaseSkill):
    name = "load_dataset"
    description = (
        "加载并清洗过程数据文件，识别统一列名、时间戳和采样周期，"
        "并把清洗后的数据写入会话上下文供后续技能复用。"
    )
    input_model = LoadDatasetInputs
    risk_level = "low"
    preconditions = []
    effects = [
        {"key": "cleaned_df", "description": "清洗后的历史数据"},
        {"key": "dt", "description": "采样周期"},
    ]
    stage = "data_analysis"

    def run(self, inputs: LoadDatasetInputs, ctx: LoopContext) -> SkillResult:
        provider = provider_registry.get("dataset_loading", inputs.provider)
        if provider is None:
            return SkillResult(success=False, reasoning=f"未知数据加载 provider: {inputs.provider}")

        result = provider.load(
            csv_path=ctx.csv_path,
            selected_loop_prefix=inputs.loop_prefix or ctx.loop_prefix or None,
            start_time=inputs.start_time,
            end_time=inputs.end_time,
            context={"ctx": ctx},
        )

        cleaned = result["cleaned_df"]
        dt = float(result["dt"])
        ctx.cleaned_df = cleaned
        ctx.dt = dt
        if inputs.loop_prefix:
            ctx.loop_prefix = inputs.loop_prefix

        warnings: list[str] = []
        if len(cleaned) < 100:
            warnings.append(f"数据样本偏少（{len(cleaned)} 点），可能影响辨识质量")
        if "SV" not in cleaned.columns:
            warnings.append("未发现 SV 列，无法基于设定值阶跃来定位窗口")

        return SkillResult(
            success=True,
            data={
                "provider": result.get("provider", provider.name),
                "data_points": int(result.get("data_points", len(cleaned))),
                "sampling_time": round(dt, 4),
                "time_span_sec": round(float(result.get("meta", {}).get("time_span_sec", dt * len(cleaned))), 1),
                "columns": list(result.get("columns", [])),
                "loops_in_csv": list(result.get("loops_in_csv", [])),
                "selected_loop_prefix": ctx.loop_prefix or "",
            },
            warnings=warnings,
            reasoning=f"已加载 {len(cleaned)} 行数据，采样周期 {dt:.3f}s",
        )
