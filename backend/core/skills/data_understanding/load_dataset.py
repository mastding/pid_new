"""技能：加载并清洗数据集。

LLM 在整定流程的第一步必调。完成 CSV 读取、列归一化、时间戳解析、
数值清洗、PV 去噪，并把 cleaned_df / dt 写入 LoopContext。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.algorithms.data_analysis import _detect_loops, _load_clean_only
from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class LoadDatasetInputs(BaseModel):
    loop_prefix: str | None = Field(
        None,
        description=(
            "回路前缀（如 'L1_'）。CSV 含多个回路时用于选择目标回路；"
            "单回路或列名无前缀时传 null。"
        ),
    )
    start_time: str | None = Field(
        None,
        description="时间段起点（ISO 字符串，如 '2026-03-03 08:00:00'）。null 表示不裁剪。",
    )
    end_time: str | None = Field(
        None,
        description="时间段终点（ISO 字符串）。null 表示不裁剪。",
    )


@register
class LoadDatasetSkill(BaseSkill):
    name = "load_dataset"
    description = (
        "加载并清洗 CSV 数据集。完成读取、列名归一化（PV/MV/SV/timestamp）、"
        "时间戳解析、缺失值插值、PV 去噪。把清洗后的数据缓存到会话上下文，"
        "供后续辨识/整定技能使用。返回数据行数、采样周期、列清单、检测到的回路列表。"
    )
    input_model = LoadDatasetInputs

    def run(self, inputs: LoadDatasetInputs, ctx: LoopContext) -> SkillResult:
        cleaned, dt = _load_clean_only(
            csv_path=ctx.csv_path,
            selected_loop_prefix=inputs.loop_prefix or ctx.loop_prefix or None,
            start_time=inputs.start_time,
            end_time=inputs.end_time,
        )

        # 写入服务端上下文（不暴露给 LLM）
        ctx.cleaned_df = cleaned
        ctx.dt = dt
        if inputs.loop_prefix:
            ctx.loop_prefix = inputs.loop_prefix

        # 检测同一份 CSV 中存在的全部回路（仅用于摘要展示）
        from core.algorithms.data_analysis import _read_csv
        try:
            raw = _read_csv(ctx.csv_path)
            loops = _detect_loops(raw)
            loop_summary = [
                {"prefix": l["prefix"], "pv_col": l.get("pv_col", ""), "mv_col": l.get("mv_col", "")}
                for l in loops
            ]
        except Exception:
            loop_summary = []

        warnings: list[str] = []
        time_span_sec = float(dt) * len(cleaned)
        if len(cleaned) < 100:
            warnings.append(f"数据样本偏少（{len(cleaned)} 点），可能影响辨识质量")
        if "SV" not in cleaned.columns:
            warnings.append("CSV 中未发现 SV 列，将无法基于设定值阶跃来定位窗口")

        return SkillResult(
            success=True,
            data={
                "data_points": len(cleaned),
                "sampling_time": round(float(dt), 4),
                "time_span_sec": round(time_span_sec, 1),
                "columns": [c for c in ["timestamp", "SV", "PV", "MV"] if c in cleaned.columns],
                "loops_in_csv": loop_summary,
                "selected_loop_prefix": ctx.loop_prefix or "",
            },
            warnings=warnings,
            reasoning=f"已加载 {len(cleaned)} 行数据，采样周期 {dt:.3f}s",
        )
