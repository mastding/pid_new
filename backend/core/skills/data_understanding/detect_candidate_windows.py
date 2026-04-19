"""技能：检测候选辨识窗口。

LLM 在辨识之前调用，了解可用的窗口及其质量分。结果同时写入
LoopContext.candidate_windows，供后续辨识技能使用。
"""
from __future__ import annotations

from pydantic import BaseModel

from core.algorithms.data_analysis import build_candidate_windows
from core.skills.base import BaseSkill, LoopContext, NoInputs, SkillResult
from core.skills.registry import register


def _summarize_window(w: dict, idx: int) -> dict:
    """把窗口字典压成给 LLM 看的紧凑摘要（去掉 numpy 数组等大对象）。"""
    return {
        "index": idx,
        "start": int(w.get("window_start_idx", 0)),
        "end": int(w.get("window_end_idx", 0)),
        "n_points": int(w.get("window_end_idx", 0)) - int(w.get("window_start_idx", 0)),
        "type": w.get("type", ""),
        "amplitude": round(float(w.get("amplitude", 0.0)), 4),
        "score": round(float(w.get("window_quality_score", 0.0)), 4),
        "usable": bool(w.get("window_usable_for_id", False)),
        "source": w.get("window_source", ""),
        "mv_span": round(float(w.get("window_mv_span", 0.0)), 4),
        "pv_span": round(float(w.get("window_pv_span", 0.0)), 4),
        "corr": round(float(w.get("window_corr", 0.0)), 4),
    }


@register
class DetectCandidateWindowsSkill(BaseSkill):
    name = "detect_candidate_windows"
    description = (
        "在已加载的清洗数据上检测候选辨识窗口。窗口由 SV 阶跃 / MV 阶跃 / "
        "MV 大幅变化等触发条件确定，并按质量分排序。"
        "返回每个窗口的起止索引、长度、来源类型、质量分、是否可用于辨识。"
        "调用前必须先调用 load_dataset。"
    )
    input_model = NoInputs

    def run(self, inputs: NoInputs, ctx: LoopContext) -> SkillResult:
        if ctx.cleaned_df is None or ctx.dt is None:
            return SkillResult(
                success=False,
                reasoning="未检测到已加载的数据集，请先调用 load_dataset 技能。",
            )

        windows, step_events = build_candidate_windows(ctx.cleaned_df, ctx.dt)

        # 写入上下文（保留完整 dict，便于辨识技能使用）
        ctx.candidate_windows = windows

        summaries = [_summarize_window(w, i) for i, w in enumerate(windows)]
        usable_count = sum(1 for s in summaries if s["usable"])

        warnings: list[str] = []
        if not windows:
            warnings.append("未检测到任何候选窗口（数据可能没有阶跃或显著变化）")
        elif usable_count == 0:
            warnings.append(
                f"检测到 {len(windows)} 个候选窗口但均未通过质量门槛，"
                "辨识阶段将启用兜底逻辑（在所有候选窗口上尝试拟合）。"
            )

        return SkillResult(
            success=True,
            data={
                "candidate_count": len(windows),
                "usable_count": usable_count,
                "step_event_count": len(step_events),
                "windows": summaries,
            },
            warnings=warnings,
            reasoning=(
                f"检测到 {len(windows)} 个候选窗口，其中 {usable_count} 个通过质量门槛"
                if windows
                else "未发现候选窗口"
            ),
        )
