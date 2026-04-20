"""技能：综合数据画像。

一次性把 PV 量程、MV 饱和、死区、噪声、振荡、干扰分析聚合成
一个紧凑的 JSON 给 LLM。建议在 load_dataset 之后第一步就调此技能。
画像同时写入 LoopContext.data_profile，供后续决策点复用。

注：回路类型由用户通过 ctx.loop_type 指定，不在此自动推断。
"""
from __future__ import annotations

from pydantic import BaseModel

from core.skills.base import BaseSkill, LoopContext, NoInputs, SkillResult
from core.skills.data_understanding import _analyzers as A
from core.skills.registry import register


def _text_summary(profile: dict) -> str:
    """把结构化画像压成一句话，便于 LLM 快速扫读。"""
    parts: list[str] = []
    pv = profile["pv_stats"]
    mv = profile["mv_stats"]
    noise = profile["noise"]
    osc = profile["oscillation"]
    dz = profile["deadzone"]

    parts.append(f"PV 范围 {pv['min']}~{pv['max']}（跨度 {pv['range']}）")
    parts.append(f"MV 范围 {mv['min']}~{mv['max']}")

    if mv["saturation_high_pct"] > 5 or mv["saturation_low_pct"] > 5:
        parts.append(
            f"MV 饱和显著（触顶 {mv['saturation_high_pct']}% / 触底 {mv['saturation_low_pct']}%）"
        )
    parts.append(f"PV 噪声水平 {noise['noise_level']}（σ≈{noise['pv_noise_std']}）")

    # 死区：始终展示判定结果（含分母与滞后窗口），不只在"疑似"时输出
    events_total = dz.get("events_total", 0)
    lag_used = dz.get("lag_used_s", 0.0)
    mv_thr = dz.get("mv_step_threshold", 0.0)
    if events_total == 0:
        parts.append(
            f"死区：未检测到 MV 阶跃事件（滞后窗 {lag_used}s，MV 阶跃阈值 {mv_thr}），无法判定"
        )
    else:
        ratio = dz["evidence_ratio"]
        evidence = dz.get("evidence_count", 0)
        suspect = "疑似存在" if ratio > 0.3 else "未见明显"
        parts.append(
            f"死区{suspect}（{evidence}/{events_total} 次 MV 阶跃 PV 无响应，占比 {ratio:.0%}，滞后窗 {lag_used}s）"
        )
    if osc["detected"]:
        parts.append(f"检测到振荡 T≈{osc['period_sec']}s")

    return "；".join(parts)


@register
class SummarizeDataSkill(BaseSkill):
    name = "summarize_data"
    description = (
        "生成数据画像，覆盖 6 个维度："
        "(1) PV 量程与物理单位兼容性；"
        "(2) MV 饱和占比；"
        "(3) 控制死区估计；"
        "(4) PV 噪声水平；"
        "(5) 振荡检测（FFT 主频）；"
        "(6) SV/MV 阶跃与 PV 漂移计数。"
        "回路类型由用户提供（ctx.loop_type），不在此推断。"
        "调用前必须先调用 load_dataset。结果同时缓存到会话上下文。"
    )
    input_model = NoInputs

    def run(self, inputs: NoInputs, ctx: LoopContext) -> SkillResult:
        if ctx.cleaned_df is None or ctx.dt is None:
            return SkillResult(
                success=False,
                reasoning="未检测到已加载的数据集，请先调用 load_dataset 技能。",
            )

        df = ctx.cleaned_df
        dt = float(ctx.dt)

        # 先算噪声，其它分析器依赖 pv_noise_std
        noise = A.analyze_noise(df)

        profile = {
            "pv_stats": A.analyze_pv_range(df),
            "mv_stats": A.analyze_mv_saturation(df),
            "noise": noise,
            "deadzone": A.analyze_deadzone(
                df,
                pv_noise_std=noise["pv_noise_std"],
                dt=dt,
                loop_type=getattr(ctx, "loop_type", None),
            ),
            "oscillation": A.analyze_oscillation(df, dt=dt),
            "disturbance": A.analyze_disturbance(df),
        }
        profile["text_summary"] = _text_summary(profile)

        # 写入上下文
        ctx.data_profile = profile

        # 汇总警告（不在 profile 里，避免 LLM 重复读到）
        warnings: list[str] = []
        if profile["mv_stats"]["saturation_high_pct"] > 30:
            warnings.append("MV 长时间触顶，辨识结果可能失真")
        if profile["mv_stats"]["saturation_low_pct"] > 30:
            warnings.append("MV 长时间触底，辨识结果可能失真")
        if profile["deadzone"]["evidence_ratio"] > 0.5:
            warnings.append("死区证据占比过高，小幅整定可能无效")
        if profile["noise"]["noise_level"] == "high":
            warnings.append("PV 噪声水平偏高，建议辨识前再评估是否需加强去噪")

        return SkillResult(
            success=True,
            data=profile,
            warnings=warnings,
            reasoning=profile["text_summary"],
        )
