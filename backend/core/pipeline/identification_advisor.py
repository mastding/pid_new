"""LLM 顾问：辨识结束后的模型评审。

在 fit_best_model 选定 best 模型之后调用，输入 best + 全部 attempts + 数据画像，
让 deepseek-reasoner 出三选一裁决：
  - accept：模型可信，正常走整定
  - downgrade：模型有可疑，但仍走整定，强制评估限分（tuning_unreliable=true）
  - reject：模型不可信，直接 abort 流水线

LLM 失败/超时/解析失败一律返回 None，调用方默认 accept，不阻塞用户。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是 PID 智能整定专家。当前任务：评审一次系统辨识的结果。

辨识算法已经在多窗口×多模型类型的笛卡尔积里按 AIC 选了"最优"模型。
候选模型池（不同回路类型顺序不同）：
  - FO       一阶（无死时）
  - FOPDT    一阶+死时           K·exp(-Ls)/(Ts+1)
  - SOPDT    二阶过阻尼+死时      K·exp(-Ls)/((T1s+1)(T2s+1))
  - IPDT     纯积分+死时          K·exp(-Ls)/s
  - SOPDT_UNDER  二阶欠阻尼+死时  K·exp(-Ls)/(T²s²+2ζTs+1), 0<ζ<1
                  → 用于振荡型对象（管线共振/阀门振荡），有 zeta 字段
  - IFOPDT   积分+一阶+死时       K·exp(-Ls)/(s·(Ts+1))
                  → 用于液位/储能型回路，比纯 IPDT 更细

你的职责是判断这个"最优"是否真的可信，避免算法在数据本身不行的情况下还硬给参数。

判据要点：
1. K 符号是否合理：流量/压力/液位回路 K 通常为正（MV 增→PV 增）。负 K 在量程内一般不合理，除非控制器是反作用。
2. T 是否符合回路类型量级：
   - flow: 1~30s 为常见，<1s 通常是过拟合噪声
   - pressure: 5~120s
   - temperature: 30s~30min
   - level: 60s~30min（IPDT/IFOPDT 是积分对象，主看 K 大小）
3. R² 与 NRMSE：R²>0.7 较可靠；R²<0.3 几乎是噪声拟合；NRMSE>0.5 拟合很糟
4. confidence 与各候选模型差距：如果 best 与第二名 fit_score 差距很小，说明算法犹豫，应当谨慎
5. 数据画像里的死区/饱和/噪声水平：死区>30% 通常意味着 MV 没真激励，K 估计不可信
6. 选中窗口的 corr：|corr|<0.3 说明 MV-PV 因果性弱，模型存疑
7. SOPDT_UNDER 专属：zeta 越接近 0 越激进；ζ<0.2 通常意味着对象本身在振，整定一定要走保守 LAMBDA。如果数据画像里"oscillation.detected=false"但 best 选了 SOPDT_UNDER，要存疑（可能是噪声拟合）
8. IFOPDT 专属：积分对象 + 一阶滞后，T 与 L 通常都不大；若 T 大于 L 数倍，说明真的是带显著一阶 dynamics 的积分对象（液位类），是合理选择

verdict 选择标准：
- accept：模型 R²>0.6、K 符号合理、T 在回路类型量级范围内、无明显数据问题
- downgrade：有 1-2 处可疑点（如 R² 偏低 0.3-0.6 / T 偏小但未到退化阈值 / 数据有明显噪声死区），仍可走流程但需提示用户风险
- reject：有严重问题（K 符号错 / R²<0.3 / T 接近 dt / 多个 attempts 都失败 / corr 极低），不应继续整定

输出严格要求：必须是合法 JSON，仅含三个字段：
{
  "verdict": "accept" | "downgrade" | "reject",
  "reason": "<不超过 150 字的中文裁决理由>",
  "concerns": ["<具体担忧 1>", "<具体担忧 2>", ...]   // 0~4 条，accept 时可为空数组
}
不要包含任何 Markdown 围栏、解释性前后文。"""


def _build_user_prompt(
    *,
    loop_type: str,
    data_profile: dict[str, Any],
    chosen_window_summary: dict[str, Any],
    best_model: dict[str, Any],
    attempts: list[dict[str, Any]],
    confidence: float,
) -> str:
    lines: list[str] = []
    lines.append(f"回路类型：{loop_type}")
    lines.append("")
    lines.append("【数据画像】")
    lines.append(data_profile.get("text_summary", "（无文字摘要）"))
    pv = data_profile.get("pv_stats", {})
    mv = data_profile.get("mv_stats", {})
    lines.append(
        f"PV: min={pv.get('min')}, max={pv.get('max')}, range={pv.get('range')}"
    )
    lines.append(
        f"MV: min={mv.get('min')}, max={mv.get('max')}, "
        f"触顶={mv.get('saturation_high_pct')}%, 触底={mv.get('saturation_low_pct')}%"
    )

    lines.append("")
    lines.append("【选中窗口】")
    lines.append(
        f"source={chosen_window_summary.get('source', '?')}, "
        f"score={chosen_window_summary.get('score', 0):.3f}, "
        f"n_points={chosen_window_summary.get('n_points', 0)}"
    )

    lines.append("")
    lines.append("【最终选定模型】")
    lines.append(
        f"type={best_model.get('model_type')}, "
        f"K={best_model.get('K', 0):.4f}, "
        f"T={best_model.get('T', 0):.2f}s, "
        f"T1={best_model.get('T1', 0):.2f}s, "
        f"T2={best_model.get('T2', 0):.2f}s, "
        f"L={best_model.get('L', 0):.2f}s, "
        f"zeta={best_model.get('zeta', 0):.3f}, "
        f"R²={best_model.get('r2_score', 0):.3f}, "
        f"NRMSE={best_model.get('normalized_rmse', 0):.3f}, "
        f"confidence={confidence:.2f}"
    )

    lines.append("")
    lines.append("【全部辨识尝试】（按 fit_score 降序，最多 8 条）")
    sorted_attempts = sorted(
        [a for a in attempts if a.get("success")],
        key=lambda a: float(a.get("fit_score", -1e9)),
        reverse=True,
    )[:8]
    for i, a in enumerate(sorted_attempts):
        marker = " ★" if (
            a.get("model_type") == best_model.get("model_type")
            and a.get("window_source") == best_model.get("window_source")
        ) else ""
        deg = "[退化T]" if a.get("degenerate_T") else ""
        lines.append(
            f"  [{i}]{marker} {deg} window={a.get('window_source', '?')}, "
            f"model={a.get('model_type')}, "
            f"K={a.get('K', 0):.3f}, T={a.get('T', 0):.2f}, L={a.get('L', 0):.2f}, "
            f"R²={a.get('r2_score', 0):.3f}, fit_score={a.get('fit_score', 0):.2f}, "
            f"conf={a.get('confidence', 0):.2f}"
        )

    failed = [a for a in attempts if not a.get("success")]
    if failed:
        lines.append(f"另有 {len(failed)} 次尝试失败（多为优化不收敛）")

    lines.append("")
    lines.append("请评审这次辨识结果，输出 verdict + reason + concerns。")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def review_identification_via_llm(
    *,
    loop_type: str,
    data_profile: dict[str, Any],
    chosen_window_summary: dict[str, Any],
    best_model: dict[str, Any],
    attempts: list[dict[str, Any]],
    confidence: float,
    timeout: float = 60.0,
) -> dict[str, Any] | None:
    """让 LLM 评审辨识结果，返回 {verdict, reason, concerns, reasoning_content, raw_text}。

    失败一律返回 None，调用方应默认 accept。
    """
    if not settings.model_api_key or not settings.model_api_url:
        logger.info("未配置 MODEL_API_KEY/URL，跳过模型评审")
        return None

    user_prompt = _build_user_prompt(
        loop_type=loop_type,
        data_profile=data_profile,
        chosen_window_summary=chosen_window_summary,
        best_model=best_model,
        attempts=attempts,
        confidence=confidence,
    )

    try:
        client = OpenAI(
            api_key=settings.model_api_key,
            base_url=settings.model_api_url,
            timeout=timeout,
        )
        resp = client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("模型评审 LLM 调用失败：%s", exc)
        return None

    msg = resp.choices[0].message
    raw_text = (msg.content or "").strip()
    parsed = _extract_json(raw_text)
    if not parsed:
        logger.warning("模型评审返回不是合法 JSON：%r", raw_text[:200])
        return None

    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in ("accept", "downgrade", "reject"):
        logger.warning("模型评审返回非法 verdict：%r", verdict)
        return None

    reason = str(parsed.get("reason", "")).strip() or "（LLM 未给出理由）"
    concerns_raw = parsed.get("concerns", [])
    concerns: list[str] = []
    if isinstance(concerns_raw, list):
        concerns = [str(c).strip() for c in concerns_raw if str(c).strip()][:4]

    reasoning_content = getattr(msg, "reasoning_content", None) or ""

    return {
        "verdict": verdict,
        "reason": reason,
        "concerns": concerns,
        "reasoning_content": reasoning_content,
        "raw_text": raw_text,
    }
