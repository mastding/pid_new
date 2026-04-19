"""LLM 顾问：在窗口选择决策点引入 deepseek-reasoner。

Day 4 接入策略：单次同步调用，结构化 JSON 输出。理由：
  - 数据已在服务端加载并完成画像，无需走 tools loop 让 LLM 反复调用 load_dataset。
  - 单次调用延迟可控；reasoner 思考链已能给出像样的窗口选择理由。
  - 失败时调用方应回退到确定性 fit_score 最高策略，保证流水线不因 LLM 抖动中断。

返回值 None 表示失败（解析错误 / API 报错 / 索引非法），调用方需走回退路径。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是 PID 智能整定专家。当前任务：从若干候选辨识窗口里挑出最适合做系统辨识的一个。

判据要点（按重要性递减）：
1. 窗口必须能反映 MV 对 PV 的因果激励：mv_span 足够大（>5% 量程），corr 绝对值高。
2. window_quality_score 高的优先，但不要盲信高分；要结合数据画像里的饱和/死区/噪声信息。
3. PV 漂移（drift_ratio）大的窗口可能含外扰，慎选。
4. 若多个窗口质量接近，优先选 mv_step 来源（主动激励 > 被动 SV 阶跃）。
5. 若所有窗口都不理想，仍需选一个相对最好的，并在 reasoning 中明确风险。

输出严格要求：必须是合法 JSON，仅含两个字段：
  {"chosen_index": <0~N-1 的整数>, "reasoning": "<不超过 200 字的中文理由>"}
不要包含任何 Markdown 围栏、解释性前后文。"""


def _build_user_prompt(
    *,
    data_profile: dict[str, Any],
    candidate_windows: list[dict[str, Any]],
    loop_type: str,
) -> str:
    """组装单次调用的用户消息。"""
    lines: list[str] = []
    lines.append(f"回路类型：{loop_type}")
    lines.append("")
    lines.append("数据画像：")
    lines.append(data_profile.get("text_summary", "（无文字摘要）"))

    pv = data_profile.get("pv_stats", {})
    mv = data_profile.get("mv_stats", {})
    lines.append(
        f"  PV: min={pv.get('min')}, max={pv.get('max')}, range={pv.get('range')}"
    )
    lines.append(
        f"  MV: min={mv.get('min')}, max={mv.get('max')}, "
        f"触顶占比={mv.get('saturation_high_pct')}%, 触底占比={mv.get('saturation_low_pct')}%"
    )

    lines.append("")
    lines.append(f"候选窗口（共 {len(candidate_windows)} 个）：")
    for i, w in enumerate(candidate_windows):
        lines.append(
            f"  [{i}] source={w.get('window_source', '?')}, "
            f"score={w.get('window_quality_score', 0):.3f}, "
            f"usable={w.get('window_usable_for_id')}, "
            f"n_points={int(w.get('window_end_idx', 0)) - int(w.get('window_start_idx', 0))}, "
            f"mv_span={w.get('window_mv_span', 0):.2f}, "
            f"pv_span={w.get('window_pv_span', 0):.2f}, "
            f"corr={w.get('window_corr', 0):.2f}, "
            f"drift_ratio={w.get('window_drift_ratio', 0):.2f}"
        )

    lines.append("")
    lines.append("请选出最适合做系统辨识的窗口索引，并给出简短中文理由。")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any] | None:
    """容错地从 LLM 回答里抠出 JSON 对象。"""
    if not text:
        return None
    # 直接尝试
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 退一步：找第一个 { ... } 段
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def choose_window_via_llm(
    *,
    data_profile: dict[str, Any],
    candidate_windows: list[dict[str, Any]],
    loop_type: str,
    timeout: float = 60.0,
) -> dict[str, Any] | None:
    """让 LLM 在候选窗口里选一个，返回 {chosen_index, reasoning, raw_text}。

    失败（API 异常 / JSON 解析失败 / 索引越界）一律返回 None，由调用方回退。
    """
    if not candidate_windows:
        return None
    if not settings.model_api_key or not settings.model_api_url:
        logger.info("未配置 MODEL_API_KEY/URL，跳过 LLM 顾问")
        return None

    n = len(candidate_windows)
    user_prompt = _build_user_prompt(
        data_profile=data_profile,
        candidate_windows=candidate_windows,
        loop_type=loop_type,
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
    except Exception as exc:  # 网络 / API / 鉴权
        logger.warning("LLM 顾问调用失败：%s", exc)
        return None

    msg = resp.choices[0].message
    raw_text = (msg.content or "").strip()
    parsed = _extract_json(raw_text)
    if not parsed:
        logger.warning("LLM 顾问返回不是合法 JSON：%r", raw_text[:200])
        return None

    idx_raw = parsed.get("chosen_index")
    try:
        chosen_index = int(idx_raw)
    except (TypeError, ValueError):
        logger.warning("LLM 顾问返回的 chosen_index 不是整数：%r", idx_raw)
        return None
    if not (0 <= chosen_index < n):
        logger.warning("LLM 顾问返回的 chosen_index=%d 超出 [0, %d)", chosen_index, n)
        return None

    reasoning = str(parsed.get("reasoning", "")).strip() or "（LLM 未给出理由）"
    reasoning_content = getattr(msg, "reasoning_content", None) or ""

    return {
        "chosen_index": chosen_index,
        "reasoning": reasoning,
        "reasoning_content": reasoning_content,
        "raw_text": raw_text,
    }
