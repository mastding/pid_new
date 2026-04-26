"""Phase 2: 辨识精修顾问。

在 model_review verdict=downgrade 之后调用，让 LLM 决定下一轮辨识应该
**改什么参数**重试，比如：
- 换一个候选窗口（force_window_index）
- 缩小模型类型范围（force_model_types）
- 提供死时初值提示（hint_L）

LLM 失败/超时/解析失败 → 返回 retry=False，runner 走 Phase 3 兜底。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from config import settings

from core.model_config import store as model_cfg_store

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是 PID 智能整定专家。当前任务：拿到上一轮辨识被评审降级（downgrade）的诊断后，
判断"再辨识一次"是否值得，以及具体改什么参数。

可用的候选模型（按回路类型预设的优先级）：
  - FO         一阶（无死时）
  - FOPDT      一阶+死时
  - SOPDT      二阶过阻尼+死时
  - IPDT       纯积分+死时
  - SOPDT_UNDER 二阶欠阻尼+死时（振荡型对象，0<ζ<1）
  - IFOPDT     积分+一阶+死时（液位/储能型）

可用的候选窗口由 windows_summary 给出，每个有 index/source/n_points/score/corr。

判断思路：
1. 上轮 best 的问题在哪？（concerns 已给）
2. 改什么能消除这个问题：
   - "T 偏小到塌缩"  → 试更长窗口、或换更高阶模型 (FOPDT→SOPDT, FO→FOPDT)
   - "K 符号反"      → 检查窗口里 MV-PV 是否真的同向，挑 corr 同号的窗口
   - "R² 极低"       → 窗口可能没真激励，挑 score 高且 corr 大的窗口
   - "振荡未识别"    → force_model_types 加入 SOPDT_UNDER
   - "积分对象拟合差" → force_model_types=[IFOPDT, IPDT]
   - "L 不合理"     → 给 hint_L
3. 如果上一轮已经把所有合理窗口/模型都试过了 → retry=false, 不要重试浪费 token

输出严格要求：必须是合法 JSON，仅含字段：
{
  "retry": true | false,
  "rationale": "<不超过 150 字的中文重试理由或放弃理由>",
  "force_window_index": <int | null>,           // null = 不限制窗口
  "force_model_types": ["FOPDT", ...],           // 空数组 = 用回路类型默认
  "hint_L": <number | null>                      // null = 不提示
}
不要包含任何 Markdown 围栏、解释性前后文。"""


def _build_user_prompt(
    *,
    loop_type: str,
    round_idx: int,
    max_rounds: int,
    data_profile: dict[str, Any],
    windows_summary: list[dict[str, Any]],
    algorithm_comparison: list[dict[str, Any]],
    last_best: dict[str, Any],
    last_attempts: list[dict[str, Any]],
    last_review: dict[str, Any],
    history_summary: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(f"回路类型：{loop_type}")
    lines.append(f"当前是第 {round_idx} / {max_rounds} 轮重试决策")
    lines.append("")
    lines.append("【数据画像摘要】")
    lines.append(data_profile.get("text_summary", "（无）"))

    lines.append("")
    lines.append("【可用候选窗口】")
    for w in windows_summary[:8]:
        lines.append(
            f"  index={w.get('index')}, source={w.get('source')}, "
            f"n={w.get('n_points')}, score={w.get('score', 0):.2f}, "
            f"corr={w.get('corr', 0):.2f}"
        )

    if algorithm_comparison:
        lines.append("")
        lines.append("[Window algorithm family comparison: best fit_score per family]")
        for item in algorithm_comparison[:6]:
            lines.append(
                f"  algorithm={item.get('algorithm_label') or item.get('algorithm')}, "
                f"best_window={item.get('window_source')}, model={item.get('model_type')}, "
                f"window_score={float(item.get('window_quality_score', 0) or 0):.2f}, "
                f"R2={float(item.get('r2_score', 0) or 0):.3f}, "
                f"fit_score={float(item.get('fit_score', 0) or 0):.2f}, "
                f"confidence={float(item.get('confidence', 0) or 0):.2f}"
            )

    lines.append("")
    lines.append("【上一轮被降级的最优模型】")
    lines.append(
        f"type={last_best.get('model_type')}, "
        f"K={last_best.get('K', 0):.4f}, T={last_best.get('T', 0):.2f}s, "
        f"L={last_best.get('L', 0):.2f}s, R²={last_best.get('r2_score', 0):.3f}, "
        f"window={last_best.get('window_source')}"
    )

    lines.append("")
    lines.append("【上一轮 LLM 评审结论】")
    lines.append(f"verdict={last_review.get('verdict')}")
    lines.append(f"reason={last_review.get('reason', '')}")
    concerns = last_review.get("concerns", []) or []
    if concerns:
        lines.append("concerns:")
        for c in concerns:
            lines.append(f"  - {c}")

    lines.append("")
    lines.append("【上一轮全部 attempts（按 fit_score 降序，最多 6 条）】")
    succ = sorted(
        [a for a in last_attempts if a.get("success")],
        key=lambda a: float(a.get("fit_score", -1e9)),
        reverse=True,
    )[:6]
    for a in succ:
        lines.append(
            f"  window={a.get('window_source')}, model={a.get('model_type')}, "
            f"K={a.get('K', 0):.3f}, T={a.get('T', 0):.2f}, L={a.get('L', 0):.2f}, "
            f"R²={a.get('r2_score', 0):.3f}, fit_score={a.get('fit_score', 0):.2f}"
        )
    n_fail = sum(1 for a in last_attempts if not a.get("success"))
    if n_fail:
        lines.append(f"另有 {n_fail} 次拟合失败")

    if history_summary:
        lines.append("")
        lines.append("【已经试过的辨识方案（避免重复）】")
        for h in history_summary:
            lines.append(
                f"  round {h.get('round')}: window_idx={h.get('window_index', 'auto')}, "
                f"models={h.get('model_types', 'default')}, "
                f"L_hint={h.get('hint_L', '-')} → best={h.get('best_type')} "
                f"R²={h.get('best_r2', 0):.3f} → {h.get('verdict')}"
            )

    lines.append("")
    lines.append("请决定是否重试以及改什么参数，输出 JSON。")
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


_VALID_MODEL_TYPES = {"FO", "FOPDT", "SOPDT", "IPDT", "SOPDT_UNDER", "IFOPDT"}


def ask_refinement_via_llm(
    *,
    loop_type: str,
    round_idx: int,
    max_rounds: int,
    data_profile: dict[str, Any],
    windows_summary: list[dict[str, Any]],
    last_best: dict[str, Any],
    last_attempts: list[dict[str, Any]],
    last_review: dict[str, Any],
    history_summary: list[dict[str, Any]],
    algorithm_comparison: list[dict[str, Any]] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any] | None:
    """让 LLM 决定下一轮辨识方案。

    返回 None 时调用方应当视为 retry=false。
    成功返回 {retry, rationale, force_window_index, force_model_types,
    hint_L, reasoning_content, raw_text}。
    """
    model_cfg = model_cfg_store.get()
    if not model_cfg.model_api_key or not model_cfg.model_api_url:
        return None

    user_prompt = _build_user_prompt(
        loop_type=loop_type,
        round_idx=round_idx,
        max_rounds=max_rounds,
        data_profile=data_profile,
        windows_summary=windows_summary,
        algorithm_comparison=algorithm_comparison or [],
        last_best=last_best,
        last_attempts=last_attempts,
        last_review=last_review,
        history_summary=history_summary,
    )

    try:
        client = OpenAI(
            api_key=model_cfg.model_api_key,
            base_url=model_cfg.model_api_url,
            timeout=timeout,
        )
        resp = client.chat.completions.create(
            model=model_cfg.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("精修顾问 LLM 调用失败：%s", exc)
        return None

    msg = resp.choices[0].message
    raw_text = (msg.content or "").strip()
    parsed = _extract_json(raw_text)
    if not parsed:
        logger.warning("精修顾问返回非合法 JSON：%r", raw_text[:200])
        return None

    retry = bool(parsed.get("retry", False))
    rationale = str(parsed.get("rationale", "")).strip() or "（LLM 未给出理由）"

    # window index：必须是 int 或 None
    fw = parsed.get("force_window_index")
    try:
        force_window_index = int(fw) if fw is not None else None
    except (TypeError, ValueError):
        force_window_index = None

    # model types：白名单过滤
    fm_raw = parsed.get("force_model_types") or []
    if not isinstance(fm_raw, list):
        fm_raw = []
    force_model_types = [
        str(m).upper().strip() for m in fm_raw
        if str(m).upper().strip() in _VALID_MODEL_TYPES
    ]

    # L hint
    hl = parsed.get("hint_L")
    try:
        hint_L = float(hl) if hl is not None else None
    except (TypeError, ValueError):
        hint_L = None

    reasoning_content = getattr(msg, "reasoning_content", None) or ""

    return {
        "retry": retry,
        "rationale": rationale,
        "force_window_index": force_window_index,
        "force_model_types": force_model_types,
        "hint_L": hint_L,
        "reasoning_content": reasoning_content,
        "raw_text": raw_text,
    }
