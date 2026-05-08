"""LLM advisor that turns ontology/profile context into window-algorithm policy."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from core.model_config import store as model_cfg_store
from core.pipeline.ontology_policy_builder import _build_algorithm_plan
from core.pipeline.window_policy_models import WindowSelectionPolicy
from core.pipeline.window_policy_usage import enrich_policy_field_usage

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是 PID 整定窗口候选智能体中的“策略生成器”。

你的任务不是直接选择最终窗口，而是根据回路画像和本体/MCP 查询结果，生成给确定性窗口算法族使用的 JSON 策略。

必须遵守：
1. 只输出合法 JSON，不要 Markdown。
2. 策略要服务于算法输入：算法族优先级、窗口前后长度、稳态扫描窗口、扫描步长、候选数量、激励/饱和/漂移/噪声门槛。
3. 不确定时宁可保守，不要禁用所有算法族；rolling_scan 只能作为兜底诊断，通常降级。
4. 如果本体说明时间常数、时滞、增益方向、最小阶跃或噪声容忍度，应显式转成字段。

输出字段：
{
  "preferred_algorithm_families": ["mv_step"|"mv_ramp"|"sp_step"|"steady_disturbance"|"rolling_scan"],
  "deprioritized_algorithm_families": [...],
  "disabled_algorithm_families": [...],
  "min_mv_excitation": <number|null>,
  "min_sp_excitation": <number|null>,
  "min_pv_response": <number|null>,
  "max_mv_saturation_ratio": <0~1|null>,
  "max_pv_noise_ratio": <0~1|null>,
  "max_drift_ratio": <number|null>,
  "expected_dead_time_range_s": [low, high] 或 null,
  "expected_time_constant_range_s": [low, high] 或 null,
  "expected_gain_sign": "positive"|"negative"|"unknown",
  "min_window_points": <int>,
  "min_window_duration_s": <seconds>,
  "max_window_points": <int|null>,
  "pre_window_s": <seconds|null>,
  "post_window_s": <seconds|null>,
  "steady_scan_window_s": <seconds|null>,
  "steady_scan_step_s": <seconds|null>,
  "merge_gap_s": <seconds|null>,
  "max_candidates_per_family": <int>,
  "allowed_operating_states": [string],
  "avoid_operating_states": [string],
  "rationale": "<不超过300字中文说明>",
  "ontology_evidence": [{"fact": "<本体事实>", "source": "<来源>"}]
}
"""


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _range_or_none(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        low = float(value[0])
        high = float(value[1])
    except (TypeError, ValueError):
        return None
    if low < 0 or high < low:
        return None
    return (low, high)


def _list_of_strings(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    result = [str(item).strip() for item in value if str(item).strip()]
    return result


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    value_f = _float_or_none(value)
    if value_f is None:
        return None
    return int(round(value_f))


def _merge_policy(base: dict[str, Any], proposed: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    list_fields = {
        "preferred_algorithm_families",
        "deprioritized_algorithm_families",
        "disabled_algorithm_families",
        "allowed_operating_states",
        "avoid_operating_states",
    }
    float_fields = {
        "min_mv_excitation",
        "min_sp_excitation",
        "min_pv_response",
        "max_mv_saturation_ratio",
        "max_pv_noise_ratio",
        "max_drift_ratio",
        "min_window_duration_s",
        "pre_window_s",
        "post_window_s",
        "steady_scan_window_s",
        "steady_scan_step_s",
        "merge_gap_s",
    }
    int_fields = {"min_window_points", "max_window_points", "max_candidates_per_family"}
    range_fields = {"expected_dead_time_range_s", "expected_time_constant_range_s"}

    for field in list_fields:
        value = _list_of_strings(proposed.get(field))
        if value is not None:
            merged[field] = value
    for field in float_fields:
        if field in proposed:
            merged[field] = _float_or_none(proposed.get(field))
    for field in int_fields:
        if field in proposed:
            merged[field] = _int_or_none(proposed.get(field))
    for field in range_fields:
        value = _range_or_none(proposed.get(field))
        if value is not None:
            merged[field] = value

    gain_sign = str(proposed.get("expected_gain_sign") or "").strip()
    if gain_sign in {"positive", "negative", "unknown"}:
        merged["expected_gain_sign"] = gain_sign
    rationale = str(proposed.get("rationale") or "").strip()
    if rationale:
        merged["rationale"] = rationale[:600]

    preferred = merged.get("preferred_algorithm_families") or []
    deprioritized = merged.get("deprioritized_algorithm_families") or []
    disabled = merged.get("disabled_algorithm_families") or []
    merged["algorithm_plan"] = _build_algorithm_plan(
        preferred=preferred,
        deprioritized=deprioritized,
        disabled=disabled,
        loop_type=str(merged.get("loop_type") or ""),
    )
    merged["policy_version"] = "phase2-llm"
    merged["confidence"] = max(float(merged.get("confidence") or 0.0), 0.78)

    facts = dict(merged.get("ontology_facts") or {})
    evidence = proposed.get("ontology_evidence")
    if isinstance(evidence, list):
        facts["evidence"] = [
            {
                "fact": str(item.get("fact", ""))[:240],
                "source": str(item.get("source", ""))[:160],
            }
            for item in evidence
            if isinstance(item, dict) and (item.get("fact") or item.get("source"))
        ][:10]
    merged["ontology_facts"] = facts
    return enrich_policy_field_usage(WindowSelectionPolicy(**merged).model_dump())


def _build_user_prompt(
    *,
    base_policy: dict[str, Any],
    data_profile: dict[str, Any],
    mcp_context: dict[str, Any] | None,
    frontend_context: str | None,
) -> str:
    profile_text = str(data_profile.get("text_summary") or "无")
    pv = data_profile.get("pv_stats", {}) if isinstance(data_profile.get("pv_stats"), dict) else {}
    mv = data_profile.get("mv_stats", {}) if isinstance(data_profile.get("mv_stats"), dict) else {}
    raw_profile = dict(data_profile)
    raw_profile.pop("process_prior", None)
    raw_profile_json = json.dumps(raw_profile, ensure_ascii=False, default=str)
    if len(raw_profile_json) > 12000:
        raw_profile_json = raw_profile_json[:12000] + "\n...(LoopFeatures raw profile truncated)"
    mcp_content = str((mcp_context or {}).get("content") or "")
    if len(mcp_content) > 12000:
        mcp_content = mcp_content[:12000] + "\n...（MCP 内容已截断）"
    frontend_text = (frontend_context or "").strip()
    if len(frontend_text) > 4000:
        frontend_text = frontend_text[:4000] + "\n...（前端上下文已截断）"

    return "\n".join([
        "基础默认策略 JSON：",
        json.dumps(base_policy, ensure_ascii=False),
        "",
        "历史数据画像摘要：",
        profile_text,
        f"PV统计: {json.dumps(pv, ensure_ascii=False)}",
        f"MV统计: {json.dumps(mv, ensure_ascii=False)}",
        "LoopFeatures raw JSON (process_prior removed):",
        raw_profile_json,
        "",
        "本体/MCP查询结果：",
        mcp_content or "无 MCP 内容",
        "",
        "前端图谱兜底上下文：",
        frontend_text or "无",
        "",
        "请输出修正后的窗口算法策略 JSON。",
    ])


def ask_window_policy_via_llm(
    *,
    base_policy: dict[str, Any],
    data_profile: dict[str, Any],
    mcp_context: dict[str, Any] | None,
    frontend_context: str | None = None,
    timeout: float = 60.0,
) -> dict[str, Any] | None:
    """Return a validated policy dict or None if the policy LLM is unavailable."""
    model_cfg = model_cfg_store.get()
    if not model_cfg.model_api_key or not model_cfg.model_api_url:
        logger.info("未配置模型 API，跳过窗口策略 LLM 顾问")
        return None

    prompt = _build_user_prompt(
        base_policy=base_policy,
        data_profile=data_profile,
        mcp_context=mcp_context,
        frontend_context=frontend_context,
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
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning(
            "窗口策略 LLM 调用失败：%s: %s | url=%s model=%s cause=%r",
            type(exc).__name__,
            exc,
            model_cfg.model_api_url,
            model_cfg.model_name,
            getattr(exc, "__cause__", None),
        )
        return None

    msg = resp.choices[0].message
    raw_text = (msg.content or "").strip()
    parsed = _extract_json(raw_text)
    if not parsed:
        logger.warning("窗口策略 LLM 返回不是合法 JSON：%r", raw_text[:300])
        return None

    try:
        policy = _merge_policy(base_policy, parsed)
    except Exception as exc:
        logger.warning("窗口策略 LLM JSON 校验失败：%s", exc)
        return None

    policy["llm_policy_raw_text"] = raw_text[:8000]
    policy["llm_policy_reasoning_content"] = getattr(msg, "reasoning_content", None) or ""
    return policy
