"""Ontology-backed rule evaluation for historical PID loops.

The ontology service is expected to become the source of operating-case,
variable, limit and loop-relation facts. This module deliberately keeps the
first integration deterministic: ontology text is parsed for facts when
possible, but rule outcomes always expose missing facts instead of inventing
them.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from core.history.store import (
    assess_loop,
    compute_loop_cpk,
    compute_loop_harris,
    get_loop,
    get_loop_profile_bundle,
)
from core.pipeline.ontology_mcp_context import fetch_loop_ontology_context_via_mcp


RULE_PACK_VERSION = "5203-loop-diagnosis-v1"
ONTOLOGY_FACT_TIMEOUT_S = 3.0
ONTOLOGY_FACT_CACHE_TTL_S = 300.0
ONTOLOGY_FACT_ERROR_CACHE_TTL_S = 30.0
ONTOLOGY_FACT_SCHEMA_VERSION = "ontology_facts.v1"

_FACT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _cache_key(loop_id: str, loop_type: str, max_chars: int) -> str:
    return f"{loop_id}|{loop_type}|{max_chars}"


def _clone_payload(value: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return dict(value)


def clear_ontology_fact_cache() -> None:
    _FACT_CACHE.clear()


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _dig(data: Any, *keys: str) -> Any:
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _json_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    # MCP tools often wrap JSON in markdown fences or explanatory text.
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _first_number_near(text: str, labels: tuple[str, ...]) -> float | None:
    for label in labels:
        pattern = rf"{re.escape(label)}\s*[:：=]\s*(-?\d+(?:\.\d+)?)"
        match = re.search(pattern, text, re.I)
        if match:
            return _to_float(match.group(1))
    return None


def _first_value(data: dict[str, Any], names: tuple[str, ...]) -> Any:
    lowered = {str(k).lower(): v for k, v in data.items()}
    for name in names:
        if name in data:
            return data[name]
        value = lowered.get(name.lower())
        if value is not None:
            return value
    return None


def _walk_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        found.append(value)
        for child in value.values():
            found.extend(_walk_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_dicts(child))
    return found


def _parse_spec_limits(ontology_context: dict[str, Any] | None) -> dict[str, Any]:
    if not ontology_context:
        return {"lsl": None, "usl": None, "source": "missing", "evidence": []}

    text = str(ontology_context.get("content") or "")
    parsed = _json_from_text(text)
    lsl_keys = ("pv_lsl", "lsl", "lower_spec_limit", "lower_limit", "pv_lower_limit", "pv_low")
    usl_keys = ("pv_usl", "usl", "upper_spec_limit", "upper_limit", "pv_upper_limit", "pv_high")

    evidence: list[str] = []
    if parsed:
        for item in _walk_dicts(parsed):
            lsl = _to_float(_first_value(item, lsl_keys))
            usl = _to_float(_first_value(item, usl_keys))
            if lsl is not None or usl is not None:
                evidence.append("structured_json")
            if lsl is not None and usl is not None:
                return {
                    "lsl": lsl,
                    "usl": usl,
                    "source": "ontology_structured_json",
                    "evidence": evidence,
                }

    lsl = _first_number_near(text, ("PV_LSL", "LSL", "PV下限", "PV 下限", "规格下限", "下限"))
    usl = _first_number_near(text, ("PV_USL", "USL", "PV上限", "PV 上限", "规格上限", "上限"))
    if lsl is not None or usl is not None:
        evidence.append("ontology_text_regex")
    return {
        "lsl": lsl,
        "usl": usl,
        "source": "ontology_text" if lsl is not None and usl is not None else "missing",
        "evidence": evidence,
    }


def _parse_ontology_facts(loop_id: str, loop_type: str, ontology_context: dict[str, Any] | None) -> dict[str, Any]:
    text = str((ontology_context or {}).get("content") or "")
    parsed = _json_from_text(text) or {}
    case_id = (
        _first_value(parsed, ("case_id", "operating_case", "operating_condition"))
        if isinstance(parsed, dict)
        else None
    )
    relation_hints = []
    for keyword, relation_type in (("串级", "cascade"), ("前馈", "feedforward"), ("MISO", "miso"), ("多输入", "miso")):
        if keyword.lower() in text.lower():
            relation_hints.append(relation_type)
    return {
        "schema_version": ONTOLOGY_FACT_SCHEMA_VERSION,
        "loop_id": loop_id,
        "loop_type": loop_type,
        "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_id": str(case_id).strip() if case_id else None,
        "pv_spec_limits": _parse_spec_limits(ontology_context),
        "relation_hints": sorted(set(relation_hints)),
        "raw_context": ontology_context or {},
    }


async def resolve_loop_ontology_facts(
    *,
    loop_id: str,
    loop_type: str,
    max_chars: int = 12000,
    force_refresh: bool = False,
) -> dict[str, Any]:
    key = _cache_key(loop_id, loop_type, max_chars)
    now = time.monotonic()
    if not force_refresh:
        cached = _FACT_CACHE.get(key)
        if cached and cached[0] > now:
            payload = _clone_payload(cached[1])
            payload["cache"] = {
                "hit": True,
                "ttl_s": round(max(0.0, cached[0] - now), 1),
            }
            return payload

    try:
        context = await asyncio.wait_for(
            fetch_loop_ontology_context_via_mcp(
                loop_name=loop_id,
                loop_type=loop_type,
                max_chars=max_chars,
            ),
            timeout=ONTOLOGY_FACT_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        context = {
            "source": "registered_mcp_tool",
            "error": f"ontology MCP query timeout after {ONTOLOGY_FACT_TIMEOUT_S:.0f}s",
            "content": "",
        }
    except Exception as exc:
        context = {
            "source": "registered_mcp_tool",
            "error": str(exc),
            "content": "",
        }
    payload = _parse_ontology_facts(loop_id, loop_type, context)
    has_error = bool((payload.get("raw_context") or {}).get("error"))
    ttl = ONTOLOGY_FACT_ERROR_CACHE_TTL_S if has_error else ONTOLOGY_FACT_CACHE_TTL_S
    _FACT_CACHE[key] = (time.monotonic() + ttl, _clone_payload(payload))
    payload["cache"] = {"hit": False, "ttl_s": ttl}
    return payload


def _rule(
    rule_id: str,
    title: str,
    status: str,
    *,
    severity: str = "info",
    evidence: list[Any] | None = None,
    missing_fields: list[str] | None = None,
    action: str = "",
    blocking: bool = False,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "title": title,
        "status": status,
        "severity": severity,
        "blocking": blocking,
        "evidence": evidence or [],
        "missing_fields": missing_fields or [],
        "action": action,
    }


def _data_quality_rule(features: dict[str, Any], monitoring: dict[str, Any]) -> dict[str, Any]:
    profile = features.get("data_profile") or {}
    row_count = int(profile.get("row_count") or profile.get("valid_row_count") or 0)
    irregular = _to_float(profile.get("irregular_sample_ratio")) or 0.0
    long_gaps = int(profile.get("long_gap_count") or 0)
    data_health = _dig(monitoring, "monitoring", "data_health") or {}
    health_status = str(data_health.get("status") or "").lower()
    health_score = _to_float(data_health.get("score"))
    evidence = [
        {"row_count": row_count},
        {"irregular_sample_ratio": irregular},
        {"long_gap_count": long_gaps},
        {"data_health_status": health_status or None, "score": health_score},
    ]
    if row_count < 30:
        return _rule(
            "DATA-001",
            "数据质量准入",
            "blocked",
            severity="high",
            evidence=evidence,
            action="当前窗口样本量不足，不能进入 Harris/CPK/辨识评估。",
            blocking=True,
        )
    if irregular > 0.2 or long_gaps > 0 or health_status in {"warning", "alarm", "poor"}:
        return _rule(
            "DATA-001",
            "数据质量准入",
            "warn",
            severity="medium",
            evidence=evidence,
            action="建议优先选择采样连续、缺失少、无长间断的稳定窗口。",
        )
    return _rule("DATA-001", "数据质量准入", "pass", evidence=evidence, action="数据质量满足基础计算要求。")


def _non_tuning_fault_rule(features: dict[str, Any]) -> dict[str, Any]:
    constraint = features.get("constraint_raw") or {}
    actuator = features.get("actuator_profile") or {}
    oscillation = features.get("oscillation_raw") or {}
    noise = features.get("noise_raw") or {}
    mv_sat = _to_float(constraint.get("mv_saturation_ratio")) or 0.0
    deadband = max(
        _to_float(actuator.get("mv_deadband_hint_ratio")) or 0.0,
        _to_float(actuator.get("mv_deadband_lagged_ratio")) or 0.0,
    )
    stiction = bool(actuator.get("mv_stiction_hint"))
    noise_ratio = _to_float(noise.get("pv_noise_ratio")) or 0.0
    osc = bool(oscillation.get("detected"))
    evidence = [
        {"mv_saturation_ratio": mv_sat},
        {"mv_deadband_ratio": deadband},
        {"mv_stiction_hint": stiction},
        {"pv_noise_ratio": noise_ratio},
        {"oscillation_detected": osc},
    ]
    if mv_sat >= 0.4 or stiction:
        return _rule(
            "FAULT-001",
            "非整定问题排查",
            "blocked",
            severity="high",
            evidence=evidence,
            action="优先检查阀门/执行机构或 MV 约束，当前问题不应直接靠 PID 整定解决。",
            blocking=True,
        )
    if mv_sat >= 0.1 or deadband >= 0.5 or noise_ratio >= 0.08 or osc:
        return _rule(
            "FAULT-001",
            "非整定问题排查",
            "warn",
            severity="medium",
            evidence=evidence,
            action="整定前建议复核阀门余量、死区、噪声和周期扰动。",
        )
    return _rule("FAULT-001", "非整定问题排查", "pass", evidence=evidence, action="未发现明显非整定硬故障。")


def _cpk_rule(cpk_result: dict[str, Any], spec_limits: dict[str, Any]) -> dict[str, Any]:
    cpk = cpk_result.get("cpk") or {}
    value = _to_float(cpk.get("value"))
    limits = cpk_result.get("limits") or {}
    evidence = [
        {"cpk": value, "level": cpk.get("level")},
        {"lsl": limits.get("lsl"), "usl": limits.get("usl"), "source": limits.get("source")},
    ]
    if cpk_result.get("success"):
        status = "pass" if value is not None and value >= 1.0 else "warn"
        return _rule(
            "CPK-001",
            "过程能力 CPK 评估",
            status,
            severity="info" if status == "pass" else "medium",
            evidence=evidence,
            action="CPK 已按 PV 规格上下限计算，可作为过程能力指标参与评审。",
        )
    missing = []
    if spec_limits.get("lsl") is None:
        missing.append("PV_LSL")
    if spec_limits.get("usl") is None:
        missing.append("PV_USL")
    return _rule(
        "CPK-001",
        "过程能力 CPK 评估",
        "unknown",
        severity="medium",
        evidence=evidence,
        missing_fields=missing,
        action="请在本体或历史数据中补充 PV 规格上下限；不要用报警限代替规格限。",
    )


def _harris_rule(harris_result: dict[str, Any]) -> dict[str, Any]:
    harris = harris_result.get("harris") or {}
    eta = _to_float(harris.get("eta"))
    confidence = _to_float(harris.get("confidence"))
    evidence = [
        {"eta": eta, "level": harris.get("level"), "confidence": confidence},
        {"error_basis": harris.get("error_basis"), "abort_reason": harris.get("abort_reason")},
    ]
    if not harris_result.get("success"):
        return _rule(
            "HARRIS-001",
            "Harris 最小方差性能评估",
            "unknown",
            severity="medium",
            evidence=evidence,
            action="Harris 未完成，需检查误差信号、死时间和 AR 残差模型。",
        )
    status = "pass" if eta is not None and eta >= 0.6 else "warn"
    action = "Harris 指标接近最小方差性能。" if status == "pass" else "Harris 指标偏低，说明当前闭环仍有较大优化空间。"
    if harris.get("error_basis") != "deviation_from_sp":
        action += " 当前未使用 PV-SP 跟踪误差，解释时需要注明误差信号口径。"
    return _rule("HARRIS-001", "Harris 最小方差性能评估", status, severity="info" if status == "pass" else "medium", evidence=evidence, action=action)


def _relationship_rule(ontology_facts: dict[str, Any]) -> dict[str, Any]:
    hints = ontology_facts.get("relation_hints") or []
    if hints:
        return _rule(
            "REL-001",
            "前馈/串级/MISO 关系识别",
            "warn",
            severity="medium",
            evidence=[{"relation_hints": hints}],
            action="本体提示该回路可能存在复合关系，窗口候选和整定任务应按关系链确认输入输出。",
        )
    return _rule(
        "REL-001",
        "前馈/串级/MISO 关系识别",
        "unknown",
        severity="info",
        missing_fields=["control_loop_relation"],
        action="本体未返回前馈、串级或 MISO 关系；按普通 SISO 回路处理前请确认关系表。",
    )


async def evaluate_loop_ontology_rules(
    loop_id: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    loop = get_loop(loop_id)
    if not loop:
        return {"error": "loop_id not found"}
    loop_type = str(loop.get("loop_type") or "unknown")

    ontology_facts = await resolve_loop_ontology_facts(
        loop_id=loop_id,
        loop_type=loop_type,
        force_refresh=force_refresh,
    )
    spec_limits = ontology_facts.get("pv_spec_limits") or {}
    cpk_spec_limits = spec_limits if spec_limits.get("lsl") is not None and spec_limits.get("usl") is not None else None

    bundle = get_loop_profile_bundle(loop_id, start_time=start_time, end_time=end_time)
    if bundle.get("error"):
        return bundle
    features = bundle.get("features") or {}
    monitoring = bundle.get("monitoring") or {}
    assessment = bundle.get("assessment") or assess_loop(loop_id, start_time=start_time, end_time=end_time)
    cpk_result = compute_loop_cpk(loop_id, start_time=start_time, end_time=end_time, spec_limits=cpk_spec_limits)
    harris_result = compute_loop_harris(loop_id, start_time=start_time, end_time=end_time)

    rules = [
        _rule(
            "CASE-001",
            "工况与参数集锁定",
            "pass" if ontology_facts.get("case_id") else "unknown",
            severity="info",
            evidence=[{"case_id": ontology_facts.get("case_id")}],
            missing_fields=[] if ontology_facts.get("case_id") else ["operating_case.case_id"],
            action="已按当前工况参数集评估。" if ontology_facts.get("case_id") else "本体未返回工况参数集，后续整定参数不能跨工况混用。",
        ),
        _data_quality_rule(features, monitoring),
        _non_tuning_fault_rule(features),
        _cpk_rule(cpk_result, spec_limits),
        _harris_rule(harris_result),
        _relationship_rule(ontology_facts),
    ]
    blocking = [item for item in rules if item.get("blocking")]
    warning = [item for item in rules if item.get("status") in {"warn", "unknown"}]
    return {
        "loop_id": loop_id,
        "loop_type": loop_type,
        "start_time": start_time,
        "end_time": end_time,
        "rule_pack_version": RULE_PACK_VERSION,
        "ontology_facts": ontology_facts,
        "rules": rules,
        "summary": {
            "decision": "blocked" if blocking else "review_required" if warning else "pass",
            "blocking_count": len(blocking),
            "warning_count": len(warning),
            "advisory_only": True,
            "message": "规则评估只作为工程评审和提示，不作为强约束自动拦截整定。",
        },
        "metrics": {
            "cpk": cpk_result,
            "harris": harris_result,
            "assessment": assessment,
        },
    }
