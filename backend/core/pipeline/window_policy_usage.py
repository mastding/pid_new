"""Expose how WindowSelectionPolicy fields are consumed by window providers."""
from __future__ import annotations

from typing import Any


WINDOW_POLICY_FIELD_LABELS: dict[str, str] = {
    "preferred_algorithm_families": "优先算法族",
    "deprioritized_algorithm_families": "降权算法族",
    "disabled_algorithm_families": "禁用算法族",
    "algorithm_plan": "算法族执行计划",
    "min_mv_excitation": "最小 MV 激励",
    "min_sp_excitation": "最小 SP 激励",
    "min_pv_response": "最小 PV 响应",
    "max_mv_saturation_ratio": "最大 MV 饱和比例",
    "max_pv_noise_ratio": "最大 PV 噪声比例",
    "max_drift_ratio": "最大 PV 漂移比例",
    "expected_dead_time_range_s": "预期纯滞后范围",
    "expected_time_constant_range_s": "预期时间常数范围",
    "expected_gain_sign": "预期过程增益方向",
    "min_window_points": "最小窗口点数",
    "min_window_duration_s": "最小窗口时长",
    "max_window_points": "最大窗口点数",
    "pre_window_s": "事件前窗口长度",
    "post_window_s": "事件后窗口长度",
    "steady_scan_window_s": "稳态扫描窗口长度",
    "steady_scan_step_s": "稳态扫描步长",
    "merge_gap_s": "事件合并间隔",
    "max_candidates_per_family": "每个算法族最大候选数",
    "allowed_operating_states": "允许工况",
    "avoid_operating_states": "规避工况",
    "scoring_weights": "策略评分权重",
    "hard_guards": "硬性准入规则",
    "soft_penalties": "软性扣分规则",
    "rationale": "策略说明",
    "ontology_facts": "本体事实",
}


FAMILY_CONSUMED_FIELDS: dict[str, list[str]] = {
    "sp_step": [
        "algorithm_plan",
        "preferred_algorithm_families",
        "deprioritized_algorithm_families",
        "disabled_algorithm_families",
        "min_sp_excitation",
        "pre_window_s",
        "post_window_s",
        "max_window_points",
        "merge_gap_s",
        "min_window_points",
        "min_pv_response",
        "max_mv_saturation_ratio",
        "max_drift_ratio",
    ],
    "mv_step": [
        "algorithm_plan",
        "preferred_algorithm_families",
        "deprioritized_algorithm_families",
        "disabled_algorithm_families",
        "min_mv_excitation",
        "pre_window_s",
        "post_window_s",
        "max_window_points",
        "merge_gap_s",
        "min_window_points",
        "min_pv_response",
        "max_mv_saturation_ratio",
        "max_drift_ratio",
    ],
    "mv_ramp": [
        "algorithm_plan",
        "preferred_algorithm_families",
        "deprioritized_algorithm_families",
        "disabled_algorithm_families",
        "post_window_s",
        "pre_window_s",
        "max_window_points",
        "merge_gap_s",
        "max_candidates_per_family",
        "min_window_points",
        "min_mv_excitation",
        "min_pv_response",
        "max_mv_saturation_ratio",
        "max_drift_ratio",
    ],
    "steady_disturbance": [
        "algorithm_plan",
        "preferred_algorithm_families",
        "deprioritized_algorithm_families",
        "disabled_algorithm_families",
        "steady_scan_window_s",
        "steady_scan_step_s",
        "merge_gap_s",
        "max_candidates_per_family",
        "min_window_points",
        "min_mv_excitation",
        "min_pv_response",
        "max_mv_saturation_ratio",
        "max_drift_ratio",
    ],
    "rolling_scan": [
        "algorithm_plan",
        "preferred_algorithm_families",
        "deprioritized_algorithm_families",
        "disabled_algorithm_families",
        "pre_window_s",
        "post_window_s",
        "max_window_points",
        "min_window_points",
        "min_pv_response",
        "max_mv_saturation_ratio",
        "max_drift_ratio",
    ],
}


DOWNSTREAM_HINT_FIELDS = {
    "expected_dead_time_range_s",
    "expected_time_constant_range_s",
    "expected_gain_sign",
}


DISPLAY_ONLY_FIELDS = {
    "max_pv_noise_ratio",
    "min_window_duration_s",
    "allowed_operating_states",
    "avoid_operating_states",
    "scoring_weights",
    "hard_guards",
    "soft_penalties",
    "rationale",
    "ontology_facts",
}


FIELD_USAGE_NOTES: dict[str, str] = {
    "algorithm_plan": "控制每个算法族是否运行、降权或禁用。",
    "preferred_algorithm_families": "参与算法族计划生成，并用于策略一致性评分。",
    "deprioritized_algorithm_families": "参与算法族计划生成，并对对应候选施加软性扣分。",
    "disabled_algorithm_families": "阻止禁用算法族运行，并硬性过滤对应候选。",
    "min_mv_excitation": "提高 MV 阶跃/稳态扰动的激励阈值，过滤 MV 激励不足的窗口。",
    "min_sp_excitation": "提高 SP 阶跃检测阈值。",
    "min_pv_response": "过滤 PV 响应低于策略阈值的窗口。",
    "max_mv_saturation_ratio": "过滤稳态扰动窗口，并对饱和候选进行扣分或提示。",
    "max_drift_ratio": "对慢漂移主导的窗口进行扣分或过滤。",
    "min_window_points": "控制稳态扫描最小长度，并过滤过短候选窗口。",
    "max_window_points": "限制阶跃、斜坡和兜底窗口的事件后截取长度。",
    "pre_window_s": "控制事件窗口的前置基线长度。",
    "post_window_s": "控制事件窗口的后置时长，并影响 MV 斜坡检测窗口。",
    "steady_scan_window_s": "控制稳态扰动滚动扫描窗口长度。",
    "steady_scan_step_s": "控制稳态扰动滚动扫描步长。",
    "merge_gap_s": "控制跨算法族事件合并距离。",
    "max_candidates_per_family": "限制 MV 斜坡和稳态扰动算法族的候选数量。",
    "expected_dead_time_range_s": "作为辨识/模型评审上下文下传，当前窗口生成器不直接消费。",
    "expected_time_constant_range_s": "作为辨识/模型评审上下文下传，当前窗口生成器不直接消费。",
    "expected_gain_sign": "作为模型合理性上下文下传，当前窗口生成器不直接消费。",
    "max_pv_noise_ratio": "用于审计和页面展示，当前确定性窗口算法暂不直接消费。",
    "min_window_duration_s": "用于审计和页面展示，具体窗口长度当前由前后置窗口和稳态扫描参数决定。",
    "allowed_operating_states": "score_window 会识别候选窗口 operating_state，并过滤不在本体允许集合内的窗口。",
    "avoid_operating_states": "score_window 会识别候选窗口 operating_state，并阻断或扣分本体规避工况中的窗口。",
}


def _field_consumers() -> dict[str, list[str]]:
    consumers: dict[str, list[str]] = {}
    for family, fields in FAMILY_CONSUMED_FIELDS.items():
        for field in fields:
            consumers.setdefault(field, []).append(family)
    return consumers


def build_policy_field_usage() -> list[dict[str, Any]]:
    """Return an auditable field-to-provider consumption map."""
    consumers = _field_consumers()
    rows: list[dict[str, Any]] = []
    fields = list(WINDOW_POLICY_FIELD_LABELS)
    for field in fields:
        used_by = consumers.get(field, [])
        if used_by:
            status = "consumed"
        elif field in DOWNSTREAM_HINT_FIELDS:
            status = "downstream_hint"
        else:
            status = "display_only"
        rows.append({
            "field": field,
            "label": WINDOW_POLICY_FIELD_LABELS.get(field, field),
            "status": status,
            "consumed_by": used_by,
            "note": FIELD_USAGE_NOTES.get(field, ""),
        })
    return rows


def enrich_policy_field_usage(policy: dict[str, Any]) -> dict[str, Any]:
    """Attach field usage metadata to a policy dict and its algorithm plan."""
    enriched = dict(policy)
    field_usage = build_policy_field_usage()
    enriched["field_usage"] = field_usage
    plan = []
    for item in enriched.get("algorithm_plan") or []:
        if not isinstance(item, dict):
            continue
        family = str(item.get("family") or "")
        plan_item = dict(item)
        consumed_fields = FAMILY_CONSUMED_FIELDS.get(family, [])
        plan_item["consumed_policy_fields"] = consumed_fields
        plan_item["consumed_policy_field_labels"] = [
            WINDOW_POLICY_FIELD_LABELS.get(field, field) for field in consumed_fields
        ]
        plan.append(plan_item)
    enriched["algorithm_plan"] = plan
    return enriched
