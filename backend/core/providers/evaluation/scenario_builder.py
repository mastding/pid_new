"""Loop-aware simulation scenario generation for PID evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np


_TYPE_PRESETS: dict[str, dict[str, Any]] = {
    "flow": {
        "label": "流量",
        "step_ratio": 0.03,
        "min_excitation_pct": 1.0,
        "safety_margin_ratio": 0.30,
        "horizon_t": 12.0,
        "min_duration_s": 120.0,
        "max_duration_s": 1200.0,
        "focus": ["快速响应", "低超调", "抑制振荡", "MV动作不过度"],
    },
    "pressure": {
        "label": "压力",
        "step_ratio": 0.03,
        "min_excitation_pct": 0.5,
        "safety_margin_ratio": 0.25,
        "horizon_t": 10.0,
        "min_duration_s": 240.0,
        "max_duration_s": 1800.0,
        "focus": ["低超调", "快速恢复", "抑制振荡", "避免约束触碰"],
    },
    "temperature": {
        "label": "温度",
        "step_ratio": 0.02,
        "min_excitation_pct": 0.5,
        "safety_margin_ratio": 0.20,
        "horizon_t": 10.0,
        "min_duration_s": 1800.0,
        "max_duration_s": 7200.0,
        "focus": ["稳态误差", "长尾振荡", "MV动作量", "鲁棒性"],
    },
    "level": {
        "label": "液位",
        "step_ratio": 0.03,
        "min_excitation_pct": 1.0,
        "safety_margin_ratio": 0.20,
        "horizon_t": 8.0,
        "min_duration_s": 2400.0,
        "max_duration_s": 7200.0,
        "focus": ["上下限裕度", "积分累积", "MV饱和", "平稳恢复"],
    },
    "composition": {
        "label": "成分/质量",
        "step_ratio": 0.01,
        "min_excitation_pct": 0.2,
        "safety_margin_ratio": 0.15,
        "horizon_t": 12.0,
        "min_duration_s": 3600.0,
        "max_duration_s": 10800.0,
        "focus": ["保守响应", "稳态偏差", "鲁棒性", "少扰动"],
    },
    "quality": {
        "label": "成分/质量",
        "step_ratio": 0.01,
        "min_excitation_pct": 0.2,
        "safety_margin_ratio": 0.15,
        "horizon_t": 12.0,
        "min_duration_s": 3600.0,
        "max_duration_s": 10800.0,
        "focus": ["保守响应", "稳态偏差", "鲁棒性", "少扰动"],
    },
}

_DEFAULT_PRESET = {
    "label": "通用",
    "step_ratio": 0.03,
    "min_excitation_pct": 0.5,
    "safety_margin_ratio": 0.25,
    "horizon_t": 10.0,
    "min_duration_s": 600.0,
    "max_duration_s": 3600.0,
    "focus": ["闭环稳定", "低超调", "MV约束", "鲁棒性"],
}


def _finite_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _first_number(*values: Any) -> float | None:
    for value in values:
        v = _finite_float(value)
        if v is not None:
            return v
    return None


def _range_pair(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _finite_float(value[0]), _finite_float(value[1])
    return None, None


def _data_profile_from_context(context: dict[str, Any] | None) -> dict[str, Any]:
    ctx = (context or {}).get("ctx") if isinstance(context, dict) else None
    if ctx is not None and isinstance(getattr(ctx, "data_profile", None), dict):
        return dict(ctx.data_profile)
    return {}


def build_simulation_scenarios(
    *,
    loop_type: str,
    model_params: dict[str, Any],
    dt: float,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create loop-aware SP test scenarios from ontology hints and history stats."""

    profile = _data_profile_from_context(context)
    normalized = (loop_type or "unknown").strip().lower()
    preset = dict(_TYPE_PRESETS.get(normalized, _DEFAULT_PRESET))

    window_policy = profile.get("window_policy") if isinstance(profile.get("window_policy"), dict) else {}
    ontology_facts = window_policy.get("ontology_facts") if isinstance(window_policy.get("ontology_facts"), dict) else {}

    pv_stats = profile.get("pv_stats") if isinstance(profile.get("pv_stats"), dict) else {}
    sp_stats = profile.get("sp_stats") if isinstance(profile.get("sp_stats"), dict) else {}
    scale_profile = profile.get("scale_profile") if isinstance(profile.get("scale_profile"), dict) else {}
    performance_raw = profile.get("performance_raw") if isinstance(profile.get("performance_raw"), dict) else {}

    pv_scale = _first_number(
        _nested(scale_profile, "pv", "effective_span"),
        _nested(scale_profile, "pv", "observed_span"),
        pv_stats.get("span"),
    )
    lsl = _first_number(performance_raw.get("cpk_lsl"), _nested(profile, "pv_spec_limits", "lsl"))
    usl = _first_number(performance_raw.get("cpk_usl"), _nested(profile, "pv_spec_limits", "usl"))
    if (pv_scale is None or pv_scale <= 1e-9) and lsl is not None and usl is not None and usl > lsl:
        pv_scale = usl - lsl

    operating_point = _first_number(
        sp_stats.get("median") if sp_stats.get("available") else None,
        sp_stats.get("mean") if sp_stats.get("available") else None,
        pv_stats.get("median"),
        pv_stats.get("mean"),
        50.0,
    ) or 50.0
    if pv_scale is None or pv_scale <= 1e-9:
        pv_scale = max(abs(operating_point) * 0.10, 10.0)

    type_step = float(preset["step_ratio"]) * pv_scale
    historical_step = _first_number(
        sp_stats.get("p95_abs_step"),
        sp_stats.get("median_abs_step"),
        pv_stats.get("p95_abs_step"),
    )
    if historical_step is not None and historical_step <= max(1e-9, 0.001 * pv_scale):
        historical_step = None

    min_excitation_pct = _first_number(
        window_policy.get("min_sp_excitation"),
        window_policy.get("min_mv_excitation"),
        ontology_facts.get("min_excitation_pct"),
        preset["min_excitation_pct"],
    ) or float(preset["min_excitation_pct"])
    min_excitation = max(0.0, min_excitation_pct / 100.0 * pv_scale)

    upper_room = usl - operating_point if usl is not None else None
    lower_room = operating_point - lsl if lsl is not None else None
    finite_rooms = [room for room in [upper_room, lower_room] if room is not None and room > 0]
    nearest_room = min(finite_rooms) if finite_rooms else None
    safety_limited_step = None
    if nearest_room is not None:
        safety_limited_step = float(preset["safety_margin_ratio"]) * nearest_room

    candidates = [type_step]
    if historical_step is not None:
        candidates.append(historical_step)
    if safety_limited_step is not None:
        candidates.append(safety_limited_step)
    target_step = max(0.0, min(v for v in candidates if v is not None and v > 0))
    constrained_by_safety = target_step < min_excitation and safety_limited_step is not None
    if not constrained_by_safety:
        target_step = max(target_step, min_excitation)
    if target_step <= 1e-9:
        target_step = max(1.0, 0.03 * pv_scale)

    if upper_room is not None and lower_room is not None:
        primary_direction = 1.0 if upper_room >= lower_room else -1.0
    else:
        primary_direction = 1.0

    sp_initial = operating_point
    sp_final = operating_point + primary_direction * target_step
    reverse_initial = operating_point
    reverse_final = operating_point - primary_direction * target_step

    t1 = _first_number(model_params.get("T1"), model_params.get("T"), 10.0) or 10.0
    dead_time = _first_number(model_params.get("L"), 0.0) or 0.0
    ont_t_lo, ont_t_hi = _range_pair(window_policy.get("expected_time_constant_range_s"))
    ont_l_lo, ont_l_hi = _range_pair(window_policy.get("expected_dead_time_range_s"))
    if ont_t_hi is not None and t1 < ont_t_hi * 0.25:
        horizon_t = max(float(preset["horizon_t"]), 0.5 * ont_t_hi / max(t1, 1e-6))
    else:
        horizon_t = float(preset["horizon_t"])

    duration_s = max(float(preset["min_duration_s"]), horizon_t * max(t1, 1e-6) + 3.0 * max(dead_time, 0.0))
    duration_s = min(float(preset["max_duration_s"]), duration_s)
    sim_dt = max(0.05, min(float(dt), max(t1, 1e-6) / 10.0))
    n_steps = min(50000, max(500, int(np.ceil(duration_s / sim_dt))))
    duration_s = n_steps * sim_dt

    basis_parts = [
        f"{preset['label']}回路模板",
        "历史SP/PV尺度",
    ]
    if lsl is not None and usl is not None:
        basis_parts.append("PV规格上下限")
    if window_policy:
        basis_parts.append("本体窗口策略")

    warning = ""
    if constrained_by_safety:
        warning = "安全裕度小于本体最小激励，按安全裕度生成保守场景，评估结论不应直接视为上线许可。"

    scenarios = [
        {
            "id": "nominal_sp_step",
            "label": "标称设定值阶跃",
            "role": "primary",
            "sp_initial": round(sp_initial, 6),
            "sp_final": round(sp_final, 6),
            "step_size": round(abs(sp_final - sp_initial), 6),
            "duration_s": round(duration_s, 3),
            "n_steps": n_steps,
            "dt": round(sim_dt, 6),
        },
        {
            "id": "reverse_sp_step",
            "label": "反向设定值阶跃",
            "role": "reverse",
            "sp_initial": round(reverse_initial, 6),
            "sp_final": round(reverse_final, 6),
            "step_size": round(abs(reverse_final - reverse_initial), 6),
            "duration_s": round(duration_s, 3),
            "n_steps": n_steps,
            "dt": round(sim_dt, 6),
        },
        {
            "id": "robustness_worst_case",
            "label": "模型扰动鲁棒性",
            "role": "robustness",
            "sp_initial": round(sp_initial, 6),
            "sp_final": round(sp_final, 6),
            "step_size": round(abs(sp_final - sp_initial), 6),
            "duration_s": round(duration_s, 3),
            "n_steps": n_steps,
            "dt": round(sim_dt, 6),
        },
    ]

    return {
        "source": "loop_aware",
        "loop_type": normalized,
        "loop_label": preset["label"],
        "basis": "、".join(basis_parts),
        "focus": preset["focus"],
        "operating_point": round(operating_point, 6),
        "pv_scale": round(pv_scale, 6),
        "step_size": round(target_step, 6),
        "step_percent_of_span": round(target_step / max(pv_scale, 1e-9) * 100.0, 3),
        "min_excitation": round(min_excitation, 6),
        "min_excitation_pct": round(min_excitation_pct, 3),
        "safety_limited_step": round(safety_limited_step, 6) if safety_limited_step is not None else None,
        "historical_step": round(historical_step, 6) if historical_step is not None else None,
        "constraints": {
            "lsl": lsl,
            "usl": usl,
            "upper_room": round(upper_room, 6) if upper_room is not None else None,
            "lower_room": round(lower_room, 6) if lower_room is not None else None,
            "constrained_by_safety": constrained_by_safety,
        },
        "ontology_inputs": {
            "expected_time_constant_range_s": [ont_t_lo, ont_t_hi] if ont_t_lo is not None or ont_t_hi is not None else None,
            "expected_dead_time_range_s": [ont_l_lo, ont_l_hi] if ont_l_lo is not None or ont_l_hi is not None else None,
            "process_direction": ontology_facts.get("process_direction") or window_policy.get("expected_gain_sign"),
        },
        "warning": warning,
        "primary": scenarios[0],
        "reverse": scenarios[1],
        "scenarios": scenarios,
    }
