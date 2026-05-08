"""Build a structured window-selection policy from loop profile and ontology."""
from __future__ import annotations

from typing import Any

from core.pipeline.window_policy_models import OntologyFacts, WindowSelectionPolicy
from core.pipeline.window_policy_usage import enrich_policy_field_usage


_LOOP_TYPE_PRIORS: dict[str, dict[str, Any]] = {
    "flow": {
        "T": (1.0, 30.0),
        "L": (0.0, 30.0),
        "min_excitation": 1.0,
        "min_duration": 180.0,
        "families": ["mv_step", "mv_ramp", "sp_step", "steady_disturbance"],
    },
    "pressure": {
        "T": (5.0, 300.0),
        "L": (0.0, 120.0),
        "min_excitation": 0.5,
        "min_duration": 300.0,
        "families": ["mv_step", "mv_ramp", "sp_step", "steady_disturbance"],
    },
    "temperature": {
        "T": (30.0, 1800.0),
        "L": (0.0, 600.0),
        "min_excitation": 0.5,
        "min_duration": 900.0,
        "families": ["mv_ramp", "mv_step", "sp_step", "steady_disturbance"],
    },
    "level": {
        "T": (60.0, 3600.0),
        "L": (0.0, 900.0),
        "min_excitation": 1.0,
        "min_duration": 1200.0,
        "families": ["mv_ramp", "steady_disturbance", "mv_step", "sp_step"],
    },
}

_ALL_WINDOW_ALGORITHM_FAMILIES = [
    "mv_step",
    "mv_ramp",
    "sp_step",
    "steady_disturbance",
    "rolling_scan",
]


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _detect_gain_sign(text: str) -> str:
    lowered = text.lower()
    positive_hits = ["正作用", "正增益", "mv增", "mv 增", "pv增", "pv 增", "positive"]
    negative_hits = ["反作用", "负增益", "mv增pv降", "mv 增 pv 降", "negative"]
    if any(token in lowered for token in negative_hits):
        return "negative"
    if any(token in lowered for token in positive_hits):
        return "positive"
    return "unknown"


def _derive_window_algorithm_inputs(prior: dict[str, Any]) -> dict[str, Any]:
    """Translate process time-scale priors into deterministic window inputs."""
    t_low, t_high = prior["T"]
    l_low, l_high = prior["L"]
    min_duration = float(prior["min_duration"])

    # Identification windows should generally cover the dead time plus several
    # dominant time constants. The values are upper-bounded to keep historical
    # scans responsive when ontology returns broad ranges.
    post_s = max(min_duration, 4.0 * float(t_high) + float(l_high))
    post_s = min(post_s, 7200.0)
    pre_s = max(30.0, min(max(float(l_high), post_s * 0.12), 600.0))

    steady_scan_s = max(min_duration, 3.0 * float(t_high) + float(l_high))
    steady_scan_s = min(steady_scan_s, 7200.0)
    steady_step_s = max(30.0, min(steady_scan_s / 4.0, 600.0))

    merge_gap_s = max(60.0, min(max(float(l_high), 180.0), 900.0))
    max_window_points = int(max(80, min(round(post_s / 1.0), 2000)))
    return {
        "pre_window_s": round(pre_s, 3),
        "post_window_s": round(post_s, 3),
        "steady_scan_window_s": round(steady_scan_s, 3),
        "steady_scan_step_s": round(steady_step_s, 3),
        "merge_gap_s": round(merge_gap_s, 3),
        "max_window_points": max_window_points,
    }


def _build_algorithm_plan(
    *,
    preferred: list[str],
    deprioritized: list[str],
    disabled: list[str],
    loop_type: str,
) -> list[dict[str, Any]]:
    preferred_set = set(preferred)
    deprioritized_set = set(deprioritized)
    disabled_set = set(disabled)
    ordered = list(dict.fromkeys([*preferred, *_ALL_WINDOW_ALGORITHM_FAMILIES, *deprioritized, *disabled]))
    plan: list[dict[str, Any]] = []
    for family in ordered:
        if family in disabled_set:
            state = "disabled"
            reason = "本体/策略明确不建议该算法族参与正式窗口候选"
        elif family in deprioritized_set:
            state = "deprioritized"
            reason = "可作为兜底诊断窗口，但正式辨识时降低优先级"
        elif family in preferred_set:
            state = "preferred"
            reason = f"{loop_type or 'unknown'} 回路优先从该算法族中寻找可解释激励片段"
        else:
            state = "available"
            reason = "允许参与候选，但不作为本体策略优先推荐"
        plan.append({
            "family": family,
            "state": state,
            "reason": reason,
        })
    return plan


def build_window_selection_policy(
    *,
    loop_name: str,
    loop_type: str,
    data_profile: dict[str, Any] | None = None,
    mcp_context: dict[str, Any] | None = None,
    frontend_context: str | None = None,
) -> dict[str, Any]:
    """Return a serializable phase-1 policy for window selection.

    This intentionally starts conservative: it exposes ontology/profile-derived
    constraints to the UI and later stages, but does not yet rewrite the
    deterministic window generation algorithm.
    """
    normalized_type = (loop_type or "").strip().lower()
    prior = _LOOP_TYPE_PRIORS.get(normalized_type, _LOOP_TYPE_PRIORS["flow"])
    algorithm_inputs = _derive_window_algorithm_inputs(prior)
    raw_answer = ""
    source = "none"
    confidence = 0.35
    evidence: list[dict[str, Any]] = []

    if mcp_context and mcp_context.get("content"):
        raw_answer = str(mcp_context.get("content", ""))
        source = "mcp"
        confidence = 0.72
        evidence.append({
            "fact": "已从注册 MCP 工具获取回路本体上下文",
            "source": str(mcp_context.get("server_name") or "MCP"),
        })
    elif frontend_context:
        raw_answer = frontend_context
        source = "frontend"
        confidence = 0.55
        evidence.append({
            "fact": "已使用前端图谱/本体上下文",
            "source": "frontend_ontology_context",
        })

    gain_sign = _detect_gain_sign(raw_answer)
    if gain_sign != "unknown":
        evidence.append({
            "fact": f"本体/上下文提示过程增益方向为 {gain_sign}",
            "source": source,
        })

    profile = data_profile or {}
    quality = profile.get("data_quality") if isinstance(profile.get("data_quality"), dict) else {}
    if quality:
        evidence.append({
            "fact": "已合并历史数据画像中的数据质量摘要",
            "source": "LoopProfile",
        })

    facts = OntologyFacts(
        loop_id=loop_name or "",
        source=source if source in {"mcp", "frontend"} else "default",
        confidence=confidence,
        process_direction=gain_sign,  # type: ignore[arg-type]
        expected_dead_time_range_s=prior["L"],
        expected_time_constant_range_s=prior["T"],
        min_excitation_pct=prior["min_excitation"],
        max_noise_ratio=0.25,
        avoid_conditions=[
            "MV 长时间饱和或贴边",
            "PV 被明显外部扰动主导",
            "工况频繁切换或负荷快速变化",
            "窗口点数过少或激励不足",
        ],
        evidence=evidence,
        raw_answer=raw_answer[:4000] if raw_answer else None,
    )
    preferred_families = list(prior["families"])
    deprioritized_families = ["rolling_scan"]
    disabled_families: list[str] = []

    policy = WindowSelectionPolicy(
        loop_id=loop_name or "",
        loop_type=normalized_type or loop_type,
        policy_version="phase1-default",
        confidence=confidence,
        preferred_algorithm_families=preferred_families,
        deprioritized_algorithm_families=deprioritized_families,
        disabled_algorithm_families=disabled_families,
        algorithm_plan=_build_algorithm_plan(
            preferred=preferred_families,
            deprioritized=deprioritized_families,
            disabled=disabled_families,
            loop_type=normalized_type or loop_type,
        ),
        min_mv_excitation=prior["min_excitation"],
        min_sp_excitation=prior["min_excitation"],
        max_mv_saturation_ratio=0.05,
        max_pv_noise_ratio=0.25,
        min_pv_response=None,
        max_drift_ratio=0.9,
        expected_dead_time_range_s=prior["L"],
        expected_time_constant_range_s=prior["T"],
        expected_gain_sign=gain_sign,  # type: ignore[arg-type]
        min_window_points=30,
        min_window_duration_s=float(prior["min_duration"]),
        max_window_points=algorithm_inputs["max_window_points"],
        pre_window_s=algorithm_inputs["pre_window_s"],
        post_window_s=algorithm_inputs["post_window_s"],
        steady_scan_window_s=algorithm_inputs["steady_scan_window_s"],
        steady_scan_step_s=algorithm_inputs["steady_scan_step_s"],
        merge_gap_s=algorithm_inputs["merge_gap_s"],
        max_candidates_per_family=6,
        allowed_operating_states=["stable", "mild_load_change", "steady_disturbance"],
        avoid_operating_states=[
            "hard_saturation",
            "manual_intervention",
            "strong_oscillation",
            "startup_shutdown",
        ],
        scoring_weights={
            "excitation": 0.25,
            "response": 0.25,
            "stability": 0.15,
            "ontology_consistency": 0.25,
            "constraint": 0.10,
        },
        hard_guards=[
            {"name": "no_usable_window", "action": "block_formal_identification"},
        ],
        soft_penalties=[
            {"name": "outside_typical_time_scale", "action": "decrease_confidence"},
            {"name": "gain_sign_conflict", "action": "decrease_confidence"},
        ],
        rationale=(
            "Phase 1 先根据回路类型和本体上下文生成窗口候选策略；"
            "策略中的窗口长度、扫描步长、算法族优先级和质量门槛会作为确定性窗口算法的输入。"
        ),
        ontology_facts=facts,
    )
    return enrich_policy_field_usage(_model_dump(policy))
