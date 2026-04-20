"""PID tuning rules: strategy selection + IMC / Lambda / ZN / CHR implementations.

Migrated from pid_new/backend/skills/pid_tuning_skills.py
           and pid_new/backend/services/pid_tuning_service.py

Simplification vs. pid_new:
- Experience / knowledge-graph guidance removed — handled by the LLM consultant.
- No history-seed PID blending.
- select_best_strategy() evaluates all strategies, returns best + all candidates.
"""
from __future__ import annotations

from typing import Any

_ALL_STRATEGIES = ["IMC", "LAMBDA", "ZN", "CHR"]


# ── Utilities ────────────────────────────────────────────────────────────────

def _safe_div(num: float, den: float, fallback: float = 0.0) -> float:
    return num / den if abs(den) > 1e-9 else fallback


def _clamp(Kp: float, Ki: float, Kd: float) -> dict[str, float]:
    return {
        "Kp": float(min(max(Kp, 0.0), 1e4)),
        "Ki": float(min(max(Ki, 0.0), 1e4)),
        "Kd": float(min(max(Kd, 0.0), 1e4)),
    }


# ── Per-model tuning rules ────────────────────────────────────────────────────

def tune_fo(K: float, T: float, strategy: str) -> dict[str, Any]:
    """First-order (no dead-time) tuning."""
    s = (strategy or "LAMBDA").strip().upper()
    abs_k = max(abs(float(K)), 1e-6)
    T = max(float(T), 1e-3)

    if s in {"IMC", "LAMBDA", "LAMBDA_TUNING"}:
        lam = max(0.8 * T, 1e-3)
        Kp, Ti, Td = T / (abs_k * lam), T, 0.0
        desc = "FO Lambda/IMC"
    elif s == "ZN":
        Kp, Ti, Td = 0.8 / abs_k, T, 0.0
        desc = "FO moderated ZN"
    else:
        Kp, Ti, Td = 0.7 / abs_k, max(1.1 * T, 1e-3), 0.0
        desc = "FO conservative fallback"

    p = _clamp(Kp, _safe_div(Kp, Ti), Kp * Td)
    p.update({"strategy": s, "model_type": "FO", "description": desc,
               "Ti": float(Ti), "Td": float(Td)})
    return p


def tune_fopdt(K: float, T: float, L: float, strategy: str) -> dict[str, Any]:
    """First-order plus dead-time tuning."""
    s = (strategy or "IMC").strip().upper()
    abs_k = max(abs(float(K)), 1e-6)
    T = max(float(T), 1e-3)
    L = max(float(L), 0.0)

    if s == "IMC":
        lam = max(L, 0.8 * T, 1e-3)
        Kp = T / (abs_k * (lam + L))
        Ti = max(T, 1e-3)
        Td = 0.5 * L if L > 0 else 0.0
        desc = "Conservative IMC"
    elif s in {"LAMBDA", "LAMBDA_TUNING"}:
        lam = max(1.5 * L, T, 1e-3)
        Kp = T / (abs_k * (lam + L))
        Ti = max(T + 0.5 * L, 1e-3)
        Td = 0.0
        desc = "Lambda Tuning"
    elif s == "ZN":
        el = max(L, 0.1 * T, 1e-3)
        Kp = 1.2 * T / (abs_k * el)
        Ti = 2.0 * el
        Td = 0.5 * el
        desc = "Aggressive Ziegler-Nichols"
    elif s in {"CHR", "CHR_0OS"}:
        el = max(L, 0.1 * T, 1e-3)
        Kp = 0.6 * T / (abs_k * el)
        Ti = max(T, 1e-3)
        Td = 0.5 * el
        desc = "CHR 0% Overshoot"
    else:
        lam = max(L, T, 1e-3)
        Kp = T / (abs_k * (lam + L))
        Ti = max(T, 1e-3)
        Td = 0.0
        desc = "Fallback IMC-like"

    p = _clamp(Kp, _safe_div(Kp, Ti), Kp * Td)
    p.update({"strategy": s, "model_type": "FOPDT", "description": desc,
               "Ti": float(Ti), "Td": float(Td)})
    return p


def tune_sopdt(K: float, T1: float, T2: float, L: float, strategy: str) -> dict[str, Any]:
    """Second-order plus dead-time tuning with shape-index adaptation."""
    s = (strategy or "LAMBDA").strip().upper()
    abs_k = max(abs(float(K)), 1e-6)
    T1 = max(float(T1), 1e-3)
    T2 = max(float(T2), 1e-3)
    L = max(float(L), 0.0)

    dominant = max(T1, T2)
    secondary = min(T1, T2)
    shape = min(secondary / max(dominant, 1e-6), 1.0)   # 0 = pure FOPDT, 1 = equal poles
    apparent_order = 1.0 + shape
    l_work = L + secondary * (0.35 + 0.20 * shape)
    t_work = max(dominant + secondary * (0.55 + 0.25 * shape), 1e-3)
    l_work = max(l_work, 0.0)
    td_ceil = max(0.18 * dominant + 0.30 * secondary, 0.0)

    if s in {"LAMBDA", "LAMBDA_TUNING"}:
        lam = max((1.05 + 0.30 * shape) * t_work, 2.0 * l_work, 1e-3)
        Kp = t_work / (abs_k * (lam + l_work))
        Ti = max(dominant + (1.05 + 0.35 * shape) * secondary + 0.45 * l_work, 1e-3)
        Td = min(secondary * (0.22 + 0.18 * shape), td_ceil)
        desc = "SOPDT native Lambda"
    elif s == "IMC":
        lam = max((0.90 + 0.20 * shape) * t_work, 1.6 * l_work, 1e-3)
        Kp = t_work / (abs_k * (lam + l_work))
        Ti = max(dominant + (0.75 + 0.20 * shape) * secondary + 0.25 * l_work, 1e-3)
        Td = min(secondary * (0.16 + 0.12 * shape), td_ceil)
        desc = "SOPDT native IMC"
    elif s == "ZN":
        el = max(l_work, (0.22 + 0.08 * shape) * t_work, 1e-3)
        Kp = 0.55 * t_work / (abs_k * el)
        Ti = max(2.8 * el + 0.40 * secondary, 1e-3)
        Td = min(0.40 * el, 0.35 * secondary)
        desc = "SOPDT native moderated ZN"
    else:  # CHR
        el = max(l_work, (0.20 + 0.05 * shape) * t_work, 1e-3)
        Kp = 0.38 * t_work / (abs_k * el)
        Ti = max(t_work + 0.35 * secondary, 1e-3)
        Td = min(0.30 * el, 0.25 * secondary)
        desc = "SOPDT native CHR-like"

    p = _clamp(Kp, _safe_div(Kp, Ti), Kp * Td)
    p.update({
        "strategy": s, "model_type": "SOPDT", "description": desc,
        "Ti": float(Ti), "Td": float(Td),
        "T1": float(T1), "T2": float(T2),
        "shape_index": float(shape), "apparent_order": float(apparent_order),
        "T_work": float(t_work), "L_work": float(l_work),
    })
    return p


def tune_ipdt(K: float, L: float, strategy: str) -> dict[str, Any]:
    """Integrating process plus dead-time tuning."""
    s = (strategy or "LAMBDA").strip().upper()
    abs_k = max(abs(float(K)), 1e-6)
    el = max(float(L), 1e-3)

    if s in {"LAMBDA", "LAMBDA_TUNING", "IMC"}:
        lam = max(2.5 * el, 1e-3)
        Kp = 1.0 / (abs_k * max(lam + el, 1e-3))
        Ti = max(4.0 * el, 1e-3)
        Td = 0.0
        desc = "Conservative Integrating Lambda"
    elif s == "ZN":
        Kp = 0.35 / max(abs_k * el, 1e-3)
        Ti = max(3.5 * el, 1e-3)
        Td = 0.0
        desc = "Integrating ZN-like"
    else:
        Kp = 0.30 / max(abs_k * el, 1e-3)
        Ti = max(4.0 * el, 1e-3)
        Td = 0.0
        desc = "Integrating conservative fallback"

    p = _clamp(Kp, _safe_div(Kp, Ti), Kp * Td)
    p.update({"strategy": s, "model_type": "IPDT", "description": desc,
               "Ti": float(Ti), "Td": float(Td)})
    return p


# ── Strategy selection ────────────────────────────────────────────────────────

def _heuristic_strategy(
    *,
    loop_type: str,
    model_type: str,
    K: float,
    T: float,
    L: float,
    model_params: dict,
    confidence: float,
    nrmse: float,
    r2: float,
) -> dict[str, str]:
    """Rule-based strategy selection. Returns {"strategy", "reason"}."""
    mt = model_type.upper()
    lt = loop_type.lower()

    # Very low confidence → most conservative
    if confidence < 0.35:
        return {"strategy": "IMC",
                "reason": "模型置信度极低，回退至最保守的 IMC 整定。"}

    # Moderate confidence → Lambda
    if confidence < 0.55 or nrmse > 0.1 or r2 < 0.75:
        return {"strategy": "LAMBDA",
                "reason": "模型质量一般，优先选用鲁棒的 Lambda 整定。"}

    high_quality = confidence >= 0.88 and nrmse <= 0.05 and r2 >= 0.97

    if mt == "FO":
        if lt in {"flow", "pressure"} and high_quality and T > 5.0:
            return {"strategy": "IMC",
                    "reason": "一阶过程且模型质量高，优选 FO-IMC 整定。"}
        return {"strategy": "LAMBDA",
                "reason": "一阶过程优先采用保守的 Lambda/IMC 整定。"}

    if mt == "IPDT":
        return {"strategy": "LAMBDA",
                "reason": "积分过程默认采用保守的积分-Lambda 整定。"}

    if mt == "SOPDT":
        t1 = float(model_params.get("T1", T))
        t2 = float(model_params.get("T2", T))
        l_val = float(model_params.get("L", L))
        dominant = max(t1, t2)
        secondary = min(t1, t2)
        shape = secondary / max(dominant, 1e-6)
        agg_tau = dominant + secondary
        tau_ratio = l_val / max(agg_tau, 1e-6)
        fast = agg_tau <= 5.0

        if shape >= 0.72:
            return {"strategy": "LAMBDA",
                    "reason": "二阶时间常数分布宽，Lambda 可有效抑制超调和振荡。"}
        if lt in {"temperature", "level"}:
            return {"strategy": "LAMBDA",
                    "reason": "温度/液位回路采用 SOPDT 时优先选用 Lambda 整定。"}
        if high_quality and 0.05 <= tau_ratio <= 0.30 and not fast and shape <= 0.45:
            return {"strategy": "IMC",
                    "reason": "主/次时间常数分离明显且拟合质量高，优选 SOPDT-IMC。"}
        if fast or tau_ratio < 0.05:
            return {"strategy": "IMC",
                    "reason": "过程较快或纯滞后极小，IMC 可保持良好阻尼。"}
        return {"strategy": "LAMBDA",
                "reason": "SOPDT 参数适中，Lambda 为更安全的默认选择。"}

    # FOPDT and fallback
    tau_ratio = L / max(T, 1e-6)
    fast = T <= 5.0

    if lt == "temperature":
        return {"strategy": "IMC",
                "reason": "温度回路惯性大，IMC 可有效抑制超调。"}
    if lt == "level":
        return {"strategy": "LAMBDA",
                "reason": "液位回路优先平滑性和鲁棒性，选用 Lambda。"}
    if lt == "pressure":
        strat = "IMC" if tau_ratio >= 0.3 else "LAMBDA"
        return {"strategy": strat,
                "reason": "压力回路常较快，保守策略可控制振荡。"}
    if lt == "flow":
        if high_quality and tau_ratio >= 0.08 and not fast:
            return {"strategy": "ZN",
                    "reason": "流量回路模型质量高且纯滞后足够，可尝试适度 ZN 整定。"}
        if fast or tau_ratio < 0.08:
            return {"strategy": "IMC",
                    "reason": "流量过程快或纯滞后小，IMC 抑制振荡。"}
        return {"strategy": "LAMBDA",
                "reason": "流量回路模型可用但不适合激进整定，选用 Lambda。"}

    return {"strategy": "IMC",
            "reason": "未知回路类型，使用默认鲁棒 IMC 策略。"}


def apply_tuning_rules(
    K: float,
    T: float,
    L: float,
    strategy: str,
    model_type: str = "FOPDT",
    model_params: dict | None = None,
) -> dict[str, Any]:
    """Dispatch to the correct tune_* function for (model_type, strategy)."""
    mt = (model_type or "FOPDT").strip().upper()
    mp = model_params or {}

    if mt == "FO":
        return tune_fo(K=float(mp.get("K", K)), T=float(mp.get("T", T)), strategy=strategy)
    if mt == "SOPDT":
        return tune_sopdt(
            K=float(mp.get("K", K)),
            T1=float(mp.get("T1", mp.get("T", T))),
            T2=float(mp.get("T2", mp.get("T", T))),
            L=float(mp.get("L", L)),
            strategy=strategy,
        )
    if mt == "SOPDT_UNDER":
        # 欠阻尼：用 T1=T2=T 走 SOPDT 整定路径（保守近似），上层会把策略限制为 LAMBDA
        T_val = float(mp.get("T", T))
        return tune_sopdt(
            K=float(mp.get("K", K)),
            T1=T_val,
            T2=T_val,
            L=float(mp.get("L", L)),
            strategy=strategy,
        )
    if mt == "IPDT":
        return tune_ipdt(K=float(mp.get("K", K)), L=float(mp.get("L", L)), strategy=strategy)
    if mt == "IFOPDT":
        # 积分+一阶+死时：等效死时 = L + T，走 IPDT 整定路径
        L_eff = float(mp.get("L", L)) + float(mp.get("T", T))
        return tune_ipdt(K=float(mp.get("K", K)), L=L_eff, strategy=strategy)
    # Default: FOPDT
    return tune_fopdt(
        K=float(mp.get("K", K)),
        T=float(mp.get("T", T)),
        L=float(mp.get("L", L)),
        strategy=strategy,
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def select_best_strategy(
    *,
    K: float,
    T: float,
    L: float,
    dt: float,
    loop_type: str = "flow",
    model_type: str = "FOPDT",
    model_params: dict | None = None,
    confidence: float = 1.0,
    nrmse: float = 0.0,
    r2: float = 1.0,
) -> dict[str, Any]:
    """Evaluate all strategies and return the recommended PID parameters.

    Returns:
        {
            "best": {Kp, Ki, Kd, strategy, description, Ti, Td, ...},
            "heuristic_strategy": str,
            "heuristic_reason": str,
            "all_candidates": [list of per-strategy dicts],
        }
    """
    mt = (model_type or "FOPDT").strip().upper()
    mp = model_params or {}

    heuristic = _heuristic_strategy(
        loop_type=loop_type,
        model_type=mt,
        K=K, T=T, L=L,
        model_params=mp,
        confidence=confidence,
        nrmse=nrmse,
        r2=r2,
    )
    preferred = heuristic["strategy"]

    # IPDT and SOPDT should not use ZN or CHR
    if mt == "IPDT" and preferred in {"ZN", "CHR"}:
        preferred = "LAMBDA"
    if mt == "SOPDT" and preferred == "ZN":
        preferred = "LAMBDA"
    # 新增模型的策略守卫：
    # SOPDT_UNDER 振荡型对象，必须用最保守的 LAMBDA，避免激进策略放大振荡
    if mt == "SOPDT_UNDER":
        preferred = "LAMBDA"
    # IFOPDT 积分型，禁用 ZN/CHR
    if mt == "IFOPDT" and preferred in {"ZN", "CHR"}:
        preferred = "LAMBDA"

    # Evaluate all strategies for this model type
    candidates: list[dict[str, Any]] = []
    for strat in _ALL_STRATEGIES:
        try:
            result = apply_tuning_rules(K=K, T=T, L=L, strategy=strat,
                                        model_type=mt, model_params=mp)
            result["is_recommended"] = (strat == preferred)
            candidates.append(result)
        except Exception:
            pass  # skip strategies that fail for this model type

    # 物理量级守卫：TI 不得低于该回路类型最小合理值；PB（=100/Kp）须落在 [5%, 1000%]。
    # 不达标的策略标 unreliable，且不允许充当 best。
    _TI_MIN = {"flow": 2.0, "pressure": 10.0, "temperature": 60.0, "level": 60.0}
    ti_min = _TI_MIN.get((loop_type or "").lower().strip(), 2.0)
    PB_MIN, PB_MAX = 5.0, 1000.0
    for c in candidates:
        kp = float(c.get("Kp", 0.0))
        ti = float(c.get("Ti", 0.0))
        pb = (100.0 / kp) if kp > 1e-9 else float("inf")
        reasons = []
        if ti > 0 and ti < ti_min:
            reasons.append(f"TI={ti:.1f}s 低于{loop_type}最小合理值 {ti_min:.0f}s")
        if not (PB_MIN <= pb <= PB_MAX):
            reasons.append(f"PB={pb:.2f}% 越界 [{PB_MIN},{PB_MAX}]")
        if reasons:
            c["unreliable"] = True
            c["unreliable_reason"] = "; ".join(reasons)

    reliable = [c for c in candidates if not c.get("unreliable")]
    if reliable:
        best = next((c for c in reliable if c["strategy"] == preferred), reliable[0])
        tuning_unreliable = False
        unreliable_reason = ""
    else:
        # 所有策略都越界 —— 拿原 preferred 顶上但标红，让 evaluation/前端能感知
        best = next((c for c in candidates if c["strategy"] == preferred), candidates[0] if candidates else None)
        tuning_unreliable = True
        unreliable_reason = (
            f"所有候选策略 PID 参数物理量级不合理（最小 TI={ti_min:.0f}s，PB ∈ [{PB_MIN},{PB_MAX}]%），"
            f"通常意味着辨识模型时间常数塌缩，建议沿用现场参数或重做手动阶跃测试"
        )

    return {
        "best": best,
        "heuristic_strategy": preferred,
        "heuristic_reason": heuristic["reason"],
        "all_candidates": candidates,
        "tuning_unreliable": tuning_unreliable,
        "tuning_unreliable_reason": unreliable_reason,
    }
