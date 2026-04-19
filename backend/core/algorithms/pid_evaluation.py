"""PID evaluation: closed-loop simulation and performance assessment.

Migrated from pid_new/backend/skills/rating.py
           and pid_new/backend/services/pid_evaluation_service.py

Contains a self-contained closed-loop simulator (no external deps beyond NumPy)
and a three-layer scoring scheme:
  Layer 1 — closed-loop performance (step response metrics)
  Layer 2 — method confidence (passed in from identification)
  Layer 3 — weighted final rating
"""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np


# ── Closed-loop simulator ─────────────────────────────────────────────────────

def _simulate(
    model_params: dict[str, Any],
    pid_params: dict[str, float],
    *,
    sp_initial: float = 50.0,
    sp_final: float = 60.0,
    n_steps: int = 500,
    dt: float = 1.0,
) -> dict[str, Any]:
    """Simulate a closed-loop step response.

    model_params: {model_type, K, T1 (or T), T2, L}
    pid_params:   {Kp, Ki, Kd}
    """
    eps = 1e-10
    mt = str(model_params.get("model_type", "FOPDT")).upper()
    K = float(model_params.get("K", 1.0))
    T1 = max(float(model_params.get("T1", model_params.get("T", 10.0))), eps)
    T2 = max(float(model_params.get("T2", 0.0)), 0.0)
    L = max(float(model_params.get("L", 0.0)), 0.0)
    Kp = float(pid_params.get("Kp", 1.0))
    Ki = float(pid_params.get("Ki", 0.0))
    Kd = float(pid_params.get("Kd", 0.0))

    pv_hist = np.zeros(n_steps)
    mv_hist = np.zeros(n_steps)
    sp_hist = np.zeros(n_steps)

    sp_change = sp_final - sp_initial
    mv0 = np.clip(50.0 - sp_change / (abs(K) + eps) * 0.3, 5.0, 95.0)

    delta_pv = delta_x2 = integral = prev_error = 0.0
    delay_steps = int(L / max(dt, eps))
    mv_buf: deque[float] = deque([0.0] * (delay_steps + 1))
    step_t = 10  # apply SP step at t=10

    for t in range(n_steps):
        sp = sp_initial if t < step_t else sp_final
        sp_hist[t] = sp
        pv = sp_initial + delta_pv
        pv_hist[t] = pv

        error = sp - pv
        integral += error * dt
        integral = float(np.clip(integral, -100.0 / (abs(Ki) + eps),
                                             100.0 / (abs(Ki) + eps)))
        deriv = (error - prev_error) / dt if t > 0 else 0.0

        mv = np.clip(mv0 + Kp * error + Ki * integral + Kd * deriv, 0.0, 100.0)
        mv_hist[t] = mv
        prev_error = error

        mv_buf.append(mv - mv0)
        delta_mv = mv_buf.popleft()

        if mt == "IPDT":
            delta_pv += K * delta_mv * dt
        elif T2 > eps:
            dv = delta_pv + (dt / T1) * (K * delta_mv - delta_pv)
            delta_x2 += (dt / max(T2, T1 * 0.1)) * (dv - delta_x2)
            delta_pv = delta_x2
        else:
            delta_pv += (dt / T1) * (K * delta_mv - delta_pv)

    # ── Performance metrics ───────────────────────────────────────────────────
    resp = pv_hist[step_t:]
    if len(resp) < 10 or abs(sp_change) < eps:
        return {
            "is_stable": False, "overshoot": 0.0,
            "settling_time": -1, "steady_state_error": 100.0,
            "oscillation_count": 0, "decay_ratio": 1.0, "rise_time": -1,
            "pv_history": pv_hist.tolist(),
            "mv_history": mv_hist.tolist(),
            "sp_history": sp_hist.tolist(),
        }

    tail = resp[-max(10, len(resp) // 10):]
    sse = abs(float(np.mean(tail)) - sp_final) / (abs(sp_change) + eps) * 100.0

    overshoot = (
        max(0.0, (float(np.max(resp)) - sp_final) / sp_change * 100.0)
        if sp_change > 0
        else max(0.0, (sp_final - float(np.min(resp))) / abs(sp_change) * 100.0)
    )

    tol = 0.02 * abs(sp_change)
    settling_time: float = float("inf")
    for i in range(len(resp) - 1, -1, -1):
        if abs(resp[i] - sp_final) > tol:
            if i < len(resp) - 1:
                settling_time = (i + 1) * dt
            break
    else:
        settling_time = 0.0

    err_sig = resp - sp_final
    zc = np.where(np.diff(np.signbit(err_sig)))[0]
    osc_count = int(len(zc) // 2)
    # Decay ratio: only count peaks that have actually overshot the SP
    # (same side as the step direction). Undershoot ringing in heavily
    # damped responses must NOT inflate decay_ratio above 1.
    sign = 1.0 if sp_change > 0 else -1.0
    signed = err_sig * sign  # positive = overshoot, negative = undershoot
    peaks = []
    for i in range(1, len(signed) - 1):
        if signed[i] > 0 and signed[i] > signed[i - 1] and signed[i] > signed[i + 1]:
            peaks.append(signed[i])
        if len(peaks) >= 2:
            break
    decay_ratio = peaks[1] / peaks[0] if len(peaks) >= 2 and peaks[0] > eps else 0.0

    # Rise time (10% → 90% of step)
    t10, t90 = sp_initial + 0.1 * sp_change, sp_initial + 0.9 * sp_change
    r_start = r_end = None
    for i, p in enumerate(resp):
        if sp_change > 0:
            if r_start is None and p >= t10: r_start = i
            if r_end is None and p >= t90: r_end = i; break
        else:
            if r_start is None and p <= t10: r_start = i
            if r_end is None and p <= t90: r_end = i; break
    rise_time = (r_end - r_start) * dt if r_start is not None and r_end is not None else float("inf")

    is_stable = (
        settling_time < 600.0
        and sse < 8.0
        and overshoot < (65.0 if decay_ratio <= 0.6 else 30.0)
        and decay_ratio < 0.8
    )

    return {
        "is_stable": is_stable,
        "overshoot": round(overshoot, 2),
        "settling_time": round(settling_time, 2) if settling_time < float("inf") else -1,
        "steady_state_error": round(sse, 2),
        "oscillation_count": osc_count,
        "decay_ratio": round(decay_ratio, 4),
        "rise_time": round(rise_time, 2) if rise_time < float("inf") else -1,
        "pv_history": pv_hist.tolist(),
        "mv_history": mv_hist.tolist(),
        "sp_history": sp_hist.tolist(),
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def _performance_score(sim: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Layer 1: closed-loop step-response score (0–10).

    Weights: overshoot 25%, settling_time 20%, steady_state_error 25%,
             oscillation_count 15%, decay_ratio 15%.
    Instability applies a 0.4× multiplier (not a veto).
    """
    overshoot = float(sim.get("overshoot", 0))
    settling = float(sim.get("settling_time", -1))
    sse = float(sim.get("steady_state_error", 0))
    osc = int(sim.get("oscillation_count", 0))
    dr = float(sim.get("decay_ratio", 0))
    stable = bool(sim.get("is_stable", True))

    # Overshoot score
    if overshoot <= 2: os_s = 10.0
    elif overshoot <= 5: os_s = 9.0
    elif overshoot <= 10: os_s = 8.0
    elif overshoot <= 15: os_s = 7.0
    elif overshoot <= 25: os_s = 6.0
    elif overshoot <= 40: os_s = 4.0
    elif overshoot <= 60: os_s = 2.5
    elif overshoot <= 100: os_s = 1.5
    else: os_s = max(0.0, 1.0 - (overshoot - 100) / 200)

    # Settling-time score
    if settling < 0:
        st_s = 0.0  # never settled
    elif settling <= 15: st_s = 10.0
    elif settling <= 30: st_s = 9.0
    elif settling <= 60: st_s = 7.5
    elif settling <= 120: st_s = 6.0
    elif settling <= 300: st_s = 4.0
    elif settling <= 600: st_s = 2.0
    else: st_s = 1.0

    # Steady-state error score
    if sse <= 0.5: sse_s = 10.0
    elif sse <= 1.0: sse_s = 9.0
    elif sse <= 2.0: sse_s = 7.5
    elif sse <= 5.0: sse_s = 5.5
    elif sse <= 10.0: sse_s = 3.5
    elif sse <= 20.0: sse_s = 2.0
    else: sse_s = max(0.0, 1.0 - (sse - 20) / 50)

    # Oscillation count score
    if osc == 0: oc_s = 7.0
    elif osc <= 2: oc_s = 10.0
    elif osc <= 4: oc_s = 7.0
    elif osc <= 6: oc_s = 5.0
    elif osc <= 10: oc_s = 3.0
    else: oc_s = max(0.0, 2.0 - (osc - 10) / 5)

    # Decay-ratio score
    if dr <= 0.1: dr_s = 10.0
    elif dr <= 0.25: dr_s = 9.0
    elif dr <= 0.5: dr_s = 6.0
    elif dr <= 0.8: dr_s = 3.0
    elif dr <= 1.0: dr_s = 1.0
    else: dr_s = 0.0

    raw = 0.25 * os_s + 0.20 * st_s + 0.25 * sse_s + 0.15 * oc_s + 0.15 * dr_s
    # Soft penalty for instability: deduct per offending criterion instead of
    # a flat 0.4× multiplier. Each per-criterion sub-score already reflects
    # its own badness; the multiplier was double-counting and produced the
    # "all metrics good but score 3.3" pathology.
    penalty = 0.0
    if not stable:
        if dr >= 0.8:
            penalty += 1.5 if dr < 1.2 else 2.5
        if overshoot >= 30:
            penalty += 1.0
        if sse >= 8.0:
            penalty += 1.5
        if settling < 0 or settling >= 600:
            penalty += 1.5
        penalty = min(penalty, 4.0)  # cap so a single divergent metric can't zero everything
    score = round(min(10.0, max(0.0, raw - penalty)), 2)

    return score, {
        "is_stable": stable,
        "overshoot": round(overshoot, 2),  "overshoot_score": round(os_s, 2),
        "settling_time": round(settling, 2), "settling_time_score": round(st_s, 2),
        "steady_state_error": round(sse, 2), "steady_state_error_score": round(sse_s, 2),
        "oscillation_count": osc,             "oscillation_count_score": round(oc_s, 2),
        "decay_ratio": round(dr, 4),          "decay_ratio_score": round(dr_s, 2),
        "raw_score": round(raw, 2),
    }


def _final_rating(perf_score: float, confidence: float) -> float:
    """Layer 3: weighted final score (0–10)."""
    raw = 0.7 * perf_score + 0.3 * confidence * 10.0
    if perf_score <= 1.0:
        raw = min(raw, 3.0)
    elif perf_score <= 3.0:
        raw = min(raw, 5.0)
    if confidence < 0.2:
        raw = min(raw, 6.0)
    return round(min(10.0, max(0.0, raw)), 2)


# ── Robustness ────────────────────────────────────────────────────────────────

def _perturb(model_params: dict[str, Any]) -> list[dict[str, Any]]:
    """Three perturbed variants: ±10% K/T, ±20% L."""
    mt = str(model_params.get("model_type", "FOPDT")).upper()
    combos = [
        {"k": 0.9, "t": 0.9, "l": 1.2},
        {"k": 1.1, "t": 1.1, "l": 0.9},
        {"k": 1.0, "t": 1.2, "l": 1.2},
    ]
    out = []
    for c in combos:
        if mt == "SOPDT":
            out.append({
                "model_type": "SOPDT",
                "K": float(model_params.get("K", 1.0)) * c["k"],
                "T1": max(float(model_params.get("T1", 10.0)) * c["t"], 1e-3),
                "T2": max(float(model_params.get("T2", 0.0)) * c["t"], 0.0),
                "L": max(float(model_params.get("L", 0.0)) * c["l"], 0.0),
            })
        elif mt == "IPDT":
            out.append({
                "model_type": "IPDT",
                "K": float(model_params.get("K", 1.0)) * c["k"],
                "L": max(float(model_params.get("L", 1.0)) * c["l"], 1e-3),
            })
        else:
            out.append({
                "model_type": mt,
                "K": float(model_params.get("K", 1.0)) * c["k"],
                "T1": max(float(model_params.get("T1", model_params.get("T", 10.0))) * c["t"], 1e-3),
                "T2": 0.0,
                "L": max(float(model_params.get("L", 0.0)) * c["l"], 0.0),
            })
    return out


# ── Main entry point ──────────────────────────────────────────────────────────

def evaluate_pid_params(
    *,
    Kp: float,
    Ki: float,
    Kd: float,
    model_type: str = "FOPDT",
    model_params: dict[str, Any] | None = None,
    K: float,
    T: float,
    L: float,
    dt: float,
    loop_type: str = "flow",
    confidence: float = 1.0,
    tuning_unreliable: bool = False,
    tuning_unreliable_reason: str = "",
) -> dict[str, Any]:
    """Simulate closed-loop response and evaluate PID performance.

    Runs forward step (50→60) + reverse step (60→50) + 3 robustness variants.

    Returns dict with:
        passed, performance_score, final_rating, is_stable,
        overshoot_percent, settling_time_s, recommendation,
        simulation (forward step trace), robustness_score.
    """
    mp = dict(model_params or {})
    # Normalise model_params to always contain T1/T2/L
    mt = (mp.get("model_type") or model_type or "FOPDT").strip().upper()
    mp.setdefault("model_type", mt)
    mp.setdefault("K", K)
    if mt == "SOPDT":
        mp.setdefault("T1", mp.get("T", T))
        mp.setdefault("T2", mp.get("T2", 0.0))
    elif mt == "IPDT":
        mp.setdefault("T1", max(T, 1e-3))
        mp.setdefault("T2", 0.0)
    else:
        mp.setdefault("T1", T)
        mp.setdefault("T2", 0.0)
    mp.setdefault("L", L)

    pid = {"Kp": Kp, "Ki": Ki, "Kd": Kd}
    # Adaptive simulation step: for fast plants (small T1), CSV sampling time
    # would alias and produce phantom oscillations. Use min(dt, T1/10).
    t1_for_sim = float(mp.get("T1", mp.get("T", 10.0)))
    sim_dt = max(0.05, min(float(dt), t1_for_sim / 10.0))
    n_steps = max(500, int(600.0 / sim_dt))

    # Forward step
    fwd = _simulate(mp, pid, sp_initial=50.0, sp_final=60.0, n_steps=n_steps, dt=sim_dt)
    perf_score, perf_details = _performance_score(fwd)

    # Reverse step (penalty for asymmetry)
    rev = _simulate(mp, pid, sp_initial=60.0, sp_final=50.0, n_steps=n_steps, dt=sim_dt)
    rev_score, _ = _performance_score(rev)

    # Robustness: perturbed models
    rob_scores: list[float] = []
    for variant in _perturb(mp):
        vsim = _simulate(variant, pid, sp_initial=50.0, sp_final=60.0, n_steps=n_steps, dt=sim_dt)
        s, _ = _performance_score(vsim)
        rob_scores.append(s)
    rob_score = round(
        min(rob_scores) * 0.6 + (sum(rob_scores) / len(rob_scores)) * 0.4, 2
    ) if rob_scores else 0.0

    # Reality check：用回路类型典型时间常数再仿一次，与辨识模型对照。
    # 若辨识模型 T 远低于典型值（说明可能塌缩），用典型 T 跑出来通常会发散，
    # 从而把"用塌缩模型自欺自评"的高分压回原形。
    _TYPICAL_T = {"flow": 5.0, "pressure": 30.0, "temperature": 300.0, "level": 600.0}
    typical_T = _TYPICAL_T.get((loop_type or "").lower().strip(), 0.0)
    reality_score = perf_score
    reality_diverged = False
    if typical_T > 0:
        reality_mp = dict(mp)
        reality_mp["T1"] = typical_T
        if mt == "SOPDT":
            reality_mp["T2"] = typical_T * 0.3
        else:
            reality_mp["T2"] = 0.0
        rsim = _simulate(reality_mp, pid, sp_initial=50.0, sp_final=60.0, n_steps=n_steps, dt=sim_dt)
        reality_score, _ = _performance_score(rsim)
        # 名义评分与典型评分差距过大 → 模型不可信
        if (perf_score - reality_score) > 3.0 or not rsim["is_stable"]:
            reality_diverged = True

    # MV constraint check
    mv_hist = fwd.get("mv_history", [])
    sat_pct = (sum(1 for v in mv_hist if v <= 0.5 or v >= 99.5) / max(len(mv_hist), 1)) * 100.0
    mv_tv = sum(abs(mv_hist[i] - mv_hist[i - 1]) for i in range(1, len(mv_hist)))
    constraint_score = round(
        min(10.0, max(0.0, 10.0
            - min(5.0, sat_pct / 8.0)
            - min(3.0, mv_tv / 250.0)
            - (2.5 if not fwd["is_stable"] else 0.0))),
        2,
    )

    # Combined online-readiness score
    readiness = round(min(10.0, max(0.0,
        0.45 * perf_score
        + 0.20 * rev_score
        + 0.20 * rob_score
        + 0.15 * constraint_score
    )), 2)

    final = _final_rating(perf_score, confidence)
    passed = bool(readiness >= 7.0 and fwd["is_stable"] and rev["is_stable"] and sat_pct < 35.0)

    # 硬上限 1：低置信度永远不能"可以上线"
    cap_reasons: list[str] = []
    if confidence < 0.5:
        if perf_score > 5.0:
            perf_score = 5.0
        if final > 5.0:
            final = 5.0
        if readiness > 5.0:
            readiness = 5.0
        passed = False
        cap_reasons.append(f"模型置信度 {confidence:.2f} < 0.5，评分封顶 5 分")

    # 硬上限 2.5：reality check 发散 —— 用回路典型 T 跑出来不稳/差距巨大
    if reality_diverged:
        if perf_score > 4.0:
            perf_score = 4.0
        if final > 4.0:
            final = 4.0
        if readiness > 4.0:
            readiness = 4.0
        passed = False
        cap_reasons.append(
            f"Reality check 发散：用 {loop_type} 典型时间常数仿真评分 {reality_score:.1f}，"
            f"与名义模型评分差距过大，提示辨识模型可能塌缩"
        )

    # 硬上限 2：整定阶段已判 unreliable（PID 物理量级不合理）必须不通过
    if tuning_unreliable:
        if perf_score > 3.0:
            perf_score = 3.0
        if final > 3.0:
            final = 3.0
        if readiness > 3.0:
            readiness = 3.0
        passed = False
        cap_reasons.append(f"整定参数物理量级不合理：{tuning_unreliable_reason}")

    if cap_reasons:
        recommendation = "暂不建议上线：" + "；".join(cap_reasons)
    elif passed and readiness >= 8.5:
        recommendation = "建议进入受控条件下的小扰动试投。"
    elif passed:
        recommendation = "建议保守试投，并保留人工确认。"
    else:
        recommendation = "暂不建议直接上线，建议继续回流优化。"

    return {
        "passed": passed,
        "performance_score": perf_score,
        "final_rating": final,
        "readiness_score": readiness,
        "robustness_score": rob_score,
        "constraint_score": constraint_score,
        "is_stable": fwd["is_stable"],
        "overshoot_percent": fwd["overshoot"],
        "settling_time_s": fwd["settling_time"],
        "steady_state_error": fwd["steady_state_error"],
        "oscillation_count": fwd["oscillation_count"],
        "decay_ratio": fwd["decay_ratio"],
        "rise_time_s": fwd["rise_time"],
        "mv_saturation_pct": round(sat_pct, 2),
        "performance_details": perf_details,
        "reality_check_score": round(reality_score, 2),
        "reality_check_typical_T": typical_T,
        "reality_check_diverged": reality_diverged,
        "score_caps_applied": cap_reasons,
        "recommendation": recommendation,
        "simulation": {
            "pv_history": fwd["pv_history"],
            "mv_history": fwd["mv_history"],
            "sp_history": fwd["sp_history"],
            "dt": sim_dt,
        },
    }
