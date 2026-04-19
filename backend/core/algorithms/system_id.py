"""System identification: process model fitting.

Key improvements over pid_new:
- #5: Dead-time estimation via normalised positive-lag cross-correlation only.
- #6: Deviation-variable normalisation (subtract initial value, scale by span)
      instead of mean-centering + std-normalisation, which distorts K.
- #7: L upper bound = window_duration / 4 (not 1/2).
- #8: Multi-start optimisation: grid of L initial values + L-BFGS-B refinement.
- #9: IPDT K init estimated from data rate, not hardcoded 0.05.
- #10: SOPDT constrained to T1 >= T2 (add T_sum, ratio parameterisation).
- #11: AIC penalty for model complexity when comparing across types.
- #12: Short-window confidence: R² weight decreases for N < 200 (fix inverted logic).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize

from core.algorithms.signal_processing import align_series, detrend_if_needed
from models.process_model import IdentificationResult, ModelConfidence, ModelType, ProcessModel

# ── Model order per loop type ────────────────────────────────────────────────

_MODEL_ORDER: dict[str, list[str]] = {
    "flow":        ["FO", "FOPDT", "SOPDT", "IPDT"],
    "pressure":    ["FO", "FOPDT", "SOPDT", "IPDT"],
    "temperature": ["SOPDT", "FOPDT", "FO", "IPDT"],
    "level":       ["IPDT", "FOPDT", "FO", "SOPDT"],
}

# Number of free parameters per model (for AIC)
_N_PARAMS: dict[str, int] = {
    "FO": 2, "FOPDT": 3, "SOPDT": 4, "IPDT": 2,
}


# ── Dead-time estimation ─────────────────────────────────────────────────────

def _estimate_dead_time(mv: np.ndarray, pv: np.ndarray, dt: float) -> float:
    """Fix #5: positive-lag normalised cross-correlation."""
    mv_c = mv - float(np.mean(mv))
    pv_c = pv - float(np.mean(pv))
    mv_std = float(np.std(mv_c))
    pv_std = float(np.std(pv_c))
    if mv_std < 1e-12 or pv_std < 1e-12:
        return 0.0

    max_lag = max(3, min(int(round(60.0 / max(dt, 1e-6))), mv.size // 4))
    best_lag, best_score = 0, 0.0
    for lag in range(0, max_lag + 1):
        a = mv_c[:-lag] if lag > 0 else mv_c
        b = pv_c[lag:] if lag > 0 else pv_c
        if a.size < 10:
            break
        s = float(np.mean(a * b) / (mv_std * pv_std))
        if s > best_score + 1e-9:
            best_score, best_lag = s, lag

    # Reject if peak correlation is too weak to trust
    return best_lag * dt if best_score > 0.1 else 0.0


# ── Normalisation helpers ────────────────────────────────────────────────────

def _normalise(mv: np.ndarray, pv: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Fix #6: deviation-variable normalisation.

    Subtract initial value, divide by span. This preserves the physical
    gain ratio K = ΔPV/ΔMV and avoids distortion from std-based scaling.
    """
    mv0, pv0 = float(mv[0]), float(pv[0])
    mv_d = mv - mv0
    pv_d = pv - pv0
    mv_span = float(np.max(np.abs(mv_d))) or 1.0
    pv_span = float(np.max(np.abs(pv_d))) or 1.0
    return mv_d / mv_span, pv_d / pv_span, mv_span, pv_span, mv0, pv0


# ── Simulation functions ─────────────────────────────────────────────────────

def _sim_fo(mv: np.ndarray, K: float, T: float, dt: float) -> np.ndarray:
    T = max(T, 1e-6)
    alpha = dt / (T + dt)
    y = np.zeros_like(mv, dtype=float)
    for i in range(len(mv)):
        y[i] = (1.0 - alpha) * (y[i - 1] if i > 0 else 0.0) + K * alpha * mv[i]
    return y


def _sim_fopdt(mv: np.ndarray, K: float, T: float, L: float, dt: float) -> np.ndarray:
    T, L = max(T, 1e-6), max(L, 0.0)
    alpha = dt / (T + dt)
    d = int(round(L / dt))
    y = np.zeros_like(mv, dtype=float)
    for i in range(len(mv)):
        u = mv[i - d] if i >= d else 0.0
        y[i] = (1.0 - alpha) * (y[i - 1] if i > 0 else 0.0) + K * alpha * u
    return y


def _sim_sopdt(mv: np.ndarray, K: float, T1: float, T2: float, L: float, dt: float) -> np.ndarray:
    T1, T2, L = max(T1, 1e-6), max(T2, 1e-6), max(L, 0.0)
    a1, a2 = dt / (T1 + dt), dt / (T2 + dt)
    d = int(round(L / dt))
    y1 = np.zeros_like(mv, dtype=float)
    y2 = np.zeros_like(mv, dtype=float)
    for i in range(len(mv)):
        u = mv[i - d] if i >= d else 0.0
        y1[i] = (1.0 - a1) * (y1[i - 1] if i > 0 else 0.0) + a1 * u
        y2[i] = (1.0 - a2) * (y2[i - 1] if i > 0 else 0.0) + a2 * y1[i]
    return K * y2


def _sim_ipdt(mv: np.ndarray, K: float, L: float, dt: float) -> np.ndarray:
    L = max(L, 0.0)
    d = int(round(L / dt))
    y = np.zeros_like(mv, dtype=float)
    for i in range(len(mv)):
        u = mv[i - d] if i >= d else 0.0
        y[i] = (y[i - 1] if i > 0 else 0.0) + K * dt * u
    return y


# ── Fit quality ──────────────────────────────────────────────────────────────

def _fit_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    nrmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"r2_score": float(r2), "normalized_rmse": nrmse}


def _aic(n_params: int, n_points: int, nrmse: float) -> float:
    """Fix #11: AIC penalty — discourages unnecessary model complexity."""
    sse = nrmse ** 2 * n_points
    if sse <= 0:
        sse = 1e-12
    return n_points * float(np.log(sse / n_points)) + 2.0 * n_params


# ── Per-model fitters ────────────────────────────────────────────────────────

def _fit_fo(mv_n: np.ndarray, pv_n: np.ndarray, dt: float, window_dt: float) -> dict[str, Any]:
    T_max = max(mv_n.size * dt, 10.0)
    K_sign = 1.0 if float(np.corrcoef(mv_n, pv_n)[0, 1]) >= 0 else -1.0
    K_init = K_sign * max(float(np.max(np.abs(pv_n))) / max(float(np.max(np.abs(mv_n))), 1e-6), 0.1)

    def obj(p):
        K, T = p
        return float(np.mean((pv_n - _sim_fo(mv_n, K, max(T, dt), dt)) ** 2))

    res = optimize.minimize(
        obj, [K_init, max(dt * 5, window_dt / 6)],
        bounds=[(-20.0, 20.0), (dt, T_max)], method="L-BFGS-B",
    )
    K, T = float(res.x[0]), float(res.x[1])
    metrics = _fit_metrics(_sim_fo(mv_n, K, T, dt), pv_n)
    return {"K": K, "T": T, "L": 0.0, **metrics}


def _fit_fopdt(mv_n: np.ndarray, pv_n: np.ndarray, dt: float, window_dt: float, L_hint: float) -> dict[str, Any]:
    """Fix #7 + #8: tighter L bound (window/4) + multi-start on L."""
    T_max = max(mv_n.size * dt, 10.0)
    L_max = min(window_dt / 4.0, T_max / 2.0)   # fix #7
    K_sign = 1.0 if float(np.corrcoef(mv_n, pv_n)[0, 1]) >= 0 else -1.0
    K_init = K_sign * max(float(np.max(np.abs(pv_n))) / max(float(np.max(np.abs(mv_n))), 1e-6), 0.1)

    def obj(p):
        K, T, L = p
        return float(np.mean((pv_n - _sim_fopdt(mv_n, K, max(T, dt), max(L, 0), dt)) ** 2))

    # Fix #8: multi-start over L initial values
    L_candidates = sorted(set([
        0.0,
        min(L_hint, L_max),
        L_max * 0.1,
        L_max * 0.25,
        L_max * 0.5,
    ]))
    best_res, best_val = None, float("inf")
    T_init = max(dt * 10.0, window_dt / 5.0)
    for L0 in L_candidates:
        r = optimize.minimize(
            obj, [K_init, T_init, L0],
            bounds=[(-20.0, 20.0), (dt, T_max), (0.0, L_max)],
            method="L-BFGS-B",
        )
        if r.fun < best_val:
            best_val, best_res = r.fun, r
    K, T, L = [float(v) for v in best_res.x]  # type: ignore[union-attr]
    metrics = _fit_metrics(_sim_fopdt(mv_n, K, T, L, dt), pv_n)
    return {"K": K, "T": max(T, dt), "L": max(L, 0.0), **metrics}


def _fit_sopdt(mv_n: np.ndarray, pv_n: np.ndarray, dt: float, window_dt: float, L_hint: float) -> dict[str, Any]:
    """Fix #10: parameterise as (T_sum, ratio) so T1 >= T2 always."""
    T_max = max(mv_n.size * dt, 10.0)
    L_max = min(window_dt / 4.0, T_max / 2.0)
    K_sign = 1.0 if float(np.corrcoef(mv_n, pv_n)[0, 1]) >= 0 else -1.0
    K_init = K_sign * max(float(np.max(np.abs(pv_n))) / max(float(np.max(np.abs(mv_n))), 1e-6), 0.1)
    T_sum_init = max(dt * 10.0, window_dt / 4.0)

    def obj(p):
        K, T_sum, ratio, L = p
        T1 = T_sum * max(ratio, 0.5)
        T2 = T_sum * (1.0 - max(ratio, 0.5))
        T2 = max(T2, dt)
        return float(np.mean((pv_n - _sim_sopdt(mv_n, K, T1, T2, max(L, 0), dt)) ** 2))

    L0 = min(L_hint, L_max)
    res = optimize.minimize(
        obj, [K_init, T_sum_init, 0.7, L0],
        bounds=[(-20.0, 20.0), (dt * 2, T_max * 2), (0.5, 0.99), (0.0, L_max)],
        method="L-BFGS-B",
    )
    K, T_sum, ratio, L = [float(v) for v in res.x]
    T1 = T_sum * max(ratio, 0.5)
    T2 = max(T_sum * (1.0 - max(ratio, 0.5)), dt)
    metrics = _fit_metrics(_sim_sopdt(mv_n, K, T1, T2, L, dt), pv_n)
    return {"K": K, "T1": T1, "T2": T2, "T": T1 + T2, "L": max(L, 0.0), **metrics}


def _fit_ipdt(mv_n: np.ndarray, pv_n: np.ndarray, dt: float, window_dt: float) -> dict[str, Any]:
    """Fix #9: estimate K_init from observed PV rate."""
    L_max = min(window_dt / 4.0, mv_n.size * dt / 2.0)
    # Estimate integrating gain from data: K ≈ ΔPV / (ΔMV × Δt)
    mv_mean_abs = float(np.mean(np.abs(mv_n))) or 1.0
    pv_rate = float(np.max(np.abs(np.diff(pv_n)))) / max(dt, 1e-6)
    K_init = pv_rate / max(mv_mean_abs / dt, 1e-9)
    K_sign = 1.0 if float(np.sum(np.diff(pv_n))) >= 0 else -1.0
    K_init = K_sign * max(abs(K_init), 1e-4)
    K_range = max(abs(K_init) * 10, 1.0)

    def obj(p):
        K, L = p
        return float(np.mean((pv_n - _sim_ipdt(mv_n, K, max(L, 0), dt)) ** 2))

    L0 = min(_estimate_dead_time(mv_n, pv_n, dt), L_max)
    res = optimize.minimize(
        obj, [K_init, L0],
        bounds=[(-K_range, K_range), (0.0, L_max)],
        method="L-BFGS-B",
    )
    K, L = float(res.x[0]), float(res.x[1])
    metrics = _fit_metrics(_sim_ipdt(mv_n, K, L, dt), pv_n)
    return {"K": K, "T": 0.0, "L": max(L, 0.0), **metrics}


# ── Confidence ───────────────────────────────────────────────────────────────

def _confidence(
    nrmse: float,
    r2: float,
    n_points: int,
    drift_ratio: float = 0.0,
    sat_ratio: float = 0.0,
) -> ModelConfidence:
    """Fix #12: short-window → lower R² weight (not higher)."""
    n_points = max(1, n_points)
    # For N < 200 points R² is unreliable — reduce its weight
    short_factor = max(0.0, min(1.0, (200.0 - float(n_points)) / 150.0))
    r2_weight = max(0.3, 0.65 - 0.2 * short_factor)   # shrinks for small N
    rmse_weight = 1.0 - r2_weight

    rmse_ref = max(0.35, 1e-6)
    if n_points < 200:
        rmse_ref = max(rmse_ref, 0.45)
    rmse_score = max(0.0, 1.0 - (nrmse / rmse_ref) ** 1.2)

    conf = r2_weight * max(0.0, min(1.0, r2)) + rmse_weight * rmse_score

    if drift_ratio > 0.3:
        conf *= 1.0 - min((drift_ratio - 0.3) / 0.7, 1.0) * 0.18
    if sat_ratio > 0.25:
        conf *= 1.0 - min((sat_ratio - 0.25) / 0.5, 1.0) * 0.15
    if r2 < 0.4:
        conf = min(conf, 0.45)
    if r2 < 0.25:
        conf = min(conf, 0.30)

    conf = float(max(0.0, min(1.0, conf)))
    if conf >= 0.80:
        quality, rec = "excellent", "模型可信，可直接用于 PID 整定"
    elif conf >= 0.65:
        quality, rec = "good", "模型基本可信，建议结合现场经验校核"
    elif conf >= 0.45:
        quality, rec = "fair", "模型置信度偏低，建议复查数据窗口"
    else:
        quality, rec = "poor", "模型不可依，建议重新采集数据"

    return ModelConfidence(confidence=conf, quality=quality, recommendation=rec,
                           r2_score=float(r2), rmse_score=float(rmse_score))


# ── Main identification entry point ─────────────────────────────────────────

def fit_best_model(
    *,
    cleaned_df: Any,
    candidate_windows: list[dict[str, Any]],
    actual_dt: float,
    loop_type: str = "flow",
    quality_metrics: dict[str, Any] | None = None,
    force_model_types: list[str] | None = None,
) -> dict[str, Any]:
    """Try all candidate windows × model types, pick best by AIC-penalised score.

    Args:
        force_model_types: If provided, only try these model types (e.g. ["FOPDT"]).

    Returns dict with:
        model (ProcessModel), confidence (ModelConfidence),
        window_source, fit_preview, attempts, candidates, selection_reason.
    """
    default_order = _MODEL_ORDER.get(loop_type.lower().strip(), ["FOPDT", "FO", "SOPDT", "IPDT"])
    if force_model_types:
        model_order = [m.upper() for m in force_model_types if m.upper() in default_order + ["FO", "FOPDT", "SOPDT", "IPDT"]]
        if not model_order:
            model_order = default_order
    else:
        model_order = default_order
    dt = max(actual_dt, 1e-6)

    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    usable = [w for w in candidate_windows if w.get("window_usable_for_id")]
    to_fit = usable if usable else candidate_windows

    for window in to_fit:
        w_start = int(window.get("window_start_idx", 0))
        w_end = int(window.get("window_end_idx", len(cleaned_df)))
        seg = cleaned_df.iloc[w_start:w_end].reset_index(drop=True)
        if len(seg) < 15:
            continue

        mv_raw = seg["MV"].to_numpy(dtype=float)
        pv_raw = seg["PV"].to_numpy(dtype=float)
        window_dt = len(seg) * dt

        L_hint = _estimate_dead_time(mv_raw, pv_raw, dt)

        # Preprocessing: detrend + align if baseline fit is poor
        pv_work, mv_work = pv_raw, mv_raw
        pv_detrended = False
        lag_steps = 0

        # Quick baseline R² check
        mv_n0, pv_n0, mvs, pvs, mv0, pv0 = _normalise(mv_work, pv_work)
        try:
            base = _fit_fopdt(mv_n0, pv_n0, dt, window_dt, L_hint)
            if base["r2_score"] < 0.3 or base["normalized_rmse"] > 0.5:
                pv_d, detrended = detrend_if_needed(pv_raw)
                mv_adj, pv_adj, lag_steps, _ = align_series(mv_raw, pv_d, dt)
                if len(mv_adj) >= 15:
                    mv_work, pv_work = mv_adj, pv_adj
                    pv_detrended = detrended
        except Exception:
            pass

        mv_n, pv_n, mvs, pvs, mv0, pv0 = _normalise(mv_work, pv_work)
        n_pts = len(seg)
        drift = float(window.get("window_drift_ratio", 0.0))
        sat = float(window.get("window_corr", 0.0))  # note: corr used as proxy

        for model_type in model_order:
            attempt: dict[str, Any] = {
                "window_source": window.get("window_source", ""),
                "model_type": model_type,
                "points": n_pts,
                "pv_detrended": pv_detrended,
                "lag_steps": lag_steps,
                "success": False,
            }
            try:
                if model_type == "FO":
                    raw_p = _fit_fo(mv_n, pv_n, dt, window_dt)
                elif model_type == "FOPDT":
                    raw_p = _fit_fopdt(mv_n, pv_n, dt, window_dt, L_hint)
                elif model_type == "SOPDT":
                    raw_p = _fit_sopdt(mv_n, pv_n, dt, window_dt, L_hint)
                else:
                    raw_p = _fit_ipdt(mv_n, pv_n, dt, window_dt)
            except Exception as exc:
                attempt["error"] = str(exc)
                attempts.append(attempt)
                continue

            # Rescale K back to physical units
            K_phys = raw_p["K"] * pvs / mvs
            r2 = float(raw_p["r2_score"])
            nrmse = float(raw_p["normalized_rmse"])

            # Fix #11: AIC-based fit score
            aic = _aic(_N_PARAMS[model_type], n_pts, nrmse)
            # Combined score: higher R², lower NRMSE, lower AIC is better
            fit_score = 10.0 * (0.6 * max(r2, 0.0) + 0.4 * max(1.0 - nrmse, 0.0)) - 0.005 * aic

            conf = _confidence(nrmse, r2, n_pts, drift_ratio=drift)

            attempt.update({
                "K": K_phys,
                "T": float(raw_p.get("T", 0.0)),
                "T1": float(raw_p.get("T1", 0.0)),
                "T2": float(raw_p.get("T2", 0.0)),
                "L": float(raw_p.get("L", 0.0)),
                "r2_score": r2,
                "normalized_rmse": nrmse,
                "fit_score": fit_score,
                "aic": aic,
                "confidence": conf.confidence,
                "success": True,
            })
            attempts.append(attempt)

            if best is None or fit_score > float(best.get("fit_score", -1e9)):
                best = {
                    **attempt,
                    "K_phys": K_phys,
                    "raw_params": raw_p,
                    "conf": conf,
                    "window": window,
                    "seg": seg,
                    "mvs": mvs,
                    "pvs": pvs,
                }

    if best is None:
        raise ValueError("所有候选窗口均无法完成系统辨识，请检查数据质量")

    mt = best["model_type"]
    rp = best["raw_params"]
    K_phys = best["K_phys"]
    pvs = best["pvs"]
    mvs = best["mvs"]

    model = ProcessModel(
        model_type=ModelType(mt),
        K=float(K_phys),
        T=float(rp.get("T", 0.0)),
        T1=float(rp.get("T1", 0.0)),
        T2=float(rp.get("T2", 0.0)),
        L=float(rp.get("L", 0.0)),
        r2_score=float(rp["r2_score"]),
        normalized_rmse=float(rp["normalized_rmse"]),
        raw_rmse=float(rp["normalized_rmse"]) * float(pvs),
        success=True,
    )

    # Top-5 candidates for frontend display
    successful = [a for a in attempts if a.get("success")]
    candidates = sorted(successful, key=lambda a: float(a.get("fit_score", -999)), reverse=True)[:5]
    for i, c in enumerate(candidates):
        c["is_selected"] = (
            c["model_type"] == mt and
            c.get("window_source") == best.get("window_source")
        )

    selection_reason = (
        f"已对 {len(to_fit)} 个候选窗口尝试 {', '.join(model_order)} 模型，"
        f"采用 R²/NRMSE 结合 AIC 罚项评分，最终选择 {mt} 模型"
        f"（R²={model.r2_score:.3f}, NRMSE={model.normalized_rmse:.3f}）。"
    )

    return {
        "model": model,
        "confidence": best["conf"],
        "window_source": str(best.get("window_source", "")),
        "fit_preview": _build_fit_preview(best["seg"], model, actual_dt),
        "attempts": attempts,
        "candidates": candidates,
        "selection_reason": selection_reason,
    }


def _build_fit_preview(seg: Any, model: ProcessModel, dt: float, max_pts: int = 200) -> dict[str, Any]:
    pv = seg["PV"].to_numpy(dtype=float)
    mv = seg["MV"].to_numpy(dtype=float)
    mv_d = mv - mv[0]
    n = len(seg)

    mt = model.model_type.value
    if mt == "FO":
        sim = _sim_fo(mv_d, model.K, model.T, dt)
    elif mt == "FOPDT":
        sim = _sim_fopdt(mv_d, model.K, model.T, model.L, dt)
    elif mt == "SOPDT":
        sim = _sim_sopdt(mv_d, model.K, model.T1, model.T2, model.L, dt)
    else:
        sim = _sim_ipdt(mv_d, model.K, model.L, dt)
    pv_fit = pv[0] + sim

    step = max(1, n // max_pts)
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    ts = None
    if "timestamp" in seg.columns:
        ts = seg["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

    points = []
    for i in indices:
        pt: dict[str, Any] = {
            "index": i, "pv": float(pv[i]),
            "pv_fit": float(pv_fit[i]), "mv": float(mv[i]),
        }
        if ts:
            pt["time"] = ts[i]
        points.append(pt)

    return {
        "points": points,
        "model_type": mt,
        "x_axis": "timestamp" if ts else "index",
    }
