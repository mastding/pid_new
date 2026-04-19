"""数据画像内部分析器（非 LLM 可见技能）。

每个函数只做一件事，输入清洗后的 DataFrame，返回一个 dict。
全部启发式为通用默认值，后续遇业务特殊阈值再调。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ── PV 量程 / 物理合理性推测 ─────────────────────────────────────────────

_PHYSICAL_HINTS: list[tuple[str, tuple[float, float]]] = [
    ("valve_opening", (0.0, 100.0)),     # 阀位 %
    ("level_pct", (0.0, 100.0)),          # 液位 %
    ("pressure_kpa", (0.0, 10000.0)),     # 压力
    ("flow_rate", (0.0, 10000.0)),        # 流量
    ("temperature_c", (-50.0, 1500.0)),   # 温度
]


def analyze_pv_range(df: pd.DataFrame) -> dict[str, Any]:
    """PV 基本统计 + 工程物理量猜测。"""
    pv = df["PV"].to_numpy(dtype=float)
    pv_min, pv_max = float(np.min(pv)), float(np.max(pv))
    pv_range = pv_max - pv_min
    # 不能简单按 min/max 落在哪个区间猜，信号常从 40 走到 60（量程内局部），
    # 只给一个非常粗略的"兼容物理量"列表，让 LLM 自行判断
    compatible: list[str] = []
    for label, (lo, hi) in _PHYSICAL_HINTS:
        if pv_min >= lo - 1e-6 and pv_max <= hi + 1e-6:
            compatible.append(label)

    return {
        "min": round(pv_min, 4),
        "max": round(pv_max, 4),
        "mean": round(float(np.mean(pv)), 4),
        "std": round(float(np.std(pv, ddof=0)), 4),
        "range": round(pv_range, 4),
        "iqr": round(float(np.percentile(pv, 75) - np.percentile(pv, 25)), 4),
        "compatible_physical_units": compatible,
    }


# ── MV 饱和检测 ─────────────────────────────────────────────────────────

def analyze_mv_saturation(df: pd.DataFrame) -> dict[str, Any]:
    """估计 MV 触顶/触底的比例与最长连续饱和段。

    判据：距离 MV 观测到的最小值/最大值 < 范围的 1% 视为饱和。
    """
    mv = df["MV"].to_numpy(dtype=float)
    mv_min, mv_max = float(np.min(mv)), float(np.max(mv))
    span = mv_max - mv_min
    n = len(mv)

    if span < 1e-9 or n < 10:
        return {
            "min": round(mv_min, 4),
            "max": round(mv_max, 4),
            "saturation_high_pct": 0.0,
            "saturation_low_pct": 0.0,
            "longest_saturated_run_sec": 0.0,
        }

    threshold = span * 0.01
    near_top = mv >= (mv_max - threshold)
    near_bot = mv <= (mv_min + threshold)

    def longest_run(mask: np.ndarray) -> int:
        best = cur = 0
        for x in mask:
            cur = cur + 1 if x else 0
            if cur > best:
                best = cur
        return best

    return {
        "min": round(mv_min, 4),
        "max": round(mv_max, 4),
        "saturation_high_pct": round(100.0 * float(near_top.mean()), 2),
        "saturation_low_pct": round(100.0 * float(near_bot.mean()), 2),
        "longest_saturated_run_points": int(max(longest_run(near_top), longest_run(near_bot))),
    }


# ── 控制死区估算 ───────────────────────────────────────────────────────

def analyze_deadzone(df: pd.DataFrame, pv_noise_std: float) -> dict[str, Any]:
    """估算执行器/传感器死区。

    判据：|ΔMV| 较大但 |ΔPV| 落在噪声带内 → 计一次"死区事件"。
    占比高说明死区显著。
    """
    mv = df["MV"].to_numpy(dtype=float)
    pv = df["PV"].to_numpy(dtype=float)
    if len(mv) < 20:
        return {"evidence_ratio": 0.0, "evidence_count": 0, "estimated_width": 0.0}

    dmv = np.abs(np.diff(mv))
    dpv = np.abs(np.diff(pv))
    mv_span = float(np.max(mv) - np.min(mv))
    if mv_span < 1e-9:
        return {"evidence_ratio": 0.0, "evidence_count": 0, "estimated_width": 0.0}

    # 有"明显"MV 动作，但 PV 几乎不响应（仅噪声级别）
    mv_thr = mv_span * 0.005     # MV 动作 > 量程 0.5%
    pv_thr = max(2.0 * pv_noise_std, 1e-9)
    evidence = (dmv > mv_thr) & (dpv < pv_thr)
    evidence_count = int(evidence.sum())
    total_movement = int((dmv > mv_thr).sum())
    ratio = evidence_count / total_movement if total_movement > 0 else 0.0

    # 触发了死区判据的 MV 动作幅度估计（取 70 分位）
    width_est = float(np.percentile(dmv[evidence], 70)) if evidence_count > 0 else 0.0

    return {
        "evidence_count": evidence_count,
        "evidence_ratio": round(ratio, 4),
        "estimated_width": round(width_est, 4),
    }


# ── 噪声水平 ────────────────────────────────────────────────────────────

def analyze_noise(df: pd.DataFrame) -> dict[str, Any]:
    """估算 PV 噪声：一阶差分的 MAD 是对独立白噪声 σ 的稳健估计。

    返回的 pv_noise_std 供死区/阈值计算复用。
    """
    pv = df["PV"].to_numpy(dtype=float)
    if len(pv) < 20:
        return {"pv_noise_std": 0.0, "snr_db": 0.0, "noise_level": "unknown"}

    diffs = np.diff(pv)
    # MAD 方法：1.4826 * median(|x|) / sqrt(2) ≈ σ (一阶差分后方差翻倍)
    mad = 1.4826 * float(np.median(np.abs(diffs - np.median(diffs))))
    noise_std = mad / np.sqrt(2.0)

    signal_span = float(np.max(pv) - np.min(pv))
    snr = 20.0 * np.log10(signal_span / noise_std) if noise_std > 1e-9 else 60.0

    # 噪声水平分级（相对量程）
    # 注意：此处 PV 已经过 denoise_pv 去噪，所以阈值相比原始信号收紧。
    rel = noise_std / signal_span if signal_span > 1e-9 else 0.0
    if rel < 0.002:
        level = "low"
    elif rel < 0.005:
        level = "medium"
    else:
        level = "high"

    return {
        "pv_noise_std": round(noise_std, 6),
        "snr_db": round(float(snr), 2),
        "relative_noise_ratio": round(rel, 5),
        "noise_level": level,
    }


# ── 振荡检测 ────────────────────────────────────────────────────────────

def analyze_oscillation(df: pd.DataFrame, dt: float) -> dict[str, Any]:
    """FFT 寻找 PV 主频峰值。

    判据：峰值能量占总能量 > 15% 且频率在合理范围内 → 认为有振荡。
    """
    pv = df["PV"].to_numpy(dtype=float)
    n = len(pv)
    if n < 64 or dt <= 0:
        return {"detected": False}

    # 去均值 + 加窗（避免频谱泄漏）
    x = pv - np.mean(pv)
    window = np.hanning(n)
    x = x * window

    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=dt)

    # 跳过 DC 与极低频（> 5 倍窗口长度才可信）
    min_freq = 1.0 / (n * dt / 5.0)
    valid = freqs > min_freq
    if not valid.any():
        return {"detected": False}

    spec_valid = spectrum[valid]
    freqs_valid = freqs[valid]
    peak_idx = int(np.argmax(spec_valid))
    peak_power = float(spec_valid[peak_idx])
    total_power = float(spec_valid.sum())
    ratio = peak_power / total_power if total_power > 1e-18 else 0.0

    detected = ratio > 0.15
    peak_freq = float(freqs_valid[peak_idx])
    period = 1.0 / peak_freq if peak_freq > 1e-9 else 0.0

    return {
        "detected": bool(detected),
        "dominant_frequency_hz": round(peak_freq, 5),
        "period_sec": round(period, 2),
        "peak_power_ratio": round(ratio, 4),
    }


# ── 干扰类型识别 ────────────────────────────────────────────────────────

def analyze_disturbance(df: pd.DataFrame) -> dict[str, Any]:
    """粗略识别 SV/MV 变化类型。

    返回各类事件计数：
      sv_steps  — SV 阶跃次数（>量程 5%）
      mv_steps  — MV 阶跃次数
      pv_drift  — PV 单调漂移（整体斜率显著）
    """
    pv = df["PV"].to_numpy(dtype=float)
    n = len(pv)

    def count_steps(arr: np.ndarray, rel: float = 0.05) -> int:
        if arr.size < 10:
            return 0
        span = float(np.max(arr) - np.min(arr))
        if span < 1e-9:
            return 0
        thr = span * rel
        return int(np.sum(np.abs(np.diff(arr)) > thr))

    mv_steps = count_steps(df["MV"].to_numpy(dtype=float))
    sv_steps = count_steps(df["SV"].to_numpy(dtype=float)) if "SV" in df.columns else 0

    # PV 漂移：回归斜率 × 总时长 与量程比
    drift_ratio = 0.0
    if n >= 20:
        span = float(np.max(pv) - np.min(pv))
        if span > 1e-9:
            t = np.arange(n, dtype=float)
            slope = float(np.polyfit(t, pv, 1)[0])
            drift_ratio = abs(slope) * n / span

    return {
        "sv_step_count": sv_steps,
        "mv_step_count": mv_steps,
        "pv_drift_ratio": round(drift_ratio, 4),
        "pv_monotonic_drift": bool(drift_ratio > 0.5),
    }


# ── 回路类型推断 ────────────────────────────────────────────────────────

def classify_loop_type(df: pd.DataFrame, dt: float) -> dict[str, Any]:
    """根据响应时间常数数量级粗略推断回路类型。

    规则（通用启发式，可被用户/LLM 覆盖）：
      时间常数 < 10s  → flow
      10s - 60s       → pressure
      60s - 300s      → level
      > 300s          → temperature
    时间常数用"PV 跨越 63% 量程"的经验估计。
    """
    pv = df["PV"].to_numpy(dtype=float)
    n = len(pv)
    if n < 50:
        return {"inferred_type": "unknown", "time_constant_sec": 0.0, "confidence": 0.0, "reasons": ["样本不足"]}

    # 找一段"连续上升/下降"作为阶跃响应替代
    pv_smooth = pd.Series(pv).rolling(window=min(11, n // 5), min_periods=1, center=True).mean().to_numpy()
    span = float(np.max(pv_smooth) - np.min(pv_smooth))
    if span < 1e-9:
        return {"inferred_type": "unknown", "time_constant_sec": 0.0, "confidence": 0.0, "reasons": ["PV 无变化"]}

    # 用从 10% → 63% 的跨越时间近似时间常数
    target_lo = float(np.min(pv_smooth)) + 0.10 * span
    target_hi = float(np.min(pv_smooth)) + 0.63 * span
    idx_lo = int(np.argmax(pv_smooth >= target_lo))
    idx_hi = int(np.argmax(pv_smooth >= target_hi))
    if idx_hi <= idx_lo:
        # 尝试反向（下降过程）
        target_lo = float(np.max(pv_smooth)) - 0.10 * span
        target_hi = float(np.max(pv_smooth)) - 0.63 * span
        idx_lo = int(np.argmax(pv_smooth <= target_lo))
        idx_hi = int(np.argmax(pv_smooth <= target_hi))
    if idx_hi <= idx_lo:
        tau = 0.0
    else:
        tau = (idx_hi - idx_lo) * dt

    reasons: list[str] = [f"估计时间常数约 {tau:.1f}s"]
    if tau <= 0:
        return {"inferred_type": "unknown", "time_constant_sec": 0.0, "confidence": 0.0, "reasons": reasons}
    if tau < 10:
        t = "flow"
    elif tau < 60:
        t = "pressure"
    elif tau < 300:
        t = "level"
    else:
        t = "temperature"

    # 置信度仅粗略反映"估计有多靠谱"（有足够样本且能稳定跨越 63%）
    conf = 0.6 if (idx_hi - idx_lo) > 5 else 0.3

    return {
        "inferred_type": t,
        "time_constant_sec": round(tau, 2),
        "confidence": round(conf, 2),
        "reasons": reasons,
    }
