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

def analyze_deadzone(
    df: pd.DataFrame,
    pv_noise_std: float,
    dt: float = 1.0,
    loop_type: str | None = None,
) -> dict[str, Any]:
    """估算执行器/传感器死区。

    旧算法只比较 ΔMV[t] 与 ΔPV[t+1]（1 秒瞬时），对任何 T>dt 的过程
    都会把"过程滞后响应"误判成"死区"。FIC_21005 这种 T≈3-5s 流量回路
    在旧算法下报 54% 死区，给 PV 5s 时间观察则降到 5.6%。

    新算法两点改进：
    1. 滞后窗按 loop_type 给（flow 10s / pressure 60s / temperature 300s /
       level 600s，默认 30s），让 PV 有合理时间响应再判
    2. MV 阶跃阈值用 max(量程 1%, 8×MV抖动MAD)，区分"真指令"和"控制器抖动"
    """
    mv = df["MV"].to_numpy(dtype=float)
    pv = df["PV"].to_numpy(dtype=float)
    n = len(mv)
    base_ret = {
        "evidence_ratio": 0.0,
        "evidence_count": 0,
        "events_total": 0,
        "estimated_width": 0.0,
        "lag_used_s": 0.0,
        "mv_step_threshold": 0.0,
    }
    if n < 30:
        return base_ret

    mv_span = float(np.max(mv) - np.min(mv))
    if mv_span < 1e-9:
        return base_ret

    # MV "抖动"基线：一阶差分 MAD（与噪声估计同套路）
    dmv_raw = np.diff(mv)
    mv_mad = 1.4826 * float(np.median(np.abs(dmv_raw - np.median(dmv_raw))))
    mv_jitter = mv_mad / np.sqrt(2.0)

    # 滞后窗：按回路类型给 PV 多少时间响应
    LAG_S = {"flow": 10.0, "pressure": 60.0, "temperature": 300.0, "level": 600.0}
    lag_s = LAG_S.get((loop_type or "").strip().lower(), 30.0)
    lag = max(3, int(round(lag_s / max(dt, 1e-6))))
    lag = min(lag, max(3, n // 4))  # 不超过数据 1/4 长度
    # PV "未响应"阈值：噪声 σ 的 4 倍。k 点窗口内 N(0,σ²) 的 max-min 约 3σ，
    # 用 4σ 才能稳健判定"PV 真的没动"，避免把噪声波动误算成响应。
    pv_thr = max(4.0 * pv_noise_std, 1e-9)

    # 真"MV 阶跃"阈值：比抖动大 8 倍才算指令，量程 1% 兜底
    mv_step_thr = max(mv_span * 0.01, 8.0 * mv_jitter)

    # MV 平滑（去毛刺），再在滞后窗里看 max-min
    win = max(3, min(lag // 2, 9))
    mv_smooth = pd.Series(mv).rolling(win, min_periods=1, center=True).mean().to_numpy()

    # 不重叠扫描：步长 = lag/2，每个候选窗口看 mv 摆幅 vs pv 摆幅
    stride = max(1, lag // 2)
    events = 0
    evidence = 0
    widths: list[float] = []
    for i in range(0, n - lag + 1, stride):
        mv_seg = mv_smooth[i:i + lag]
        pv_seg = pv[i:i + lag]
        mv_range = float(mv_seg.max() - mv_seg.min())
        if mv_range < mv_step_thr:
            continue
        events += 1
        pv_range = float(pv_seg.max() - pv_seg.min())
        if pv_range < pv_thr:
            evidence += 1
            widths.append(mv_range)

    ratio = evidence / events if events > 0 else 0.0
    width_est = float(np.percentile(widths, 70)) if widths else 0.0

    return {
        "evidence_count": evidence,
        "evidence_ratio": round(ratio, 4),
        "events_total": events,
        "estimated_width": round(width_est, 4),
        "lag_used_s": round(lag * dt, 2),
        "mv_step_threshold": round(mv_step_thr, 4),
    }


# ── 噪声水平 ────────────────────────────────────────────────────────────

def analyze_noise(df: pd.DataFrame) -> dict[str, Any]:
    """估算 PV 噪声：一阶差分的 MAD 是对独立白噪声 σ 的稳健估计。

    返回的 pv_noise_std 供死区/阈值计算复用。

    三处优化（vs 旧版）：
    1. 量化栅格兜底：当 DCS 把 PV 量化到固定栅格（如 0.01）时，diff 中
       大量为 0，MAD → 0，σ 被严重低估。用相邻不同值的最小间距推算栅格，
       并以 grid/2 作为 σ 下限。
    2. SNR 用 detrend 后 std：原口径用 max-min(PV) 当信号能量，会被 SP
       变化、漂移吹大。改用线性 detrend 后的 std 表征"真实信号能量"。
    3. 分级阈值放宽：DCS 流量计典型噪声 0.3-1.5%，原 0.5% 阈值过严。
       改为 low <0.5% / medium <1.5% / high ≥1.5%。
    """
    pv = df["PV"].to_numpy(dtype=float)
    if len(pv) < 20:
        return {"pv_noise_std": 0.0, "snr_db": 0.0, "noise_level": "unknown"}

    diffs = np.diff(pv)
    # MAD 方法：1.4826 * median(|x|) / sqrt(2) ≈ σ (一阶差分后方差翻倍)
    mad = 1.4826 * float(np.median(np.abs(diffs - np.median(diffs))))
    noise_std = mad / np.sqrt(2.0)

    # ── 优化 1：量化栅格兜底 ────────────────────────────────────────
    # 量化情况下 diff 大量为 0，MAD 退化。用非零 |diff| 的最小值做栅格推断。
    nz = np.abs(diffs[np.abs(diffs) > 1e-12])
    if nz.size > 0:
        # 用 5% 分位数代替 min，避开浮点尾巴；若该值在数据中频繁出现则视作栅格步长
        quant_grid = float(np.percentile(nz, 5))
        # 仅在 σ 估计明显小于栅格的一半时兜底（说明 MAD 被量化吃掉了）
        floor = 0.5 * quant_grid
        if noise_std < floor:
            noise_std = floor

    # ── 优化 2：SNR 用 detrend 后 std 表征"真实信号能量"────────────
    # 线性去趋势：剔除慢漂移与 SP 整体变化，剩下的 std 才是过程响应能量
    n = len(pv)
    if n >= 2:
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, pv, 1)
        pv_detrended = pv - (slope * x + intercept)
        signal_std = float(np.std(pv_detrended))
    else:
        signal_std = 0.0
    snr = 20.0 * np.log10(signal_std / noise_std) if noise_std > 1e-9 and signal_std > 1e-9 else 0.0

    # ── 优化 3：分级阈值放宽 ─────────────────────────────────────────
    # 相对量程仍用原 max-min，方便和现有指标对比；分级口径放宽到 DCS 实际范围
    signal_span = float(np.max(pv) - np.min(pv))
    rel = noise_std / signal_span if signal_span > 1e-9 else 0.0
    if rel < 0.005:
        level = "low"
    elif rel < 0.015:
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


