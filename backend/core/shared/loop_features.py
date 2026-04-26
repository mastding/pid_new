"""Raw loop feature extraction for historical/online monitoring.

This module intentionally computes only observable statistics from PV/MV/SP
series. It does not run window detection, oscillation diagnosis, model
identification, tuning readiness, or root-cause diagnosis. Those are separate
skills that should consume these raw facts.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _finite(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _float(value: Any, digits: int = 6) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, digits)


def _int(value: Any) -> int | None:
    try:
        if value is None or not np.isfinite(float(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ")
    return str(value)


def _safe_div(num: float, den: float) -> float | None:
    if not den:
        return None
    return num / den


def _series_stats(series: pd.Series | None, *, sample_time_s: float, duration_h: float) -> dict[str, Any] | None:
    if series is None:
        return None

    raw = pd.to_numeric(series, errors="coerce")
    arr = _finite(raw)
    missing_ratio = float(raw.isna().mean()) if len(raw) else 1.0
    if len(arr) == 0:
        return {
            "available": False,
            "count": 0,
            "missing_ratio": _float(missing_ratio),
        }

    diffs = np.diff(raw.to_numpy(dtype=float))
    finite_diffs = _finite(diffs)
    span = float(np.max(arr) - np.min(arr))
    std = float(np.std(arr))
    mean = float(np.mean(arr))
    abs_steps = np.abs(finite_diffs)
    flat_step_threshold = max(1e-9, span * 0.001)
    large_step_threshold = float(np.percentile(abs_steps, 95)) if len(abs_steps) else 0.0
    if large_step_threshold <= 0:
        large_step_threshold = flat_step_threshold

    return {
        "available": True,
        "count": int(len(arr)),
        "missing_ratio": _float(missing_ratio),
        "min": _float(np.min(arr)),
        "max": _float(np.max(arr)),
        "mean": _float(mean),
        "median": _float(np.median(arr)),
        "std": _float(std),
        "variance": _float(np.var(arr)),
        "p01": _float(np.percentile(arr, 1)),
        "p05": _float(np.percentile(arr, 5)),
        "p25": _float(np.percentile(arr, 25)),
        "p75": _float(np.percentile(arr, 75)),
        "p95": _float(np.percentile(arr, 95)),
        "p99": _float(np.percentile(arr, 99)),
        "iqr": _float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "mad": _float(np.median(np.abs(arr - np.median(arr)))),
        "span": _float(span),
        "cv": _float(_safe_div(std, abs(mean))),
        "skewness": _float(pd.Series(arr).skew()),
        "kurtosis": _float(pd.Series(arr).kurt()),
        "mean_abs_step": _float(np.mean(abs_steps)) if len(abs_steps) else None,
        "median_abs_step": _float(np.median(abs_steps)) if len(abs_steps) else None,
        "p95_abs_step": _float(np.percentile(abs_steps, 95)) if len(abs_steps) else None,
        "p99_abs_step": _float(np.percentile(abs_steps, 99)) if len(abs_steps) else None,
        "max_abs_step": _float(np.max(abs_steps)) if len(abs_steps) else None,
        "rms_step": _float(np.sqrt(np.mean(np.square(abs_steps)))) if len(abs_steps) else None,
        "flat_step_ratio": _float(np.mean(abs_steps <= flat_step_threshold)) if len(abs_steps) else None,
        "large_step_threshold": _float(large_step_threshold),
        "large_step_count": int(np.sum(abs_steps >= large_step_threshold)) if len(abs_steps) else 0,
        "large_step_ratio": _float(np.mean(abs_steps >= large_step_threshold)) if len(abs_steps) else None,
        "trend_slope_per_hour": _float(_trend_slope_per_hour(arr, sample_time_s)),
        "trend_delta_total": _float(arr[-1] - arr[0]),
        "duration_h": _float(duration_h, 3),
    }


def _trend_slope_per_hour(arr: np.ndarray, sample_time_s: float) -> float | None:
    if len(arr) < 2 or sample_time_s <= 0:
        return None
    x_h = np.arange(len(arr), dtype=float) * sample_time_s / 3600.0
    if float(np.max(x_h) - np.min(x_h)) <= 0:
        return None
    return float(np.polyfit(x_h, arr, 1)[0])


def _sample_profile(df: pd.DataFrame, sample_time_s: float) -> dict[str, Any]:
    if "timestamp" not in df.columns or len(df) < 2:
        return {
            "sample_time_median_s": _float(sample_time_s),
            "sample_time_mean_s": _float(sample_time_s),
            "sample_time_min_s": _float(sample_time_s),
            "sample_time_max_s": _float(sample_time_s),
            "sample_time_std_s": 0.0,
            "sample_interval_p95_s": _float(sample_time_s),
            "sample_interval_p99_s": _float(sample_time_s),
        }

    deltas = df["timestamp"].diff().dt.total_seconds().dropna()
    valid = deltas[(deltas > 0) & np.isfinite(deltas)]
    if valid.empty:
        return _sample_profile(df.drop(columns=["timestamp"]), sample_time_s)
    return {
        "sample_time_median_s": _float(valid.median()),
        "sample_time_mean_s": _float(valid.mean()),
        "sample_time_min_s": _float(valid.min()),
        "sample_time_max_s": _float(valid.max()),
        "sample_time_std_s": _float(valid.std(ddof=0)),
        "sample_interval_p95_s": _float(valid.quantile(0.95)),
        "sample_interval_p99_s": _float(valid.quantile(0.99)),
    }


def _data_quality_raw(df: pd.DataFrame, sample_time_s: float) -> dict[str, Any]:
    row_count = len(df)
    cols = [c for c in ["PV", "MV", "SV"] if c in df.columns]
    missing = {c.lower(): float(pd.to_numeric(df[c], errors="coerce").isna().mean()) for c in cols}
    duplicate_timestamp_count = 0
    non_monotonic_timestamp_count = 0
    irregular_sample_count = 0
    long_gap_count = 0
    max_gap_s = None
    if "timestamp" in df.columns and row_count:
        duplicate_timestamp_count = int(df["timestamp"].duplicated().sum())
        deltas = df["timestamp"].diff().dt.total_seconds().dropna()
        non_monotonic_timestamp_count = int((deltas <= 0).sum())
        positive = deltas[deltas > 0]
        if not positive.empty and sample_time_s > 0:
            irregular_sample_count = int((abs(positive - sample_time_s) > max(1e-6, sample_time_s * 0.2)).sum())
            long_gap_count = int((positive > sample_time_s * 3.0).sum())
            max_gap_s = float(positive.max())

    def constant_ratio(col: str) -> float | None:
        if col not in df.columns:
            return None
        arr = _finite(df[col])
        if len(arr) < 2:
            return None
        span = float(np.max(arr) - np.min(arr))
        threshold = max(1e-9, span * 0.001)
        return float(np.mean(np.abs(np.diff(arr)) <= threshold))

    def outlier_count(col: str) -> int | None:
        if col not in df.columns:
            return None
        arr = _finite(df[col])
        if len(arr) < 4:
            return 0
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            return 0
        return int(np.sum((arr < q1 - 3.0 * iqr) | (arr > q3 + 3.0 * iqr)))

    return {
        "missing_ratio_total": _float(float(df[cols].isna().mean().mean())) if cols else None,
        "missing_ratio_pv": _float(missing.get("pv", 1.0)),
        "missing_ratio_mv": _float(missing.get("mv", 1.0)),
        "missing_ratio_sp": _float(missing.get("sv")) if "SV" in df.columns else None,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "duplicate_timestamp_ratio": _float(_safe_div(duplicate_timestamp_count, row_count) or 0.0),
        "non_monotonic_timestamp_count": non_monotonic_timestamp_count,
        "irregular_sample_count": irregular_sample_count,
        "irregular_sample_ratio": _float(_safe_div(irregular_sample_count, max(row_count - 1, 1)) or 0.0),
        "long_gap_count": long_gap_count,
        "max_gap_s": _float(max_gap_s),
        "constant_pv_ratio": _float(constant_ratio("PV")),
        "constant_mv_ratio": _float(constant_ratio("MV")),
        "constant_sp_ratio": _float(constant_ratio("SV")) if "SV" in df.columns else None,
        "pv_outlier_count": outlier_count("PV"),
        "mv_outlier_count": outlier_count("MV"),
        "sp_outlier_count": outlier_count("SV") if "SV" in df.columns else None,
    }


def _mv_stats_extra(mv: pd.Series, sample_time_s: float, duration_h: float) -> dict[str, Any]:
    arr = pd.to_numeric(mv, errors="coerce").to_numpy(dtype=float)
    diffs = _finite(np.diff(arr))
    if len(diffs) == 0:
        return {
            "active_step_ratio": None,
            "move_count": 0,
            "move_count_per_hour": None,
            "direction_reversal_count": 0,
            "direction_reversal_per_hour": None,
            "total_travel": 0.0,
            "travel_per_hour": None,
        }
    abs_steps = np.abs(diffs)
    move_threshold = max(0.03, float(np.percentile(abs_steps, 60)))
    active = abs_steps > move_threshold
    signs = np.sign(diffs)
    reversals = active[1:] & active[:-1] & (signs[1:] * signs[:-1] < 0)
    move_count = int(np.sum(active))
    reversal_count = int(np.sum(reversals))
    total_travel = float(np.sum(abs_steps))
    return {
        "active_step_threshold": _float(move_threshold),
        "active_step_ratio": _float(np.mean(active)),
        "move_count": move_count,
        "move_count_per_hour": _float(_safe_div(move_count, duration_h)),
        "direction_reversal_count": reversal_count,
        "direction_reversal_per_hour": _float(_safe_div(reversal_count, duration_h)),
        "total_travel": _float(total_travel),
        "travel_per_hour": _float(_safe_div(total_travel, duration_h)),
    }


def _sp_stats(sp: pd.Series | None, sample_time_s: float, duration_h: float) -> dict[str, Any]:
    base = _series_stats(sp, sample_time_s=sample_time_s, duration_h=duration_h)
    if base is None:
        return {"available": False, "reason": "SP/SV column not available"}
    if not base.get("available"):
        base["reason"] = "SP/SV column has no finite values"
        return base
    arr = pd.to_numeric(sp, errors="coerce").to_numpy(dtype=float)
    diffs = _finite(np.diff(arr))
    threshold = max(0.2, float(np.percentile(np.abs(diffs), 95))) if len(diffs) else 0.2
    changes = np.abs(diffs) >= threshold if len(diffs) else np.array([], dtype=bool)
    hold_lengths = _hold_segment_lengths(arr, threshold, sample_time_s)
    base.update({
        "change_threshold": _float(threshold),
        "change_count": int(np.sum(changes)) if len(changes) else 0,
        "change_count_per_hour": _float(_safe_div(float(np.sum(changes)), duration_h)) if len(changes) else None,
        "hold_segment_count": len(hold_lengths),
        "median_hold_duration_s": _float(np.median(hold_lengths)) if hold_lengths else None,
        "max_hold_duration_s": _float(np.max(hold_lengths)) if hold_lengths else None,
    })
    return base


def _hold_segment_lengths(arr: np.ndarray, change_threshold: float, sample_time_s: float) -> list[float]:
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return []
    values = arr[finite_mask]
    if len(values) == 0:
        return []
    lengths: list[int] = []
    start = 0
    diffs = np.abs(np.diff(values))
    for i, diff in enumerate(diffs, start=1):
        if diff >= change_threshold:
            lengths.append(i - start)
            start = i
    lengths.append(len(values) - start)
    return [float(n) * sample_time_s for n in lengths if n > 0]


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return None
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) <= 0 or float(np.std(bb)) <= 0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = pd.Series(a).rank().to_numpy(dtype=float)
    bb = pd.Series(b).rank().to_numpy(dtype=float)
    return _corr(aa, bb)


def _best_lag_corr(a: np.ndarray, b: np.ndarray, sample_time_s: float, max_lag_s: float) -> dict[str, Any]:
    max_lag = int(max(0, min(len(a) // 3, round(max_lag_s / sample_time_s)))) if sample_time_s > 0 else 0
    best_corr: float | None = None
    best_lag = 0
    for lag in range(max_lag + 1):
        if lag == 0:
            x, y = a, b
        else:
            x, y = a[:-lag], b[lag:]
        c = _corr(x, y)
        if c is None:
            continue
        if best_corr is None or abs(c) > abs(best_corr):
            best_corr = c
            best_lag = lag
    return {
        "corr": _float(best_corr),
        "lag_s": _float(best_lag * sample_time_s, 3),
    }


def _pv_mv_relation_raw(df: pd.DataFrame, sample_time_s: float) -> dict[str, Any]:
    pv = pd.to_numeric(df["PV"], errors="coerce").to_numpy(dtype=float)
    mv = pd.to_numeric(df["MV"], errors="coerce").to_numpy(dtype=float)
    dpv = np.diff(pd.Series(pv).rolling(3, center=True, min_periods=1).median().to_numpy(dtype=float))
    dmv = np.diff(pd.Series(mv).rolling(3, center=True, min_periods=1).median().to_numpy(dtype=float))
    level_best = _best_lag_corr(mv, pv, sample_time_s, max_lag_s=3600.0)
    diff_best = _best_lag_corr(dmv, dpv, sample_time_s, max_lag_s=3600.0)
    peak_abs = max(abs(level_best["corr"] or 0.0), abs(diff_best["corr"] or 0.0))
    raw_direction = "uncertain"
    direction_corr = diff_best["corr"] if diff_best["corr"] is not None else level_best["corr"]
    if direction_corr is not None and abs(direction_corr) >= 0.08:
        raw_direction = "positive" if direction_corr > 0 else "negative"
    return {
        "pearson_corr_pv_mv": _float(_corr(pv, mv)),
        "spearman_corr_pv_mv": _float(_rank_corr(pv, mv)),
        "pearson_corr_dpv_dmv": _float(_corr(dpv, dmv)),
        "best_lag_corr_pv_mv": level_best["corr"],
        "best_lag_s_pv_mv": level_best["lag_s"],
        "best_lag_corr_dpv_dmv": diff_best["corr"],
        "best_lag_s_dpv_dmv": diff_best["lag_s"],
        "estimated_direction_raw": raw_direction,
        "cross_correlation_peak_abs": _float(peak_abs),
    }


def _sp_tracking_raw(df: pd.DataFrame) -> dict[str, Any]:
    if "SV" not in df.columns:
        return {"sp_available": False, "reason": "SP/SV column not available"}
    pv = pd.to_numeric(df["PV"], errors="coerce").to_numpy(dtype=float)
    sp = pd.to_numeric(df["SV"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(pv) & np.isfinite(sp)
    if int(mask.sum()) == 0:
        return {"sp_available": False, "reason": "SP/SV column has no finite values"}
    err = pv[mask] - sp[mask]
    abs_err = np.abs(err)
    pv_span = float(np.nanmax(pv[mask]) - np.nanmin(pv[mask]))
    t = np.arange(len(err), dtype=float)
    return {
        "sp_available": True,
        "error_mean": _float(np.mean(err)),
        "error_median": _float(np.median(err)),
        "error_std": _float(np.std(err)),
        "error_abs_mean": _float(np.mean(abs_err)),
        "error_abs_median": _float(np.median(abs_err)),
        "error_abs_p95": _float(np.percentile(abs_err, 95)),
        "error_abs_max": _float(np.max(abs_err)),
        "error_rms": _float(np.sqrt(np.mean(np.square(err)))),
        "error_bias": _float(np.mean(err)),
        "error_within_1pct_span_ratio": _float(np.mean(abs_err <= pv_span * 0.01)) if pv_span else None,
        "error_within_2pct_span_ratio": _float(np.mean(abs_err <= pv_span * 0.02)) if pv_span else None,
        "error_within_5pct_span_ratio": _float(np.mean(abs_err <= pv_span * 0.05)) if pv_span else None,
        "iae": _float(np.sum(abs_err)),
        "itae": _float(np.sum(t * abs_err)),
    }


def _event_raw(df: pd.DataFrame, sample_time_s: float, duration_h: float) -> dict[str, Any]:
    pv = pd.to_numeric(df["PV"], errors="coerce").to_numpy(dtype=float)
    mv = pd.to_numeric(df["MV"], errors="coerce").to_numpy(dtype=float)
    sp = pd.to_numeric(df["SV"], errors="coerce").to_numpy(dtype=float) if "SV" in df.columns else None

    pv_diff = _finite(np.diff(pv))
    mv_diff = _finite(np.diff(mv))
    sp_diff = _finite(np.diff(sp)) if sp is not None else np.array([], dtype=float)
    pv_threshold = _change_threshold(pv_diff, floor=0.05)
    mv_threshold = _change_threshold(mv_diff, floor=0.05)
    sp_threshold = _change_threshold(sp_diff, floor=0.2)

    pv_large = np.abs(np.diff(pv)) >= pv_threshold if len(pv) > 1 else np.array([], dtype=bool)
    mv_large = np.abs(np.diff(mv)) >= mv_threshold if len(mv) > 1 else np.array([], dtype=bool)
    sp_large = np.abs(np.diff(sp)) >= sp_threshold if sp is not None and len(sp) > 1 else np.array([], dtype=bool)
    ramp_window_s = 300.0
    ramp_n = max(2, int(round(ramp_window_s / sample_time_s))) if sample_time_s > 0 else 2
    mv_ramp = pd.Series(mv).diff(ramp_n).abs().to_numpy(dtype=float)
    mv_ramp_threshold = max(0.5, float(np.nanpercentile(mv_ramp, 95))) if np.isfinite(mv_ramp).any() else 0.5
    mv_ramp_events = np.isfinite(mv_ramp) & (mv_ramp >= mv_ramp_threshold)

    common_len = min(len(pv_large), len(mv_large))
    simultaneous = int(np.sum(pv_large[:common_len] & mv_large[:common_len])) if common_len else 0
    pv_without_mv = int(np.sum(pv_large[:common_len] & ~mv_large[:common_len])) if common_len else 0
    mv_without_pv = int(np.sum(mv_large[:common_len] & ~pv_large[:common_len])) if common_len else 0

    return {
        "pv_large_change_count": int(np.sum(pv_large)),
        "pv_large_change_per_hour": _float(_safe_div(float(np.sum(pv_large)), duration_h)),
        "pv_large_change_threshold": _float(pv_threshold),
        "mv_adjacent_change_count": int(np.sum(mv_large)),
        "mv_adjacent_change_per_hour": _float(_safe_div(float(np.sum(mv_large)), duration_h)),
        "mv_adjacent_change_threshold": _float(mv_threshold),
        "mv_ramp_change_count": int(np.sum(mv_ramp_events)),
        "mv_ramp_change_per_hour": _float(_safe_div(float(np.sum(mv_ramp_events)), duration_h)),
        "mv_ramp_window_s": _float(ramp_window_s),
        "mv_ramp_change_threshold": _float(mv_ramp_threshold),
        "sp_change_count": int(np.sum(sp_large)) if len(sp_large) else 0,
        "sp_change_per_hour": _float(_safe_div(float(np.sum(sp_large)), duration_h)) if len(sp_large) else None,
        "sp_change_threshold": _float(sp_threshold) if len(sp_diff) else None,
        "simultaneous_sp_mv_change_count": _simultaneous_count(sp_large, mv_large),
        "pv_change_without_mv_change_count": pv_without_mv,
        "mv_change_without_pv_change_count": mv_without_pv,
        "simultaneous_pv_mv_change_count": simultaneous,
    }


def _change_threshold(diffs: np.ndarray, *, floor: float) -> float:
    if len(diffs) == 0:
        return floor
    abs_diff = np.abs(diffs)
    mad = float(np.median(np.abs(abs_diff - np.median(abs_diff))))
    return max(floor, float(np.percentile(abs_diff, 95)), mad * 6.0)


def _simultaneous_count(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    return int(np.sum(a[:n] & b[:n]))


def _constraint_raw(df: pd.DataFrame, sample_time_s: float) -> dict[str, Any]:
    pv = pd.to_numeric(df["PV"], errors="coerce").to_numpy(dtype=float)
    mv = pd.to_numeric(df["MV"], errors="coerce").to_numpy(dtype=float)
    sp = pd.to_numeric(df["SV"], errors="coerce").to_numpy(dtype=float) if "SV" in df.columns else None
    mv_low = np.isfinite(mv) & (mv <= 1.0)
    mv_high = np.isfinite(mv) & (mv >= 99.0)
    mv_sat = mv_low | mv_high
    seg_lengths = _true_segment_lengths(mv_sat, sample_time_s)
    return {
        "pv_near_observed_min_ratio": _near_percentile_ratio(pv, 1.0, lower=True),
        "pv_near_observed_max_ratio": _near_percentile_ratio(pv, 99.0, lower=False),
        "mv_near_0_ratio": _float(np.mean(mv_low)) if len(mv) else None,
        "mv_near_100_ratio": _float(np.mean(mv_high)) if len(mv) else None,
        "mv_saturation_ratio": _float(np.mean(mv_sat)) if len(mv) else None,
        "mv_high_saturation_ratio": _float(np.mean(mv_high)) if len(mv) else None,
        "mv_low_saturation_ratio": _float(np.mean(mv_low)) if len(mv) else None,
        "sp_near_observed_min_ratio": _near_percentile_ratio(sp, 1.0, lower=True) if sp is not None else None,
        "sp_near_observed_max_ratio": _near_percentile_ratio(sp, 99.0, lower=False) if sp is not None else None,
        "longest_mv_saturation_duration_s": _float(max(seg_lengths)) if seg_lengths else 0.0,
        "mv_saturation_segment_count": len(seg_lengths),
    }


def _near_percentile_ratio(arr: np.ndarray | None, percentile: float, *, lower: bool) -> float | None:
    if arr is None:
        return None
    finite = _finite(arr)
    if len(finite) == 0:
        return None
    threshold = float(np.percentile(finite, percentile))
    if lower:
        return _float(np.mean(finite <= threshold))
    return _float(np.mean(finite >= threshold))


def _true_segment_lengths(mask: np.ndarray, sample_time_s: float) -> list[float]:
    lengths: list[float] = []
    run = 0
    for value in mask:
        if bool(value):
            run += 1
        elif run:
            lengths.append(run * sample_time_s)
            run = 0
    if run:
        lengths.append(run * sample_time_s)
    return lengths


def _frequency_raw(df: pd.DataFrame, sample_time_s: float) -> dict[str, Any]:
    pv = pd.to_numeric(df["PV"], errors="coerce")
    mv = pd.to_numeric(df["MV"], errors="coerce")
    pv_det = _detrended(pv, sample_time_s)
    mv_det = _detrended(mv, sample_time_s)
    pv_freq = _dominant_frequency(pv_det, sample_time_s)
    mv_freq = _dominant_frequency(mv_det, sample_time_s)
    return {
        "pv_detrended_std": _float(np.std(pv_det)) if len(pv_det) else None,
        "mv_detrended_std": _float(np.std(mv_det)) if len(mv_det) else None,
        "pv_zero_crossing_count": _zero_crossings(pv_det),
        "pv_zero_crossing_per_hour": _float(_safe_div(float(_zero_crossings(pv_det)), _duration_h(len(pv_det), sample_time_s))),
        "mv_zero_crossing_count": _zero_crossings(mv_det),
        "mv_zero_crossing_per_hour": _float(_safe_div(float(_zero_crossings(mv_det)), _duration_h(len(mv_det), sample_time_s))),
        "pv_dominant_period_s": pv_freq["period_s"],
        "pv_dominant_frequency_hz": pv_freq["frequency_hz"],
        "pv_dominant_power_ratio": pv_freq["power_ratio"],
        "mv_dominant_period_s": mv_freq["period_s"],
        "mv_dominant_frequency_hz": mv_freq["frequency_hz"],
        "mv_dominant_power_ratio": mv_freq["power_ratio"],
    }


def _detrended(series: pd.Series, sample_time_s: float) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce").interpolate(limit_direction="both")
    if s.empty:
        return np.array([], dtype=float)
    win = max(5, int(round(1800.0 / max(sample_time_s, 1e-6))))
    if win % 2 == 0:
        win += 1
    win = min(win, max(5, len(s) // 2 * 2 - 1)) if len(s) > 10 else 5
    baseline = s.rolling(win, center=True, min_periods=max(3, win // 5)).median()
    detrended = (s - baseline).dropna().to_numpy(dtype=float)
    return detrended[np.isfinite(detrended)]


def _dominant_frequency(arr: np.ndarray, sample_time_s: float) -> dict[str, Any]:
    if len(arr) < 8 or sample_time_s <= 0 or float(np.std(arr)) <= 0:
        return {"period_s": None, "frequency_hz": None, "power_ratio": None}
    x = arr - np.mean(arr)
    if len(x) > 8192:
        step = int(np.ceil(len(x) / 8192))
        x = x[::step]
        effective_dt = sample_time_s * step
    else:
        effective_dt = sample_time_s
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=effective_dt)
    if len(power) <= 1:
        return {"period_s": None, "frequency_hz": None, "power_ratio": None}
    power[0] = 0.0
    idx = int(np.argmax(power))
    freq = float(freqs[idx])
    total_power = float(np.sum(power))
    return {
        "period_s": _float(1.0 / freq, 3) if freq > 0 else None,
        "frequency_hz": _float(freq, 9),
        "power_ratio": _float(power[idx] / total_power) if total_power > 0 else None,
    }


def _zero_crossings(arr: np.ndarray) -> int:
    if len(arr) < 2:
        return 0
    return int(np.sum(np.signbit(arr[1:]) != np.signbit(arr[:-1])))


def _duration_h(n: int, sample_time_s: float) -> float:
    return max(0.0, (n - 1) * sample_time_s / 3600.0) if n else 0.0


def _stationarity_raw(df: pd.DataFrame, sample_time_s: float) -> dict[str, Any]:
    pv = pd.to_numeric(df["PV"], errors="coerce").to_numpy(dtype=float)
    mv = pd.to_numeric(df["MV"], errors="coerce").to_numpy(dtype=float)
    return {
        **_half_stats("pv", pv),
        **_half_stats("mv", mv),
        "pv_mean_shift_count": _rolling_shift_count(pv, sample_time_s, stat="mean"),
        "mv_mean_shift_count": _rolling_shift_count(mv, sample_time_s, stat="mean"),
        "pv_variance_shift_count": _rolling_shift_count(pv, sample_time_s, stat="std"),
        "mv_variance_shift_count": _rolling_shift_count(mv, sample_time_s, stat="std"),
        "rolling_pv_std_median": _rolling_std_stat(pv, sample_time_s, "median"),
        "rolling_pv_std_p95": _rolling_std_stat(pv, sample_time_s, "p95"),
        "rolling_mv_std_median": _rolling_std_stat(mv, sample_time_s, "median"),
        "rolling_mv_std_p95": _rolling_std_stat(mv, sample_time_s, "p95"),
    }


def _half_stats(prefix: str, arr: np.ndarray) -> dict[str, Any]:
    finite = _finite(arr)
    if len(finite) < 2:
        return {
            f"{prefix}_first_half_mean": None,
            f"{prefix}_second_half_mean": None,
            f"{prefix}_first_second_mean_delta": None,
        }
    mid = len(finite) // 2
    first = float(np.mean(finite[:mid]))
    second = float(np.mean(finite[mid:]))
    return {
        f"{prefix}_first_half_mean": _float(first),
        f"{prefix}_second_half_mean": _float(second),
        f"{prefix}_first_second_mean_delta": _float(second - first),
    }


def _rolling_shift_count(arr: np.ndarray, sample_time_s: float, *, stat: str) -> int:
    finite = pd.Series(arr).interpolate(limit_direction="both")
    if len(finite) < 10:
        return 0
    win = max(5, int(round(3600.0 / max(sample_time_s, 1e-6))))
    win = min(win, max(5, len(finite) // 4))
    if win < 5:
        return 0
    values = finite.rolling(win, min_periods=max(3, win // 3)).mean()
    if stat == "std":
        values = finite.rolling(win, min_periods=max(3, win // 3)).std()
    diffs = _finite(values.diff().to_numpy(dtype=float))
    if len(diffs) == 0:
        return 0
    threshold = max(float(np.percentile(np.abs(diffs), 95)), float(np.std(diffs)) * 2.0)
    return int(np.sum(np.abs(diffs) >= threshold))


def _rolling_std_stat(arr: np.ndarray, sample_time_s: float, mode: str) -> float | None:
    s = pd.Series(arr).interpolate(limit_direction="both")
    if len(s) < 5:
        return None
    win = max(5, int(round(1800.0 / max(sample_time_s, 1e-6))))
    win = min(win, max(5, len(s) // 3))
    values = _finite(s.rolling(win, min_periods=max(3, win // 3)).std().to_numpy(dtype=float))
    if len(values) == 0:
        return None
    if mode == "p95":
        return _float(np.percentile(values, 95))
    return _float(np.median(values))


def _noise_raw(df: pd.DataFrame, sample_time_s: float) -> dict[str, Any]:
    """Estimate observable high-frequency noise and spike evidence."""
    result: dict[str, Any] = {}
    for col in ["PV", "MV"]:
        if col not in df.columns:
            continue
        key = col.lower()
        s = pd.to_numeric(df[col], errors="coerce").interpolate(limit_direction="both")
        arr = s.to_numpy(dtype=float)
        finite = _finite(arr)
        if len(finite) < 8:
            result.update({
                f"{key}_noise_residual_std": None,
                f"{key}_noise_ratio": None,
                f"{key}_snr_db": None,
                f"{key}_spike_threshold": None,
                f"{key}_spike_count": 0,
                f"{key}_spike_ratio": None,
            })
            continue

        span = max(float(np.nanmax(finite) - np.nanmin(finite)), 1e-9)
        win = max(5, int(round(300.0 / max(sample_time_s, 1e-6))))
        win = min(win, max(5, len(s) // 5))
        if win % 2 == 0:
            win += 1
        baseline = s.rolling(win, center=True, min_periods=max(3, win // 3)).median()
        residual = _finite((s - baseline).to_numpy(dtype=float))
        residual_std = float(np.std(residual)) if len(residual) else 0.0
        signal_std = float(np.std(finite))
        noise_ratio = residual_std / span
        snr = 20.0 * np.log10(signal_std / residual_std) if residual_std > 1e-12 and signal_std > 0 else None

        diffs = _finite(np.diff(arr))
        if len(diffs):
            diff_abs = np.abs(diffs)
            diff_median = float(np.median(diff_abs))
            diff_mad = float(np.median(np.abs(diff_abs - diff_median)))
            spike_threshold = max(span * 0.05, diff_median + 8.0 * diff_mad, float(np.percentile(diff_abs, 99)))
            spike_count = int(np.sum(diff_abs >= spike_threshold))
            spike_ratio = spike_count / len(diff_abs)
        else:
            spike_threshold = None
            spike_count = 0
            spike_ratio = None

        result.update({
            f"{key}_noise_residual_std": _float(residual_std),
            f"{key}_noise_ratio": _float(noise_ratio),
            f"{key}_snr_db": _float(snr, 3),
            f"{key}_spike_threshold": _float(spike_threshold),
            f"{key}_spike_count": spike_count,
            f"{key}_spike_ratio": _float(spike_ratio),
        })
    return result


def _oscillation_raw(df: pd.DataFrame, sample_time_s: float, frequency_raw: dict[str, Any]) -> dict[str, Any]:
    """Detect periodic components as monitoring evidence, not root cause."""
    pv_power = float(frequency_raw.get("pv_dominant_power_ratio") or 0.0)
    mv_power = float(frequency_raw.get("mv_dominant_power_ratio") or 0.0)
    pv_period = frequency_raw.get("pv_dominant_period_s")
    mv_period = frequency_raw.get("mv_dominant_period_s")
    pv_zc = float(frequency_raw.get("pv_zero_crossing_per_hour") or 0.0)
    mv_zc = float(frequency_raw.get("mv_zero_crossing_per_hour") or 0.0)

    detected = False
    severity = "normal"
    confidence = 0.0
    if pv_period and pv_power >= 0.22 and pv_zc >= 4:
        detected = True
        confidence = min(1.0, 0.45 + pv_power + min(0.25, pv_zc / 120.0))
        severity = "alarm" if pv_power >= 0.45 and pv_zc >= 10 else "warning"
    elif pv_power >= 0.18 and pv_zc >= 8:
        detected = True
        confidence = min(0.75, 0.35 + pv_power + min(0.2, pv_zc / 180.0))
        severity = "warning"

    phase_hint = "unknown"
    if detected and pv_period and mv_period:
        try:
            period_delta = abs(float(pv_period) - float(mv_period)) / max(float(pv_period), 1e-9)
            if period_delta <= 0.2 and mv_power >= 0.12:
                phase_hint = "pv_mv_same_period"
            elif mv_power < 0.08:
                phase_hint = "pv_only_periodic"
        except Exception:
            phase_hint = "unknown"

    return {
        "detected": detected,
        "severity": severity,
        "confidence": _float(confidence),
        "pv_dominant_period_s": pv_period,
        "pv_dominant_power_ratio": frequency_raw.get("pv_dominant_power_ratio"),
        "pv_zero_crossing_per_hour": frequency_raw.get("pv_zero_crossing_per_hour"),
        "mv_dominant_period_s": mv_period,
        "mv_dominant_power_ratio": frequency_raw.get("mv_dominant_power_ratio"),
        "mv_zero_crossing_per_hour": frequency_raw.get("mv_zero_crossing_per_hour"),
        "phase_hint": phase_hint,
        "sample_time_s": _float(sample_time_s),
    }


def _operating_summary_raw(
    *,
    data_quality: dict[str, Any],
    pv_stats: dict[str, Any] | None,
    mv_stats: dict[str, Any] | None,
    sp_stats: dict[str, Any],
    event_raw: dict[str, Any],
    df: pd.DataFrame,
    duration_h: float,
) -> dict[str, Any]:
    return {
        "pv_activity_level_raw": _activity_level(pv_stats),
        "mv_activity_level_raw": _activity_level(mv_stats),
        "sp_activity_level_raw": _activity_level(sp_stats if sp_stats.get("available") else None),
        "data_has_enough_duration": duration_h >= 1.0,
        "data_has_enough_mv_excitation": (event_raw.get("mv_adjacent_change_count") or 0) >= 3 or (mv_stats or {}).get("span", 0) >= 1.0,
        "data_has_sp": bool(sp_stats.get("available")),
        "data_has_timestamp": "timestamp" in df.columns,
        "data_has_major_quality_issue_raw": bool(
            (data_quality.get("missing_ratio_total") or 0) > 0.2
            or (data_quality.get("irregular_sample_ratio") or 0) > 0.2
        ),
    }


def _activity_level(stats: dict[str, Any] | None) -> str:
    if not stats or not stats.get("available"):
        return "unavailable"
    span = float(stats.get("span") or 0.0)
    std = float(stats.get("std") or 0.0)
    p95_step = float(stats.get("p95_abs_step") or 0.0)
    scale = max(abs(float(stats.get("mean") or 0.0)), span, 1.0)
    rel = max(span / scale, std / scale, p95_step / scale)
    if rel >= 0.2:
        return "high"
    if rel >= 0.05:
        return "medium"
    if rel > 0:
        return "low"
    return "flat"


def extract_loop_features(
    df: pd.DataFrame,
    *,
    loop_id: str,
    loop_type: str = "unknown",
    source_file: str | None = None,
    dataset_id: str | None = None,
    sample_time_s: float | None = None,
    loop_name: str | None = None,
    tag_prefix: str | None = None,
) -> dict[str, Any]:
    """Extract raw, JSON-safe loop features from normalized history data."""
    work = df.copy()
    for col in ["PV", "MV", "SV"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")

    if "PV" not in work.columns or "MV" not in work.columns:
        raise ValueError("LoopFeatures requires PV and MV columns")

    row_count = int(len(work))
    sample_time = float(sample_time_s or 1.0)
    if sample_time_s is None and "timestamp" in work.columns and row_count > 1:
        deltas = work["timestamp"].diff().dt.total_seconds().dropna()
        deltas = deltas[(deltas > 0) & np.isfinite(deltas)]
        if not deltas.empty:
            sample_time = float(deltas.median())

    duration_s = max(0.0, float(row_count - 1) * sample_time)
    duration_h = duration_s / 3600.0 if duration_s else 0.0
    valid_cols = [c for c in ["PV", "MV", "SV"] if c in work.columns]
    valid_row_count = int(work[valid_cols].dropna(how="any").shape[0]) if valid_cols else 0

    data_profile = {
        "row_count": row_count,
        "valid_row_count": valid_row_count,
        "time_start": _iso(work["timestamp"].iloc[0]) if "timestamp" in work.columns and row_count else None,
        "time_end": _iso(work["timestamp"].iloc[-1]) if "timestamp" in work.columns and row_count else None,
        "duration_s": _float(duration_s, 3),
        "duration_h": _float(duration_h, 3),
        **_sample_profile(work, sample_time),
    }

    pv_stats = _series_stats(work["PV"], sample_time_s=sample_time, duration_h=duration_h)
    mv_stats = _series_stats(work["MV"], sample_time_s=sample_time, duration_h=duration_h)
    if mv_stats:
        mv_stats.update(_mv_stats_extra(work["MV"], sample_time, duration_h))
    sp_stats = _sp_stats(work["SV"] if "SV" in work.columns else None, sample_time, duration_h)
    data_quality = _data_quality_raw(work, sample_time)
    event_raw = _event_raw(work, sample_time, duration_h)
    frequency_raw = _frequency_raw(work, sample_time)

    return {
        "identity": {
            "loop_id": loop_id,
            "loop_name": loop_name or loop_id,
            "loop_type": loop_type or "unknown",
            "source_file": source_file,
            "dataset_id": dataset_id,
            "tag_prefix": tag_prefix,
            "pv_column": "PV",
            "mv_column": "MV",
            "sp_column": "SV" if "SV" in work.columns else None,
            "timestamp_column": "timestamp" if "timestamp" in work.columns else None,
        },
        "data_profile": data_profile,
        "data_quality_raw": data_quality,
        "pv_stats": pv_stats,
        "mv_stats": mv_stats,
        "sp_stats": sp_stats,
        "pv_mv_relation_raw": _pv_mv_relation_raw(work, sample_time),
        "sp_tracking_raw": _sp_tracking_raw(work),
        "event_raw": event_raw,
        "constraint_raw": _constraint_raw(work, sample_time),
        "frequency_raw": frequency_raw,
        "noise_raw": _noise_raw(work, sample_time),
        "oscillation_raw": _oscillation_raw(work, sample_time, frequency_raw),
        "stationarity_raw": _stationarity_raw(work, sample_time),
        "operating_summary_raw": _operating_summary_raw(
            data_quality=data_quality,
            pv_stats=pv_stats,
            mv_stats=mv_stats,
            sp_stats=sp_stats,
            event_raw=event_raw,
            df=work,
            duration_h=duration_h,
        ),
    }
