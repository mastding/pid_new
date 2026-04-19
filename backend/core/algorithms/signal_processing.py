"""Signal processing utilities: denoise, detrend, alignment.

Key improvements over pid_new:
- #13: MV never filtered; PV uses median filter (edge-preserving) even for
  high noise, not Butterworth (which smears step edges and biases dead time).
- #5: align_series only searches positive lags (causal: MV causes PV),
  uses normalised cross-correlation with peak validity check.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


# ── Noise estimation ────────────────────────────────────────────────────────

def _robust_noise(values: np.ndarray) -> float:
    """Median absolute deviation of first differences — robust noise estimate."""
    if values.size < 3:
        return 0.0
    diffs = np.diff(values.astype(float))
    med = float(np.median(diffs))
    return float(np.median(np.abs(diffs - med)))


# ── Denoising ───────────────────────────────────────────────────────────────

def _noise_ratio(signal_data: np.ndarray) -> float:
    """Ratio of diff-noise std to signal std (0 = clean, 1 = very noisy)."""
    sig = signal_data.astype(float)
    signal_std = float(np.std(sig))
    if signal_std < 1e-9:
        return 0.0
    noise_std = float(np.std(np.diff(sig)))
    return noise_std / signal_std


def denoise_pv(pv: np.ndarray, noise_level: str = "auto") -> np.ndarray:
    """Denoise PV signal with edge-preserving median filter.

    Uses median filter at all noise levels (not Butterworth) to preserve
    step edges critical for dead-time identification.

    Args:
        pv: Raw PV signal.
        noise_level: 'auto' | 'low' | 'medium' | 'high'.
    """
    pv = np.asarray(pv, dtype=float)
    if pv.size > 200_000:
        return pv

    if noise_level == "auto":
        ratio = _noise_ratio(pv)
        if ratio < 0.05:
            noise_level = "low"
        elif ratio < 0.15:
            noise_level = "medium"
        else:
            noise_level = "high"

    kernel = {"low": 3, "medium": 5, "high": 9}.get(noise_level, 3)
    return median_filter(pv, size=kernel).astype(float)


def denoise_mv(mv: np.ndarray) -> np.ndarray:
    """MV signals should NOT be smoothed — they contain step information.

    Returns the input unchanged. Exists only to make the intent explicit.
    """
    return np.asarray(mv, dtype=float)


# ── Detrending ──────────────────────────────────────────────────────────────

def detrend_if_needed(pv: np.ndarray) -> tuple[np.ndarray, bool]:
    """Remove linear trend from PV if drift > 35% of span.

    Returns:
        (detrended_pv, was_detrended)
    """
    pv = np.asarray(pv, dtype=float)
    if pv.size < 10:
        return pv, False

    span = float(np.max(pv) - np.min(pv))
    if span <= 1e-9:
        return pv, False

    # Fast slope estimate for large arrays
    if pv.size > 5000:
        slope = (pv[-1] - pv[0]) / max(float(pv.size - 1), 1.0)
        intercept = pv[0]
    else:
        x = np.arange(pv.size, dtype=float)
        slope, intercept = [float(v) for v in np.polyfit(x, pv, 1)]

    drift = abs(slope) * float(pv.size - 1)
    if drift / span < 0.35:
        return pv, False

    x = np.arange(pv.size, dtype=float)
    return pv - (slope * x + intercept), True


# ── Series alignment ────────────────────────────────────────────────────────

def align_series(
    mv: np.ndarray,
    pv: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Align MV/PV by finding the optimal positive lag via cross-correlation.

    Fix over pid_new (#5):
    - Only search positive lags (MV must precede PV — causality).
    - Use normalised cross-correlation; require peak > 0.1 to accept.

    Returns:
        (mv_aligned, pv_aligned, lag_steps, correlation_score)
        lag_steps >= 0 always.
    """
    mv = np.asarray(mv, dtype=float)
    pv = np.asarray(pv, dtype=float)

    if mv.size < 20 or pv.size < 20:
        return mv, pv, 0, 0.0

    # Search up to ~60 s of lag, capped at 25% of window length
    max_lag = max(3, min(
        int(round(60.0 / max(dt, 1e-6))),
        mv.size // 4,
    ))

    mv_c = mv - float(np.mean(mv))
    pv_c = pv - float(np.mean(pv))
    mv_std = float(np.std(mv_c))
    pv_std = float(np.std(pv_c))
    if mv_std < 1e-12 or pv_std < 1e-12:
        return mv, pv, 0, 0.0

    best_lag = 0
    best_score = -1.0
    # Only positive lags: MV leads PV
    for lag in range(0, max_lag + 1):
        if lag == 0:
            a, b = mv_c, pv_c
        else:
            a = mv_c[:-lag]
            b = pv_c[lag:]
        if a.size < 15:
            break
        score = float(np.mean(a * b) / (mv_std * pv_std))
        if score > best_score + 1e-9:
            best_score = score
            best_lag = lag

    # Reject weak correlation
    if best_score < 0.1 or best_lag == 0:
        return mv, pv, 0, best_score

    mv_adj = mv[:-best_lag]
    pv_adj = pv[best_lag:]
    if mv_adj.size < 10:
        return mv, pv, 0, best_score

    return mv_adj, pv_adj, best_lag, best_score
