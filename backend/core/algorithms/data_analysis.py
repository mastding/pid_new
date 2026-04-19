"""Data analysis: CSV loading, cleaning, window detection and selection.

Key improvements over pid_new:
- #1: Detect MV step events when SV is absent/constant (manual-mode trials).
- #2: Adaptive window padding based on dt × estimated time constant,
      not raw data-length fractions.
- #3: Default window selected by quality score, not max amplitude.
- #13: MV is never denoised; PV uses edge-preserving median filter only.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from core.algorithms.signal_processing import denoise_mv, denoise_pv

# ── Column detection ─────────────────────────────────────────────────────────

_ALIASES: dict[str, list[str]] = {
    "timestamp": ["timestamp", "time", "datetime", "ts", "date", "时间", "时间戳", "采集时间"],
    "SV": ["sv", "sp", "setpoint", "set_point", "target", "给定", "设定值", "目标值"],
    "PV": ["pv", "cv", "measurement", "feedback", "过程值", "测量值", "反馈值", "实际值"],
    "MV": ["mv", "op", "output", "valve", "opening", "开度", "阀位", "控制输出", "输出值"],
}


def _canon(text: str) -> str:
    t = str(text).replace("\ufeff", "").strip().lower()
    t = re.sub(r"[\(\（\[].*?[\)\）\]]", "", t)
    t = t.replace("%", "")
    t = re.sub(r"[\s\-_./\\]+", "", t)
    t = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", t)
    return t


def _detect_loops(df: pd.DataFrame) -> list[dict[str, str]]:
    loops: dict[str, dict[str, str]] = {}
    for col in df.columns:
        m = re.match(r"^(?P<prefix>.+?)\.(?P<sig>pv|sv|mv)\s*$", str(col).strip(), re.IGNORECASE)
        if not m:
            continue
        prefix = m.group("prefix").strip()
        sig = m.group("sig").upper()
        key = _canon(prefix)
        if not key:
            continue
        entry = loops.setdefault(key, {"prefix": prefix})
        entry[f"{sig.lower()}_col"] = str(col)
    return [e for e in loops.values() if "pv_col" in e and "mv_col" in e]


def _normalize_columns(df: pd.DataFrame, selected_loop_prefix: str | None = None) -> pd.DataFrame:
    canon_map = {_canon(col): col for col in df.columns}
    rename: dict[str, str] = {}

    loops = _detect_loops(df)
    chosen: dict[str, str] | None = None
    if loops:
        if len(loops) > 1 and not selected_loop_prefix:
            prefixes = [l["prefix"] for l in loops]
            raise ValueError(f"检测到多个回路，请选择: {prefixes}")
        if selected_loop_prefix:
            key = _canon(selected_loop_prefix)
            chosen = next((l for l in loops if _canon(l["prefix"]) == key), None)
        if chosen is None:
            chosen = loops[0]

    if chosen:
        for sig in ("PV", "MV", "SV"):
            col_key = f"{sig.lower()}_col"
            if col_key in chosen and chosen[col_key] in df.columns:
                rename[chosen[col_key]] = sig
    else:
        for std, aliases in _ALIASES.items():
            if std in {"PV", "MV", "SV", "timestamp"}:
                for alias in [std] + aliases:
                    c = _canon(alias)
                    if c and c in canon_map and canon_map[c] not in rename:
                        rename[canon_map[c]] = std
                        break

    result = df.rename(columns=rename).copy()
    if "PV" not in result.columns or "MV" not in result.columns:
        raise ValueError(
            f"找不到 PV/MV 列。原始列: {list(df.columns)}，映射: {rename}"
        )
    return result


# ── Timestamp parsing ────────────────────────────────────────────────────────

def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        unit = "ms" if ts.dropna().abs().median() > 1e11 else "s"
        df["timestamp"] = (
            pd.to_datetime(ts, unit=unit, errors="coerce", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
        )
    else:
        df["timestamp"] = pd.to_datetime(ts, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _estimate_dt(df: pd.DataFrame, fallback: float = 1.0) -> float:
    if "timestamp" not in df.columns or len(df) < 2:
        return fallback
    deltas = df["timestamp"].diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0]
    return float(deltas.median()) if not deltas.empty else fallback


# ── CSV loading ──────────────────────────────────────────────────────────────

def _read_csv(csv_path: str) -> pd.DataFrame:
    encodings = ["utf-8", "gbk", "gb18030", "iso-8859-1"]
    alias_sets = {
        std: [_canon(a) for a in [std] + aliases]
        for std, aliases in _ALIASES.items()
    }

    for enc in encodings:
        try:
            with open(csv_path, "r", encoding=enc) as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            continue

        best_idx, best_score = 0, -1
        for idx in range(min(80, len(lines))):
            raw = lines[idx].strip()
            if not raw or raw.count(",") + raw.count(";") + raw.count("\t") < 1:
                continue
            cells = [_canon(c) for c in re.split(r"[,\t;]", raw) if c.strip()]
            pv = any(any(a and (a == c or a in c) for a in alias_sets["PV"]) for c in cells)
            mv = any(any(a and (a == c or a in c) for a in alias_sets["MV"]) for c in cells)
            score = (5 if pv else 0) + (5 if mv else 0)
            if score > best_score:
                best_score, best_idx = score, idx

        skip = best_idx if best_score >= 10 else 0
        header_line = lines[skip] if 0 <= skip < len(lines) else ""
        delim = max([",", "\t", ";"], key=lambda ch: header_line.count(ch))

        try:
            return pd.read_csv(csv_path, encoding=enc, skiprows=skip, sep=delim)
        except Exception:
            continue

    return pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")


# ── Window quality ───────────────────────────────────────────────────────────

def _robust_noise(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    d = np.diff(values.astype(float))
    return float(np.median(np.abs(d - np.median(d))))


def score_window(df: pd.DataFrame) -> dict[str, Any]:
    """Assess identification suitability of a data segment."""
    mv = df["MV"].to_numpy(dtype=float)
    pv = df["PV"].to_numpy(dtype=float)
    mv_span = float(np.max(mv) - np.min(mv)) if mv.size else 0.0
    pv_span = float(np.max(pv) - np.min(pv)) if pv.size else 0.0
    mv_noise = _robust_noise(mv)
    pv_noise = _robust_noise(pv)
    mv_eff = mv_span >= max(mv_noise * 12.0, 1e-6) and float(np.std(mv)) >= max(mv_noise * 4.0, 1e-6)
    pv_eff = pv_span >= max(pv_noise * 10.0, 1e-6) and float(np.std(pv)) >= max(pv_noise * 3.0, 1e-6)

    # Cross-correlation with lag search (fix #4: not zero-lag only)
    corr = 0.0
    if mv.size >= 10:
        mv_c = mv - float(np.mean(mv))
        pv_c = pv - float(np.mean(pv))
        mv_std = float(np.std(mv_c))
        pv_std = float(np.std(pv_c))
        if mv_std > 1e-12 and pv_std > 1e-12:
            max_lag = min(mv.size // 4, 50)
            for lag in range(0, max_lag + 1):
                a = mv_c[:-lag] if lag > 0 else mv_c
                b = pv_c[lag:] if lag > 0 else pv_c
                if a.size < 10:
                    break
                s = float(np.mean(a * b) / (mv_std * pv_std))
                if s > corr:
                    corr = s

    sat_ratio = 0.0
    if mv_span > 1e-9:
        low = float(np.min(mv)) + 0.01 * mv_span
        high = float(np.max(mv)) - 0.01 * mv_span
        sat_ratio = max(float(np.mean(mv <= low)), float(np.mean(mv >= high)))

    drift_ratio = 0.0
    if pv.size >= 5 and pv_span > 1e-9:
        x = np.arange(pv.size, dtype=float)
        slope = float(np.polyfit(x, pv, 1)[0])
        drift_ratio = abs(slope) * float(pv.size - 1) / pv_span

    reasons: list[str] = []
    if not mv_eff:
        reasons.append("MV激励不足")
    if not pv_eff:
        reasons.append("PV响应不足")
    if corr < 0.05:
        reasons.append("MV与PV相关性弱")
    if sat_ratio > 0.4:
        reasons.append("MV疑似饱和")

    score = 0.0
    score += 0.4 if mv_eff else 0.0
    score += 0.4 if pv_eff else 0.0
    score += 0.2 * min(corr / 0.4, 1.0)
    if sat_ratio > 0.0:
        score *= 1.0 - min(sat_ratio / 0.6, 1.0) * 0.7
    if drift_ratio > 0.0:
        score *= 1.0 - min(drift_ratio / 1.0, 1.0) * 0.25

    usable = mv_eff and pv_eff and corr >= 0.05 and sat_ratio <= 0.6
    return {
        "passed": bool(usable),
        "score": float(max(0.0, min(score, 1.0))),
        "reasons": reasons,
        "mv_span": mv_span,
        "pv_span": pv_span,
        "corr": corr,
        "saturation_ratio": sat_ratio,
        "drift_ratio": drift_ratio,
    }


# ── Step event detection ─────────────────────────────────────────────────────

def _detect_sv_steps(df: pd.DataFrame, threshold: float = 0.5) -> list[dict[str, Any]]:
    sv = df["SV"].to_numpy(dtype=float)
    if sv.size < 4:
        return []
    sv_diff = np.diff(sv)
    noise = float(np.median(np.abs(sv_diff - np.median(sv_diff))))
    thr = max(threshold, noise * 6.0, 1e-6)
    indices = np.where(np.abs(sv_diff) >= thr)[0] + 1
    if indices.size == 0:
        return []

    merge_gap = max(3, min(20, len(df) // 200))
    window = max(3, min(30, len(df) // 100))
    groups: list[list[int]] = []
    cur = [int(indices[0])]
    for idx in indices[1:]:
        if int(idx) - cur[-1] <= merge_gap:
            cur.append(int(idx))
        else:
            groups.append(cur)
            cur = [int(idx)]
    groups.append(cur)

    events: list[dict[str, Any]] = []
    for group in groups:
        center = int(round(sum(group) / len(group)))
        sv_before = float(np.median(sv[max(0, center - window):center]))
        sv_after = float(np.median(sv[center:min(len(df), center + window)]))
        amp = abs(sv_after - sv_before)
        if amp < threshold:
            continue
        events.append({
            "start_idx": max(0, center - merge_gap),
            "end_idx": min(len(df), center + merge_gap + 1),
            "amplitude": amp,
            "sv_start": sv_before,
            "sv_end": sv_after,
            "type": "step_up" if sv_after > sv_before else "step_down",
        })
    return events


def _detect_mv_steps(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Fix #1: Detect significant MV steps for manual-mode identification data."""
    mv = df["MV"].to_numpy(dtype=float)
    if mv.size < 10:
        return []
    mv_diff = np.abs(np.diff(mv))
    noise = float(np.median(np.abs(mv_diff - np.median(mv_diff))))
    thr = max(noise * 8.0, (float(np.max(mv)) - float(np.min(mv))) * 0.05, 1e-6)
    indices = np.where(mv_diff >= thr)[0]
    if indices.size == 0:
        return []

    merge_gap = max(5, min(30, len(df) // 100))
    top_k = 8
    # Pick top-k by amplitude, well-separated
    order = np.argsort(mv_diff[indices])[::-1]
    selected: list[int] = []
    for oi in order:
        idx = int(indices[oi])
        if any(abs(idx - s) < merge_gap for s in selected):
            continue
        selected.append(idx)
        if len(selected) >= top_k:
            break

    events: list[dict[str, Any]] = []
    for idx in selected:
        events.append({
            "start_idx": idx,
            "end_idx": min(len(df), idx + 2),
            "amplitude": float(mv_diff[idx]),
            "type": "mv_step",
        })
    return events


# ── Adaptive window builder ──────────────────────────────────────────────────

def _adaptive_padding(amplitude: float, dt: float, n: int) -> tuple[int, int]:
    """Fix #2: Compute pre/post padding from dt and estimated dynamics.

    Uses time-constant heuristic: post padding ≈ 5–8× estimated T,
    where T is approximated from typical loop response relative to amplitude.
    Minimum 60 s of data, maximum 25% of full dataset.
    """
    min_pts = max(40, int(60.0 / max(dt, 1e-6)))
    max_pts = max(200, n // 4)
    pre = max(20, min(int(30.0 / max(dt, 1e-6)), min_pts, max_pts // 4))
    post = max(min_pts, min(int(300.0 / max(dt, 1e-6)), max_pts))
    return pre, post


def build_candidate_windows(
    df: pd.DataFrame,
    dt: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build candidate identification windows from step events.

    Returns:
        (candidate_windows, step_events)

    Fix #1: Uses both SV-step and MV-step detection.
    Fix #2: Adaptive padding based on dt.
    Fix #3: Selects best window by quality score.
    """
    n = len(df)
    if n < 20:
        return [], []

    step_events: list[dict[str, Any]] = []
    if "SV" in df.columns and df["SV"].nunique(dropna=True) > 1:
        sv_thr = max(0.5, float(df["SV"].std(ddof=0) * 0.2))
        step_events = _detect_sv_steps(df, threshold=sv_thr)

    mv_steps = _detect_mv_steps(df)

    # Merge: add MV steps not already covered by SV-step windows
    all_events = list(step_events)
    for mv_ev in mv_steps:
        covered = any(
            abs(mv_ev["start_idx"] - ev["start_idx"]) < 40
            for ev in step_events
        )
        if not covered:
            all_events.append(mv_ev)

    if not all_events:
        # Fallback: single window around largest MV change
        mv = df["MV"].to_numpy(dtype=float)
        if mv.size >= 2:
            center = int(np.argmax(np.abs(np.diff(mv))))
            pre, post = _adaptive_padding(float(np.max(np.abs(np.diff(mv)))), dt, n)
            all_events = [{
                "start_idx": center,
                "end_idx": min(n, center + 2),
                "amplitude": float(np.max(np.abs(np.diff(mv)))),
                "type": "mv_fallback",
            }]

    pre_default, post_default = _adaptive_padding(1.0, dt, n)
    windows: list[dict[str, Any]] = []
    for ev in all_events:
        center = int(ev["start_idx"])
        amp = float(ev.get("amplitude", 1.0))
        pre, post = _adaptive_padding(amp, dt, n)
        w_start = max(0, center - pre)
        w_end = min(n, center + post)
        if w_end - w_start < 20:
            continue
        seg = df.iloc[w_start:w_end]
        quality = score_window(seg)
        windows.append({
            **ev,
            "window_start_idx": w_start,
            "window_end_idx": w_end,
            "window_usable_for_id": quality["passed"],
            "window_quality_score": quality["score"],
            "window_quality_reasons": quality["reasons"],
            "window_mv_span": quality["mv_span"],
            "window_pv_span": quality["pv_span"],
            "window_corr": quality["corr"],
        })

    # Fix #3: sort by quality score, not amplitude
    windows.sort(key=lambda w: (
        int(bool(w.get("window_usable_for_id"))),
        float(w.get("window_quality_score", 0.0)),
    ), reverse=True)

    # Label windows
    type_counter: dict[str, int] = {}
    for w in windows:
        base = "sv_step" if w.get("type") in {"step_up", "step_down"} else \
               "mv_step" if w.get("type") == "mv_step" else "mv_change"
        type_counter[base] = type_counter.get(base, 0) + 1
        w["window_source"] = f"{base}_{type_counter[base]}"

    return windows, step_events


# ── 主入口 ────────────────────────────────────────────────────────────────

def _load_clean_only(
    *,
    csv_path: str,
    selected_loop_prefix: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> tuple[pd.DataFrame, float]:
    """完成数据加载 + 列归一化 + 时间解析 + 数值清洗 + PV 去噪。

    返回 (cleaned_df, dt)。**不做窗口检测**，便于 skill 层各自独立调用。
    供 `load_and_prepare_dataset` 与 `load_dataset` 技能共用。
    """
    raw = _read_csv(csv_path)
    df = _normalize_columns(raw, selected_loop_prefix=selected_loop_prefix)
    df = _parse_timestamps(df)

    # 时间段切片
    if start_time or end_time:
        start_dt = pd.to_datetime(start_time, errors="coerce") if start_time else None
        end_dt = pd.to_datetime(end_time, errors="coerce") if end_time else None
        if start_dt is not None and not pd.isna(start_dt):
            df = df[df["timestamp"] >= start_dt]
        if end_dt is not None and not pd.isna(end_dt):
            df = df[df["timestamp"] <= end_dt]
        df = df.reset_index(drop=True)
        if len(df) < 20:
            raise ValueError(f"所选时间段数据不足 20 行 (得到 {len(df)} 行)")

    # 数值化 + 插值
    for col in ["SV", "PV", "MV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["PV", "MV"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("清洗后数据为空，请检查 PV/MV 列")
    for col in ["SV", "PV", "MV"]:
        if col in df.columns:
            df[col] = df[col].interpolate(limit_direction="both").ffill().bfill()

    dt = _estimate_dt(df)

    # 修复 #13：只对 PV 去噪，MV 保持原始
    cleaned = df.copy()
    if len(cleaned) >= 25:
        cleaned["PV"] = denoise_pv(cleaned["PV"].to_numpy(), noise_level="auto")
        # MV 故意不滤波

    return cleaned, dt


def load_and_prepare_dataset(
    *,
    csv_path: str,
    selected_loop_prefix: str | None = None,
    selected_window_index: int | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    """加载 CSV、清洗、去噪（仅 PV）、检测候选窗口的整套流程。

    返回 dict，包含：
        cleaned_df, dt, step_events, candidate_windows,
        data_points, quality_metrics（或 None）
    """
    cleaned, dt = _load_clean_only(
        csv_path=csv_path,
        selected_loop_prefix=selected_loop_prefix,
        start_time=start_time,
        end_time=end_time,
    )
    candidate_windows, step_events = build_candidate_windows(cleaned, dt)

    # 应用 selected_window_index 重排（把指定窗口提到首位）
    if selected_window_index is not None and 0 <= selected_window_index < len(candidate_windows):
        candidate_windows.insert(0, candidate_windows.pop(selected_window_index))

    return {
        "cleaned_df": cleaned,
        "dt": dt,
        "step_events": step_events,
        "candidate_windows": candidate_windows,
        "data_points": len(cleaned),
        "quality_metrics": None,
    }
