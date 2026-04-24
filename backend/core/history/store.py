"""File-backed repository for imported historical loop data.

This is the first offline-history adapter. It intentionally stores normalized
CSV plus JSON metadata so a future historian/DB adapter can keep the same API
shape without forcing the UI to change.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.algorithms.data_analysis import (
    _detect_loops,
    _estimate_dt,
    _load_clean_only,
    _normalize_columns,
    _parse_timestamps,
    _read_csv,
    load_and_prepare_dataset,
)
from core.skills.data_understanding import _analyzers as analyzers


def history_root() -> Path:
    configured = os.getenv("PID_HISTORY_DATA_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "var" / "history"


def _index_path() -> Path:
    return history_root() / "index.json"


def _ensure_dirs() -> None:
    root = history_root()
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "normalized").mkdir(parents=True, exist_ok=True)


def _read_index() -> dict[str, Any]:
    path = _index_path()
    if not path.exists():
        return {"datasets": {}, "loops": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"datasets": {}, "loops": {}}


def _write_index(index: dict[str, Any]) -> None:
    _ensure_dirs()
    _index_path().write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    return safe.strip("._") or f"loop_{uuid.uuid4().hex[:8]}"


def _loop_tag(prefix: str, filename: str) -> str:
    text = prefix or Path(filename).stem
    text = text.split(".")[-1]
    match = re.search(r"(\d{3,6}_[A-Z]{1,4}_?\d+[A-Z]?)", text.upper())
    return match.group(1) if match else _safe_name(text).upper()


def infer_loop_type(loop_id: str) -> str:
    tag = loop_id.upper()
    if "_FIC" in tag or re.search(r"\bFIC", tag):
        return "flow"
    if "_TIC" in tag or re.search(r"\bTIC", tag):
        return "temperature"
    if "_PIC" in tag or re.search(r"\bPIC", tag):
        return "pressure"
    if "_LIC" in tag or re.search(r"\bLIC", tag):
        return "level"
    return "unknown"


def _iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ")
    return str(value)


def _series_stats(df: pd.DataFrame, dt: float) -> dict[str, Any]:
    timestamped = "timestamp" in df.columns
    start_time = _iso(df["timestamp"].iloc[0]) if timestamped and len(df) else None
    end_time = _iso(df["timestamp"].iloc[-1]) if timestamped and len(df) else None
    return {
        "rows": int(len(df)),
        "sampling_time": round(float(dt), 6),
        "start_time": start_time,
        "end_time": end_time,
        "pv_min": round(float(df["PV"].min()), 6) if "PV" in df else None,
        "pv_max": round(float(df["PV"].max()), 6) if "PV" in df else None,
        "mv_min": round(float(df["MV"].min()), 6) if "MV" in df else None,
        "mv_max": round(float(df["MV"].max()), 6) if "MV" in df else None,
    }


def _window_summary(csv_path: str, loop_type: str) -> dict[str, Any]:
    try:
        dataset = load_and_prepare_dataset(csv_path=csv_path, loop_type=loop_type)
        windows = dataset.get("candidate_windows") or []
        usable = [w for w in windows if w.get("window_usable_for_id")]
        best = windows[0] if windows else {}
        return {
            "window_count": len(windows),
            "usable_window_count": len(usable),
            "best_window_source": str(best.get("window_source", "")),
            "best_window_score": round(float(best.get("window_quality_score", 0.0)), 4) if best else None,
        }
    except Exception as exc:
        return {
            "window_count": 0,
            "usable_window_count": 0,
            "best_window_source": "",
            "best_window_score": None,
            "window_error": str(exc),
        }


def import_history_file(src_path: str, original_name: str, dataset_id: str) -> list[dict[str, Any]]:
    _ensure_dirs()
    index = _read_index()

    raw_src = Path(src_path)
    raw_name = f"{dataset_id}_{_safe_name(original_name)}"
    raw_dest = history_root() / "raw" / raw_name
    shutil.copyfile(raw_src, raw_dest)

    raw_df = _read_csv(str(raw_dest))
    raw_df = _parse_timestamps(raw_df)
    loops = _detect_loops(raw_df)
    if not loops:
        loops = [{"prefix": "", "pv_col": "PV", "mv_col": "MV", "sv_col": "SV"}]

    imported: list[dict[str, Any]] = []
    for loop in loops:
        prefix = str(loop.get("prefix", ""))
        base_loop_id = _loop_tag(prefix, original_name)
        loop_id = base_loop_id
        while loop_id in index["loops"]:
            loop_id = f"{base_loop_id}_{uuid.uuid4().hex[:6]}"

        normalized = _normalize_columns(raw_df.copy(), selected_loop_prefix=prefix or None)
        normalized = _parse_timestamps(normalized)
        for col in ["PV", "MV", "SV"]:
            if col in normalized.columns:
                normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
        normalized = normalized.dropna(subset=["PV", "MV"]).reset_index(drop=True)
        keep_cols = [c for c in ["timestamp", "SV", "PV", "MV"] if c in normalized.columns]
        normalized = normalized[keep_cols]

        csv_path = history_root() / "normalized" / f"{_safe_name(loop_id)}.csv"
        normalized.to_csv(csv_path, index=False, encoding="utf-8-sig")

        loop_type = infer_loop_type(loop_id)
        dt = _estimate_dt(normalized)
        stats = _series_stats(normalized, dt)
        windows = _window_summary(str(csv_path), loop_type if loop_type != "unknown" else "")

        record = {
            "loop_id": loop_id,
            "loop_prefix": prefix,
            "loop_type": loop_type,
            "dataset_id": dataset_id,
            "source_filename": original_name,
            "raw_path": str(raw_dest),
            "csv_path": str(csv_path),
            "imported_at": datetime.now().isoformat(timespec="seconds"),
            **stats,
            **windows,
        }
        index["loops"][loop_id] = record
        imported.append(record)

    dataset = index["datasets"].setdefault(dataset_id, {
        "dataset_id": dataset_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": [],
        "loop_ids": [],
    })
    dataset["files"].append(original_name)
    dataset["loop_ids"].extend([item["loop_id"] for item in imported])
    _write_index(index)
    return imported


def list_loops() -> list[dict[str, Any]]:
    index = _read_index()
    return sorted(index.get("loops", {}).values(), key=lambda x: str(x.get("loop_id", "")))


def get_loop(loop_id: str) -> dict[str, Any] | None:
    return _read_index().get("loops", {}).get(loop_id)


def load_loop_series(
    loop_id: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    max_points: int = 4000,
) -> dict[str, Any]:
    loop = get_loop(loop_id)
    if not loop:
        return {"points": [], "error": "loop_id not found"}

    try:
        df, dt = _load_clean_only(
            csv_path=str(loop["csv_path"]),
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as exc:
        return {"points": [], "error": str(exc)}

    n = int(len(df))
    max_points = int(max(100, min(max_points, 20000)))
    step = max(1, n // max_points) if n else 1
    indices = list(range(0, n, step))
    if indices and indices[-1] != n - 1:
        indices.append(n - 1)

    has_ts = "timestamp" in df.columns
    if has_ts:
        t_vals = df["timestamp"].iloc[indices].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        x_axis = "timestamp"
    else:
        t_vals = [round(float(i) * dt, 3) for i in indices]
        x_axis = "t"

    points: list[dict[str, Any]] = []
    for pos, idx in enumerate(indices):
        item = {
            "t": t_vals[pos],
            "pv": float(df["PV"].iloc[idx]),
            "mv": float(df["MV"].iloc[idx]),
            "sv": None,
        }
        if "SV" in df.columns:
            sv = df["SV"].iloc[idx]
            item["sv"] = float(sv) if not pd.isna(sv) else None
        points.append(item)

    return {
        "loop_id": loop_id,
        "loop_type": loop.get("loop_type", "unknown"),
        "x_axis": x_axis,
        "dt": round(float(dt), 6),
        "total_points": n,
        "sampled_points": len(points),
        "points": points,
    }


def _score_to_level(score: float) -> str:
    if score >= 0.8:
        return "excellent"
    if score >= 0.6:
        return "good"
    if score >= 0.4:
        return "fair"
    return "weak"


def assess_loop(loop_id: str) -> dict[str, Any]:
    """Assess imported loop history for monitoring, diagnosis and tuning readiness."""
    loop = get_loop(loop_id)
    if not loop:
        return {"error": "loop_id not found"}

    try:
        df, dt = _load_clean_only(csv_path=str(loop["csv_path"]))
        dataset = load_and_prepare_dataset(
            csv_path=str(loop["csv_path"]),
            loop_type=str(loop.get("loop_type") or ""),
        )
    except Exception as exc:
        return {"error": str(exc)}

    windows = dataset.get("candidate_windows") or []
    usable_windows = [w for w in windows if w.get("window_usable_for_id")]
    best_window = windows[0] if windows else {}

    noise = analyzers.analyze_noise(df)
    saturation = analyzers.analyze_mv_saturation(df)
    pv_range = analyzers.analyze_pv_range(df)
    deadzone = analyzers.analyze_deadzone(
        df,
        pv_noise_std=float(noise.get("pv_noise_std", 0.0) or 0.0),
        dt=float(dt),
        loop_type=str(loop.get("loop_type") or ""),
    )
    oscillation = analyzers.analyze_oscillation(df, float(dt))
    disturbance = analyzers.analyze_disturbance(df)

    missing_ratio = float(df[["PV", "MV"]].isna().mean().mean()) if len(df) else 1.0
    continuity_score = 1.0
    if "timestamp" in df.columns and len(df) > 2:
        deltas = df["timestamp"].diff().dt.total_seconds().dropna()
        if not deltas.empty and dt > 0:
            bad_gap_ratio = float((abs(deltas - dt) > max(dt * 0.5, 1e-6)).mean())
            continuity_score = max(0.0, 1.0 - bad_gap_ratio * 2.0)

    noise_level = str(noise.get("noise_level", "unknown"))
    noise_score = {"low": 1.0, "medium": 0.75, "high": 0.35}.get(noise_level, 0.55)
    sat_high = float(saturation.get("saturation_high_pct", 0.0) or 0.0)
    sat_low = float(saturation.get("saturation_low_pct", 0.0) or 0.0)
    saturation_score = max(0.0, 1.0 - max(sat_high, sat_low) / 60.0)

    data_quality_score = max(
        0.0,
        min(1.0, 0.35 * (1.0 - missing_ratio) + 0.25 * continuity_score + 0.2 * noise_score + 0.2 * saturation_score),
    )

    best_window_score = float(best_window.get("window_quality_score", 0.0) or 0.0)
    usable_ratio = len(usable_windows) / len(windows) if windows else 0.0
    identifiability_score = max(0.0, min(1.0, 0.65 * best_window_score + 0.35 * usable_ratio))

    diagnostic_flags: list[dict[str, Any]] = []
    if noise_level == "high":
        diagnostic_flags.append({"type": "noise", "severity": "high", "message": "PV 噪声偏高，辨识前建议增强滤波或选择更干净窗口。"})
    if max(sat_high, sat_low) > 30:
        diagnostic_flags.append({"type": "saturation", "severity": "high", "message": "MV 长时间贴近上下沿，闭环响应可能受阀位/工况限制。"})
    if float(deadzone.get("evidence_ratio", 0.0) or 0.0) > 0.5:
        diagnostic_flags.append({"type": "deadzone", "severity": "medium", "message": "存在较明显死区迹象，小幅 MV 动作可能难以驱动 PV。"})
    if bool(oscillation.get("detected", False)):
        diagnostic_flags.append({"type": "oscillation", "severity": "medium", "message": "PV 频谱存在主周期成分，可能有振荡或周期扰动。"})
    if not usable_windows:
        diagnostic_flags.append({"type": "identifiability", "severity": "high", "message": "未找到可用辨识窗口，建议补充激励或扩大历史时间范围。"})

    readiness_score = max(0.0, min(1.0, 0.45 * data_quality_score + 0.45 * identifiability_score + 0.1 * (1.0 if not diagnostic_flags else 0.65)))
    if not usable_windows:
        readiness_score = min(readiness_score, 0.35)

    recommendations: list[str] = []
    if data_quality_score < 0.6:
        recommendations.append("优先处理数据质量：检查缺失、采样间隔、噪声与 MV 饱和。")
    if identifiability_score < 0.6:
        recommendations.append("优先寻找更强 MV 激励窗口，或从更长历史区间重新导入。")
    if not recommendations:
        recommendations.append("该历史片段具备初步辨识条件，可进入整定任务并保留评审兜底。")

    return {
        "loop_id": loop_id,
        "loop_type": loop.get("loop_type", "unknown"),
        "data_quality": {
            "score": round(data_quality_score, 4),
            "level": _score_to_level(data_quality_score),
            "missing_ratio": round(missing_ratio, 5),
            "continuity_score": round(continuity_score, 4),
            "noise_score": round(noise_score, 4),
            "saturation_score": round(saturation_score, 4),
        },
        "identifiability": {
            "score": round(identifiability_score, 4),
            "level": _score_to_level(identifiability_score),
            "window_count": len(windows),
            "usable_window_count": len(usable_windows),
            "best_window_score": round(best_window_score, 4) if windows else None,
            "best_window_source": str(best_window.get("window_source", "")),
            "best_window_reasons": best_window.get("window_quality_reasons", []),
        },
        "diagnostics": {
            "pv_range": pv_range,
            "noise": noise,
            "saturation": saturation,
            "deadzone": deadzone,
            "oscillation": oscillation,
            "disturbance": disturbance,
            "flags": diagnostic_flags,
        },
        "readiness": {
            "score": round(readiness_score, 4),
            "level": _score_to_level(readiness_score),
            "recommendations": recommendations,
        },
    }


def _window_to_dict(window: dict[str, Any], df: pd.DataFrame, index: int) -> dict[str, Any]:
    start = int(window.get("window_start_idx", 0))
    end = int(window.get("window_end_idx", start))
    start = max(0, min(start, len(df)))
    end = max(start, min(end, len(df)))

    preview_idx = list(range(start, end))
    if len(preview_idx) > 240:
        step = max(1, len(preview_idx) // 240)
        preview_idx = preview_idx[::step]
        if preview_idx[-1] != end - 1:
            preview_idx.append(end - 1)

    has_ts = "timestamp" in df.columns
    preview: list[dict[str, Any]] = []
    for pos in preview_idx:
        t_value: Any = int(pos - start)
        if has_ts:
            t_value = df["timestamp"].iloc[pos].strftime("%Y-%m-%d %H:%M:%S")
        preview.append({
            "t": t_value,
            "pv": float(df["PV"].iloc[pos]),
            "mv": float(df["MV"].iloc[pos]),
        })

    start_time = None
    end_time = None
    if has_ts and end > start:
        start_time = df["timestamp"].iloc[start].strftime("%Y-%m-%d %H:%M:%S")
        end_time = df["timestamp"].iloc[end - 1].strftime("%Y-%m-%d %H:%M:%S")

    return {
        "index": index,
        "source": str(window.get("window_source", "")),
        "type": str(window.get("type", "")),
        "start_idx": start,
        "end_idx": end,
        "n_points": int(end - start),
        "start_time": start_time,
        "end_time": end_time,
        "usable": bool(window.get("window_usable_for_id", False)),
        "score": round(float(window.get("window_quality_score", 0.0) or 0.0), 4),
        "amplitude": round(float(window.get("amplitude", 0.0) or 0.0), 4),
        "mv_span": round(float(window.get("window_mv_span", 0.0) or 0.0), 4),
        "pv_span": round(float(window.get("window_pv_span", 0.0) or 0.0), 4),
        "corr": round(float(window.get("window_corr", 0.0) or 0.0), 4),
        "reasons": window.get("window_quality_reasons", []),
        "preview": preview,
    }


def list_loop_windows(loop_id: str) -> dict[str, Any]:
    """Return candidate identification windows for an imported loop."""
    loop = get_loop(loop_id)
    if not loop:
        return {"windows": [], "error": "loop_id not found"}

    try:
        dataset = load_and_prepare_dataset(
            csv_path=str(loop["csv_path"]),
            loop_type=str(loop.get("loop_type") or ""),
        )
    except Exception as exc:
        return {"windows": [], "error": str(exc)}

    df = dataset["cleaned_df"]
    windows = [
        _window_to_dict(window, df, index)
        for index, window in enumerate(dataset.get("candidate_windows") or [])
    ]
    return {
        "loop_id": loop_id,
        "loop_type": loop.get("loop_type", "unknown"),
        "dt": round(float(dataset.get("dt", 0.0) or 0.0), 6),
        "total": len(windows),
        "usable_count": sum(1 for window in windows if window["usable"]),
        "windows": windows,
    }
