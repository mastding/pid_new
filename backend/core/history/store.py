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

import numpy as np
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
from core.shared.loop_features import extract_loop_features
from core.skills.assessment.assess_loop_assessment_skill import assess_loop_assessment_from_features
from core.skills.monitoring.assess_loop_monitoring_skill import assess_loop_monitoring_from_features
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


_PV_LSL_COLUMNS = ("PV_LSL", "PV_LL", "PV_LOW", "PV_LOWER_LIMIT", "LSL")
_PV_USL_COLUMNS = ("PV_USL", "PV_HH", "PV_HIGH", "PV_UPPER_LIMIT", "USL")
_PV_SPEC_COLUMNS = _PV_LSL_COLUMNS + _PV_USL_COLUMNS


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
        for col in ["PV", "MV", "SV", *_PV_SPEC_COLUMNS]:
            if col in normalized.columns:
                normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
        normalized = normalized.dropna(subset=["PV", "MV"]).reset_index(drop=True)
        keep_cols = [c for c in ["timestamp", "SV", "PV", "MV", *_PV_SPEC_COLUMNS] if c in normalized.columns]
        normalized = normalized[keep_cols]

        csv_path = history_root() / "normalized" / f"{_safe_name(loop_id)}.csv"
        normalized.to_csv(csv_path, index=False, encoding="utf-8-sig")

        loop_type = infer_loop_type(loop_id)
        dt = _estimate_dt(normalized)
        stats = _series_stats(normalized, dt)
        # 不再在导入阶段预算候选窗口。按需触发：用户进入"窗口候选"页面点击
        # 「开始本体驱动窗口评审」或调用 `list_loop_windows` 时才计算。

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


# 不再在数据导入时预算的"窗口检测产物"字段；老的 index.json 里可能残留，需在
# 返回时剥掉，避免前端把它当作"已经选好窗口"。窗口检测改为按需触发（窗口候选页/
# 整定流水线）。
_LEGACY_WINDOW_FIELDS = (
    "window_count",
    "usable_window_count",
    "best_window_source",
    "best_window_score",
    "window_error",
)


def _strip_legacy_window_fields(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        return record
    return {k: v for k, v in record.items() if k not in _LEGACY_WINDOW_FIELDS}


def list_loops() -> list[dict[str, Any]]:
    index = _read_index()
    return sorted(
        (_strip_legacy_window_fields(item) for item in index.get("loops", {}).values()),
        key=lambda x: str(x.get("loop_id", "")),
    )


def get_loop(loop_id: str) -> dict[str, Any] | None:
    record = _read_index().get("loops", {}).get(loop_id)
    if record is None:
        return None
    return _strip_legacy_window_fields(record)


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
    # max_points <= 0 means "return all points"; useful for zooming into
    # historical trends where the operator explicitly requests full fidelity.
    if max_points and max_points > 0:
        max_points = int(max(100, min(max_points, 20000)))
        step = max(1, n // max_points) if n else 1
    else:
        step = 1
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


def _load_loop_snapshot(
    loop_id: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    loop = get_loop(loop_id)
    if not loop:
        return {"error": "loop_id not found"}

    try:
        df, dt = _load_clean_only(
            csv_path=str(loop["csv_path"]),
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as exc:
        return {"error": str(exc)}

    if len(df) < 2:
        return {
            "error": "selected time range has insufficient data",
            "loop_id": loop_id,
            "start_time": start_time,
            "end_time": end_time,
            "row_count": int(len(df)),
        }

    features = extract_loop_features(
        df,
        loop_id=loop_id,
        loop_type=str(loop.get("loop_type") or "unknown"),
        source_file=str(loop.get("source_filename") or Path(str(loop["csv_path"])).name),
        dataset_id=str(loop.get("dataset_id") or ""),
        sample_time_s=float(dt),
        loop_name=loop_id,
        tag_prefix=str(loop.get("loop_prefix") or ""),
    )
    monitoring = assess_loop_monitoring_from_features(features)
    return {
        "loop": loop,
        "df": df,
        "dt": float(dt),
        "features": features,
        "monitoring": monitoring,
        "start_time": start_time,
        "end_time": end_time,
    }


def _feature_float(features: dict[str, Any], *path: str, default: float = 0.0) -> float:
    cur: Any = features
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    try:
        if cur is None or pd.isna(cur):
            return default
        return float(cur)
    except Exception:
        return default


def _diagnostics_from_features(features: dict[str, Any]) -> dict[str, Any]:
    noise = features.get("noise_raw") or {}
    saturation = features.get("constraint_raw") or {}
    pv_range = features.get("scale_profile") or {}
    deadzone = features.get("actuator_profile") or {}
    oscillation = features.get("oscillation_raw") or {}
    disturbance = features.get("operating_condition_profile") or {}

    flags: list[dict[str, Any]] = []
    noise_ratio = _feature_float(features, "noise_raw", "pv_noise_ratio")
    if noise_ratio >= 0.08:
        flags.append({
            "type": "noise",
            "severity": "high",
            "message": "PV 噪声相对量程偏高，建议优先选择更干净的稳态片段再做辨识。",
        })
    elif noise_ratio >= 0.04:
        flags.append({
            "type": "noise",
            "severity": "medium",
            "message": "PV 噪声有一定影响，辨识前建议关注滤波和窗口质量。",
        })

    mv_saturation = _feature_float(features, "constraint_raw", "mv_saturation_ratio")
    if mv_saturation >= 0.3:
        flags.append({
            "type": "saturation",
            "severity": "high",
            "message": "MV 长时间处于约束或贴边状态，当前响应可能不能代表正常闭环特性。",
        })
    elif mv_saturation >= 0.1:
        flags.append({
            "type": "saturation",
            "severity": "medium",
            "message": "MV 存在一定贴边比例，整定前建议确认阀位或执行机构余量。",
        })

    deadband_ratio = max(
        _feature_float(features, "actuator_profile", "mv_deadband_hint_ratio"),
        _feature_float(features, "actuator_profile", "mv_deadband_lagged_ratio"),
    )
    if deadband_ratio >= 0.5:
        flags.append({
            "type": "deadzone",
            "severity": "medium",
            "message": "存在较明显死区迹象，小幅 MV 动作可能难以有效驱动 PV。",
        })

    if bool(oscillation.get("detected", False)):
        flags.append({
            "type": "oscillation",
            "severity": "medium",
            "message": "PV 频谱存在主周期成分，需区分自激振荡和周期性扰动。",
        })

    condition = str(disturbance.get("condition") or disturbance.get("primary_condition") or "")
    if "load_change" in condition:
        flags.append({
            "type": "operating_condition",
            "severity": "medium",
            "message": "当前片段包含负荷变化或过渡工况，整定建议优先选择稳定片段。",
        })

    return {
        "pv_range": pv_range,
        "noise": noise,
        "saturation": saturation,
        "deadzone": deadzone,
        "oscillation": oscillation,
        "disturbance": disturbance,
        "flags": flags,
    }


def _assessment_from_snapshot(snapshot: dict[str, Any], loop_id: str) -> dict[str, Any]:
    if snapshot.get("error"):
        return snapshot
    loop = snapshot["loop"]
    features = snapshot["features"]
    monitoring = snapshot["monitoring"]
    assessment = assess_loop_assessment_from_features(
        features,
        monitoring,
        # 准入评估只使用画像指标，不提前触发窗口候选计算。
        window_summary={},
        diagnostics=_diagnostics_from_features(features),
    )
    return {
        "loop_id": loop_id,
        "loop_type": loop.get("loop_type", "unknown"),
        **assessment,
    }


def get_loop_features(
    loop_id: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    """Return raw observable LoopFeatures for an imported historical loop."""
    snapshot = _load_loop_snapshot(loop_id, start_time=start_time, end_time=end_time)
    if snapshot.get("error"):
        return snapshot
    return snapshot["features"]


def get_loop_monitoring(
    loop_id: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    """Return monitoring snapshot derived from raw LoopFeatures."""
    snapshot = _load_loop_snapshot(loop_id, start_time=start_time, end_time=end_time)
    if snapshot.get("error"):
        return snapshot
    return {
        "loop_id": loop_id,
        "features": snapshot["features"],
        "monitoring": snapshot["monitoring"],
    }


def _score_to_level(score: float) -> str:
    if score >= 0.8:
        return "excellent"
    if score >= 0.6:
        return "good"
    if score >= 0.4:
        return "fair"
    return "weak"


def assess_loop(
    loop_id: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    """Assess imported loop history from the shared profile snapshot."""
    snapshot = _load_loop_snapshot(loop_id, start_time=start_time, end_time=end_time)
    return _assessment_from_snapshot(snapshot, loop_id)


def get_loop_profile_bundle(
    loop_id: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    """Return features, monitoring and assessment from one shared history snapshot."""
    snapshot = _load_loop_snapshot(loop_id, start_time=start_time, end_time=end_time)
    if snapshot.get("error"):
        return snapshot
    assessment = _assessment_from_snapshot(snapshot, loop_id)
    return {
        "loop_id": loop_id,
        "loop_type": snapshot["loop"].get("loop_type", "unknown"),
        "start_time": start_time,
        "end_time": end_time,
        "features": snapshot["features"],
        "monitoring": {
            "loop_id": loop_id,
            "features": snapshot["features"],
            "monitoring": snapshot["monitoring"],
        },
        "assessment": assessment,
    }


def compute_loop_harris(
    loop_id: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    error_basis: str = "auto",
    force_deadtime_samples: int | None = None,
    ar_order_override: int | None = None,
) -> dict[str, Any]:
    """Run the formal Harris closed-loop skill on an imported history loop."""
    snapshot = _load_loop_snapshot(loop_id, start_time=start_time, end_time=end_time)
    if snapshot.get("error"):
        return snapshot

    # Lazy import keeps normal history endpoints from paying external skill discovery cost.
    from core.skills import LoopContext, registry

    skill_name = "compute_harris_closed_loop"
    if skill_name not in registry.names():
        return {
            "loop_id": loop_id,
            "error": f"skill not registered: {skill_name}",
        }

    loop = snapshot["loop"]
    ctx = LoopContext(
        csv_path=str(loop["csv_path"]),
        loop_prefix=str(loop.get("loop_prefix") or ""),
        loop_type=str(loop.get("loop_type") or "unknown"),
    )
    ctx.cleaned_df = snapshot["df"]
    ctx.dt = float(snapshot["dt"])
    ctx.data_profile = snapshot["features"]

    args: dict[str, Any] = {"error_basis": error_basis or "auto"}
    if force_deadtime_samples is not None:
        args["force_deadtime_samples"] = int(force_deadtime_samples)
    if ar_order_override is not None:
        args["ar_order_override"] = int(ar_order_override)

    result = registry.invoke(skill_name, args, ctx)
    return {
        "loop_id": loop_id,
        "loop_type": loop.get("loop_type", "unknown"),
        "start_time": start_time,
        "end_time": end_time,
        "success": bool(result.success),
        "harris": result.data,
        "warnings": result.warnings,
        "reasoning": result.reasoning,
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not np.isfinite(number):
        return None
    return number


def _round_number(value: Any, digits: int = 6) -> float | None:
    number = _finite_number(value)
    return round(number, digits) if number is not None else None


def _series_spec_limit(df: pd.DataFrame, names: tuple[str, ...]) -> tuple[float | None, str | None]:
    for name in names:
        if name not in df.columns:
            continue
        values = pd.to_numeric(df[name], errors="coerce").dropna()
        if values.empty:
            continue
        return float(values.median()), name
    return None, None


def _cpk_level(cpk: float | None) -> str:
    if cpk is None:
        return "unavailable"
    if cpk >= 1.67:
        return "excellent"
    if cpk >= 1.33:
        return "good"
    if cpk >= 1.0:
        return "fair"
    return "poor"


def _cpk_reason_label(reason: str) -> str:
    return {
        "ok": "计算完成",
        "insufficient_pv_samples": "PV 有效样本不足",
        "missing_pv_spec_limits": "缺少 PV 规格上下限",
        "invalid_pv_spec_limits": "PV 规格上下限无效",
        "zero_pv_variance": "PV 波动接近 0",
    }.get(reason, reason or "原因未知")


def compute_loop_cpk(
    loop_id: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    spec_limits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute formal PV process capability CPK for an imported history loop."""
    loop = get_loop(loop_id)
    if not loop:
        return {"error": "loop_id not found"}

    try:
        df, dt = _load_clean_only(
            csv_path=str(loop["csv_path"]),
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as exc:
        return {"error": str(exc)}

    if len(df) < 2:
        return {
            "error": "selected time range has insufficient data",
            "loop_id": loop_id,
            "start_time": start_time,
            "end_time": end_time,
            "row_count": int(len(df)),
        }

    dt = float(dt)
    pv = pd.to_numeric(df.get("PV"), errors="coerce").dropna() if "PV" in df.columns else pd.Series(dtype=float)

    n_samples = int(len(pv))
    duration_s = float((n_samples - 1) * dt) if n_samples > 1 else 0.0
    data_window = {
        "n_samples": n_samples,
        "duration_s": _round_number(duration_s, 3),
        "sample_time_s": _round_number(dt, 6),
    }
    if "timestamp" in df.columns and len(df):
        data_window["time_start"] = _iso(df["timestamp"].iloc[0])
        data_window["time_end"] = _iso(df["timestamp"].iloc[-1])

    spec_limits = spec_limits or {}
    ontology_lsl = _finite_number(spec_limits.get("lsl"))
    ontology_usl = _finite_number(spec_limits.get("usl"))
    ontology_source = str(spec_limits.get("source") or "ontology")
    if ontology_lsl is not None and ontology_usl is not None:
        lsl, usl = ontology_lsl, ontology_usl
        lsl_column, usl_column = "ontology.lsl", "ontology.usl"
        limits_source = ontology_source
    else:
        lsl, lsl_column = _series_spec_limit(df, _PV_LSL_COLUMNS)
        usl, usl_column = _series_spec_limit(df, _PV_USL_COLUMNS)
        limits_source = "pv_spec_columns" if lsl_column and usl_column else "missing"
    mean = float(pv.mean()) if n_samples else None
    std = float(pv.std(ddof=1)) if n_samples > 1 else None

    warnings: list[str] = []
    cpl: float | None = None
    cpu: float | None = None
    cpk: float | None = None
    status = "ok"
    reason = "ok"

    if n_samples < 2 or mean is None or std is None:
        status = "unavailable"
        reason = "insufficient_pv_samples"
        warnings.append("PV 有效样本不足，无法计算样本均值和样本标准差。")
    elif lsl is None or usl is None:
        status = "unavailable"
        reason = "missing_pv_spec_limits"
        warnings.append("缺少 PV 规格上下限，无法计算正式 CPK。")
    elif usl <= lsl:
        status = "unavailable"
        reason = "invalid_pv_spec_limits"
        warnings.append("PV 规格上限必须大于规格下限。")
    elif std <= 1e-12:
        status = "unavailable"
        reason = "zero_pv_variance"
        warnings.append("PV 波动接近 0，样本标准差不可用于 CPK 计算。")
    else:
        cpl = (mean - lsl) / (3.0 * std)
        cpu = (usl - mean) / (3.0 * std)
        cpk = min(cpl, cpu)

    level = _cpk_level(cpk)
    recommend_action = (
        "规格能力充足，可结合 Harris 指标继续判断控制性能。"
        if level in {"excellent", "good"}
        else "规格能力不足或无法计算，建议先确认 PV 规格上下限和数据窗口。"
    )
    if level == "fair":
        recommend_action = "规格能力临界，建议复核设定值、波动来源和上下限裕量。"
    if level == "poor":
        recommend_action = "CPK 低于 1.0，当前窗口对规格边界的过程能力不足。"

    reasoning = (
        "CPK 只使用 PV 序列、PV 规格上下限和样本标准差计算。"
        f" 当前窗口有效 PV 样本 {n_samples} 个，均值 {_round_number(mean, 6)}，"
        f"样本标准差 {_round_number(std, 6)}，"
        f"LSL {_round_number(lsl, 6) if lsl is not None else '-'}，"
        f"USL {_round_number(usl, 6) if usl is not None else '-'}。"
    )
    if cpk is None:
        reasoning += f" 因{_cpk_reason_label(reason)}，未输出正式 CPK。"
    else:
        reasoning += (
            f" CPL=(均值-LSL)/(3σ)={_round_number(cpl, 6)}，"
            f"CPU=(USL-均值)/(3σ)={_round_number(cpu, 6)}，"
            f"CPK=min(CPL,CPU)={_round_number(cpk, 6)}。"
        )

    return {
        "loop_id": loop_id,
        "loop_type": loop.get("loop_type", "unknown"),
        "start_time": start_time,
        "end_time": end_time,
        "success": cpk is not None,
        "cpk": {
            "value": _round_number(cpk, 6),
            "level": level,
            "status": status,
            "reason": reason,
            "cpl": _round_number(cpl, 6),
            "cpu": _round_number(cpu, 6),
            "recommend_action": recommend_action,
        },
        "data_window": data_window,
        "limits": {
            "lsl": _round_number(lsl, 6),
            "usl": _round_number(usl, 6),
            "source": limits_source,
            "lsl_column": lsl_column,
            "usl_column": usl_column,
        },
        "statistics": {
            "mean": _round_number(mean, 6),
            "std": _round_number(std, 6),
            "min": _round_number(pv.min() if n_samples else None, 6),
            "max": _round_number(pv.max() if n_samples else None, 6),
        },
        "warnings": warnings,
        "reasoning": reasoning,
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
        "algorithm": str(window.get("window_algorithm", "")),
        "algorithm_label": str(window.get("window_algorithm_label", "")),
        "selection_basis": str(window.get("window_selection_basis", "")),
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
        "score_breakdown": window.get("window_score_breakdown", {}),
        "quality_metrics": window.get("window_quality_metrics", {}),
        "preview": preview,
    }


def list_loop_windows(
    loop_id: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    """Return candidate identification windows for an imported loop."""
    loop = get_loop(loop_id)
    if not loop:
        return {"windows": [], "error": "loop_id not found"}

    try:
        dataset = load_and_prepare_dataset(
            csv_path=str(loop["csv_path"]),
            loop_type=str(loop.get("loop_type") or ""),
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as exc:
        return {"windows": [], "error": str(exc)}

    df = dataset["cleaned_df"]
    windows = [
        _window_to_dict(window, df, index)
        for index, window in enumerate(dataset.get("candidate_windows") or [])
    ]
    algorithm_summary: dict[str, dict[str, int]] = {}
    for window in windows:
        key = window.get("algorithm") or window.get("type") or "unknown"
        item = algorithm_summary.setdefault(str(key), {"total": 0, "usable": 0})
        item["total"] += 1
        if window["usable"]:
            item["usable"] += 1
    return {
        "loop_id": loop_id,
        "loop_type": loop.get("loop_type", "unknown"),
        "dt": round(float(dataset.get("dt", 0.0) or 0.0), 6),
        "start_time": start_time,
        "end_time": end_time,
        "total": len(windows),
        "usable_count": sum(1 for window in windows if window["usable"]),
        "algorithm_summary": algorithm_summary,
        "windows": windows,
    }
