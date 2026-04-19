"""Data inspection API endpoints."""
from __future__ import annotations

import shutil
import tempfile
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile

from core.algorithms.data_analysis import (
    _detect_loops,
    _estimate_dt,
    _parse_timestamps,
    _read_csv,
    load_and_prepare_dataset,
)

router = APIRouter(tags=["data"])


def _save_upload(file: UploadFile) -> str:
    """Save uploaded file to a temp path and return it."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


def _window_to_dict(w: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Serialize a candidate window to a JSON-safe dict."""
    start, end = int(w["window_start_idx"]), int(w["window_end_idx"])
    preview_pv = df["PV"].iloc[start:end].round(4).tolist()
    preview_mv = df["MV"].iloc[start:end].round(4).tolist()
    # Down-sample preview to ≤120 points
    if len(preview_pv) > 120:
        step = len(preview_pv) // 120
        preview_pv = preview_pv[::step]
        preview_mv = preview_mv[::step]
    return {
        "index": w.get("index", 0),
        "start": start,
        "end": end,
        "n_points": end - start,
        "score": round(float(w.get("window_quality_score", 0)), 4),
        "amplitude": round(float(w.get("amplitude", 0)), 4),
        "window_usable_for_id": bool(w.get("window_usable_for_id", True)),
        "source": w.get("window_source", ""),
        "step_type": w.get("type", ""),
        "preview_pv": preview_pv,
        "preview_mv": preview_mv,
    }


@router.post("/data/inspect-loops")
async def inspect_loops(file: UploadFile = File(...)) -> dict[str, Any]:
    """Detect PID loops in uploaded CSV.

    Returns list of loop prefixes with their PV/MV/SV column names.
    If only one loop (or unnamed columns), returns a single unnamed loop.
    """
    csv_path = _save_upload(file)
    try:
        df = _read_csv(csv_path)
        df = _parse_timestamps(df)
    except Exception as exc:
        return {"loops": [], "error": str(exc)}

    loops = _detect_loops(df)
    dt = _estimate_dt(df)

    if loops:
        return {
            "loops": [
                {
                    "prefix": l["prefix"],
                    "pv_col": l.get("pv_col", ""),
                    "mv_col": l.get("mv_col", ""),
                    "sv_col": l.get("sv_col", ""),
                }
                for l in loops
            ],
            "total_rows": len(df),
            "sampling_time": round(dt, 3),
            "csv_path": csv_path,
        }

    # No structured loops found — treat as single unnamed loop
    return {
        "loops": [{"prefix": "", "pv_col": "PV", "mv_col": "MV", "sv_col": "SV"}],
        "total_rows": len(df),
        "sampling_time": round(dt, 3),
        "csv_path": csv_path,
    }


@router.post("/data/inspect-windows")
async def inspect_windows(
    file: UploadFile = File(...),
    loop_prefix: str | None = Form(None),
) -> dict[str, Any]:
    """Find candidate identification windows in CSV data.

    Returns candidate windows sorted by quality score, with PV/MV preview data.
    """
    csv_path = _save_upload(file)
    try:
        dataset = load_and_prepare_dataset(
            csv_path=csv_path,
            selected_loop_prefix=loop_prefix or None,
        )
    except ValueError as exc:
        return {"windows": [], "error": str(exc)}
    except Exception as exc:
        return {"windows": [], "error": f"数据读取失败: {exc}"}

    df = dataset["cleaned_df"]
    windows = [
        _window_to_dict({**w, "index": i}, df)
        for i, w in enumerate(dataset["candidate_windows"])
    ]

    return {
        "windows": windows,
        "total_rows": dataset["data_points"],
        "sampling_time": round(dataset["dt"], 3),
        "step_events": len(dataset.get("step_events") or []),
        "usable_count": sum(1 for w in windows if w["window_usable_for_id"]),
        "csv_path": csv_path,
    }
