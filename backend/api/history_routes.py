"""Historical loop data API endpoints."""
from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from core.history.store import (
    assess_loop,
    get_loop_features,
    get_loop_monitoring,
    get_loop,
    import_history_file,
    list_loop_windows,
    list_loops,
    load_loop_series,
)
from core.pipeline.runner import run_tuning_pipeline
from core.session_log import record_stream
from models import TuningRequest

router = APIRouter(tags=["history"])


def _save_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


@router.post("/history/import")
async def import_history(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Import one or more historical loop files into the offline repository."""
    dataset_id = uuid.uuid4().hex[:12]
    imported: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for file in files:
        tmp_path = _save_upload(file)
        try:
            imported.extend(import_history_file(tmp_path, file.filename or "uploaded", dataset_id))
        except Exception as exc:
            errors.append({"filename": file.filename or "uploaded", "error": str(exc)})

    return {
        "dataset_id": dataset_id,
        "imported_count": len(imported),
        "loops": imported,
        "errors": errors,
    }


@router.get("/history/loops")
def history_loops() -> dict[str, Any]:
    loops = list_loops()
    return {"total": len(loops), "items": loops}


@router.get("/history/loops/{loop_id}")
def history_loop(loop_id: str) -> dict[str, Any]:
    loop = get_loop(loop_id)
    if not loop:
        raise HTTPException(status_code=404, detail="loop_id not found")
    return loop


@router.get("/history/loops/{loop_id}/series")
def history_loop_series(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
    max_points: int = 4000,
) -> dict[str, Any]:
    return load_loop_series(
        loop_id,
        start_time=start_time,
        end_time=end_time,
        max_points=max_points,
    )


@router.get("/history/loops/{loop_id}/features")
def history_loop_features(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    result = get_loop_features(loop_id, start_time=start_time, end_time=end_time)
    if result.get("error") == "loop_id not found":
        raise HTTPException(status_code=404, detail="loop_id not found")
    return result


@router.get("/history/loops/{loop_id}/monitoring")
def history_loop_monitoring(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    result = get_loop_monitoring(loop_id, start_time=start_time, end_time=end_time)
    if result.get("error") == "loop_id not found":
        raise HTTPException(status_code=404, detail="loop_id not found")
    return result


@router.get("/history/loops/{loop_id}/assessment")
def history_loop_assessment(loop_id: str) -> dict[str, Any]:
    result = assess_loop(loop_id)
    if result.get("error") == "loop_id not found":
        raise HTTPException(status_code=404, detail="loop_id not found")
    return result


@router.get("/history/loops/{loop_id}/windows")
def history_loop_windows(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    result = list_loop_windows(loop_id, start_time=start_time, end_time=end_time)
    if result.get("error") == "loop_id not found":
        raise HTTPException(status_code=404, detail="loop_id not found")
    return result


def _tuning_blocked_by_assessment(assessment: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    readiness = assessment.get("tuning_readiness") or {}
    summary = assessment.get("summary") or {}
    decision = readiness.get("decision") or summary.get("decision")
    gate_checks = readiness.get("gate_checks") or []
    blocking_reasons = readiness.get("blocking_reasons") or []
    hard_failed_checks = [
        item for item in gate_checks
        if not item.get("passed") and str(item.get("severity") or "").lower() in {"critical", "high", "error", "blocked"}
    ]
    blocked = decision == "blocked" or bool(hard_failed_checks)
    return blocked, {
        "decision": decision,
        "decision_text": summary.get("decision_text"),
        "recommended_next_action": summary.get("recommended_next_action"),
        "recommended_next_action_text": summary.get("recommended_next_action_text"),
        "blocking_reasons": blocking_reasons,
        "failed_checks": hard_failed_checks,
    }


async def _blocked_history_tune_sse(loop_id: str, gate: dict[str, Any]):
    payload = {
        "type": "error",
        "stage": "tuning_gate",
        "message": "当前回路未通过整定准入校验，已阻止发起整定任务",
        "loop_id": loop_id,
        "data": gate,
    }
    yield f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
    yield "data: {\"type\": \"done\"}\n\n"


async def _history_tune_sse(request: TuningRequest, csv_path: str, loop_id: str):
    meta_init = {
        "csv_name": f"history:{loop_id}",
        "loop_type": request.loop_type,
        "loop_name": request.loop_name or loop_id,
        "use_llm_advisor": request.use_llm_advisor,
        "selected_window_index": request.selected_window_index,
        "history_loop_id": loop_id,
        "stop_after": request.stop_after,
        "algorithm_filter": request.algorithm_filter,
        "ontology_context_present": bool(request.ontology_context),
        "start_time": request.start_time,
        "end_time": request.end_time,
    }
    inner = run_tuning_pipeline(
        csv_path=csv_path,
        loop_type=request.loop_type,
        loop_name=request.loop_name or loop_id,
        selected_loop_prefix=request.selected_loop_prefix,
        selected_window_index=request.selected_window_index,
        plant_type=request.plant_type,
        scenario=request.scenario,
        control_object=request.control_object,
        use_llm_advisor=request.use_llm_advisor,
        stop_after=request.stop_after,  # type: ignore[arg-type]
        algorithm_filter=request.algorithm_filter,
        ontology_context=request.ontology_context,
        start_time=request.start_time or None,
        end_time=request.end_time or None,
    )
    async for event in record_stream(kind="tune", meta_init=meta_init, gen=inner):
        yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
    yield "data: {\"type\": \"done\"}\n\n"


@router.post("/history/loops/{loop_id}/tune/stream")
async def tune_history_loop_stream(
    loop_id: str,
    loop_type: str = Form(""),
    loop_name: str = Form(""),
    plant_type: str = Form(""),
    scenario: str = Form(""),
    control_object: str = Form(""),
    selected_window_index: int | None = Form(None),
    use_llm_advisor: bool = Form(True),
    stop_after: str | None = Form(None),
    # 逗号分隔的算法白名单，例如 "sv_step,mv_step"
    algorithm_filter: str | None = Form(None),
    ontology_context: str | None = Form(None),
    start_time: str | None = Form(None),
    end_time: str | None = Form(None),
):
    loop = get_loop(loop_id)
    if not loop:
        raise HTTPException(status_code=404, detail="loop_id not found")

    assessment = assess_loop(loop_id)
    blocked, gate = _tuning_blocked_by_assessment(assessment)
    if blocked:
        return StreamingResponse(
            _blocked_history_tune_sse(loop_id, gate),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    algorithm_filter_list: list[str] | None = None
    if algorithm_filter:
        algorithm_filter_list = [s.strip() for s in algorithm_filter.split(",") if s.strip()]
    request = TuningRequest(
        csv_path=str(loop["csv_path"]),
        loop_type=loop_type or str(loop.get("loop_type") or "flow"),
        loop_name=loop_name or loop_id,
        plant_type=plant_type,
        scenario=scenario,
        control_object=control_object,
        selected_loop_prefix=None,
        selected_window_index=selected_window_index,
        use_llm_advisor=use_llm_advisor,
        stop_after=stop_after,
        algorithm_filter=algorithm_filter_list,
        ontology_context=ontology_context,
        start_time=start_time or "",
        end_time=end_time or "",
    )
    return StreamingResponse(
        _history_tune_sse(request, str(loop["csv_path"]), loop_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
