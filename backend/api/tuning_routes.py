"""Tuning workflow API endpoints."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from core.pipeline.runner import run_tuning_pipeline
from core.session_log import record_stream
from models import TuningRequest

router = APIRouter(tags=["tuning"])


async def _sse_generator(request: TuningRequest, csv_path: str, csv_name: str):
    """Convert pipeline events to SSE format and record to session log."""
    inner = run_tuning_pipeline(
        csv_path=csv_path,
        loop_type=request.loop_type,
        loop_name=request.loop_name,
        selected_loop_prefix=request.selected_loop_prefix,
        selected_window_index=request.selected_window_index,
        plant_type=request.plant_type,
        scenario=request.scenario,
        control_object=request.control_object,
        use_llm_advisor=request.use_llm_advisor,
    )
    meta_init = {
        "csv_name": csv_name,
        "loop_type": request.loop_type,
        "loop_name": request.loop_name,
        "use_llm_advisor": request.use_llm_advisor,
        "selected_window_index": request.selected_window_index,
    }
    async for event in record_stream(kind="tune", meta_init=meta_init, gen=inner):
        yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
    yield "data: {\"type\": \"done\"}\n\n"


@router.post("/tune/stream")
async def tune_stream(
    file: UploadFile = File(...),
    loop_type: str = Form("flow"),
    loop_name: str = Form(""),
    plant_type: str = Form(""),
    scenario: str = Form(""),
    control_object: str = Form(""),
    selected_loop_prefix: str | None = Form(None),
    selected_window_index: int | None = Form(None),
    use_llm_advisor: bool = Form(True),
):
    """Upload CSV and run tuning pipeline with SSE streaming."""
    import tempfile
    import shutil

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        csv_path = tmp.name

    request = TuningRequest(
        csv_path=csv_path,
        loop_type=loop_type,
        loop_name=loop_name,
        plant_type=plant_type,
        scenario=scenario,
        control_object=control_object,
        selected_loop_prefix=selected_loop_prefix,
        selected_window_index=selected_window_index,
        use_llm_advisor=use_llm_advisor,
    )

    return StreamingResponse(
        _sse_generator(request, csv_path, file.filename or "uploaded.csv"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/tune/run")
async def tune_run(request: TuningRequest) -> dict[str, Any]:
    """Run tuning pipeline synchronously (blocking), return final result."""
    result: dict[str, Any] = {}
    async for event in run_tuning_pipeline(
        csv_path=request.csv_path,
        loop_type=request.loop_type,
        loop_name=request.loop_name,
        selected_loop_prefix=request.selected_loop_prefix,
        selected_window_index=request.selected_window_index,
        plant_type=request.plant_type,
        scenario=request.scenario,
        control_object=request.control_object,
        use_llm_advisor=request.use_llm_advisor,
    ):
        if event.get("type") == "result":
            result = event.get("data", {})
        elif event.get("type") == "error":
            return {"error": event.get("message", "Unknown error")}
    return result
