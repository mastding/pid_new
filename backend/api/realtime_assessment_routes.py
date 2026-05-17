"""Realtime assessment APIs backed by SQLite snapshots."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.realtime import (
    PrepareAutoTuningTaskRequest,
    RealtimeAssessmentRequest,
    realtime_assessment_service,
)

router = APIRouter(tags=["realtime-assessments"])


class RunRealtimeAssessmentBody(BaseModel):
    loop_ids: list[str] | None = Field(None, description="Loop ids to assess. Empty means all loops in asset.")
    asset_id: str | None = Field(None, description="Asset filter, for example 5203.")
    time_range: str = Field("8h", description="Relative window, e.g. 8h, 24h, 7d.")
    start_time: str | None = None
    end_time: str | None = None
    force_refresh: bool = False
    include_formal_metrics: bool = True
    auto_create_tasks: bool = False


class CreateAutoTuningTaskBody(BaseModel):
    confirm: bool = False
    trigger_mode: str = "manual"
    reason: str | None = None


class PrepareAutoTuningTaskBody(BaseModel):
    confirm: bool = False
    use_llm_advisor: bool = True
    selected_window_index: int | None = None


@router.post("/realtime-assessments/run")
async def run_realtime_assessment(body: RunRealtimeAssessmentBody) -> dict[str, Any]:
    request = RealtimeAssessmentRequest(
        loop_ids=body.loop_ids,
        asset_id=body.asset_id,
        time_range=body.time_range,
        start_time=body.start_time,
        end_time=body.end_time,
        force_refresh=body.force_refresh,
        include_formal_metrics=body.include_formal_metrics,
        auto_create_tasks=body.auto_create_tasks,
    )
    return await realtime_assessment_service.run(request)


@router.get("/realtime-assessments/latest")
def list_latest_realtime_assessments(
    asset_id: str | None = None,
    loop_id: str | None = None,
    risk_level: str | None = None,
    decision: str | None = None,
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    return realtime_assessment_service.latest(
        asset_id=asset_id,
        loop_id=loop_id,
        risk_level=risk_level,
        decision=decision,
        limit=limit,
    )


@router.get("/realtime-assessments/{snapshot_id}")
def get_realtime_assessment(snapshot_id: str) -> dict[str, Any]:
    snapshot = realtime_assessment_service.get(snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="snapshot_id not found")
    return snapshot


@router.post("/realtime-assessments/{snapshot_id}/tuning-task")
def create_tuning_task_from_assessment(snapshot_id: str, body: CreateAutoTuningTaskBody) -> dict[str, Any]:
    try:
        task = realtime_assessment_service.create_tuning_task(
            snapshot_id,
            confirm=body.confirm,
            trigger_mode=body.trigger_mode,
            reason=body.reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"task": task}


@router.get("/auto-tuning/tasks")
def list_auto_tuning_tasks(
    status: str | None = None,
    loop_id: str | None = None,
    asset_id: str | None = None,
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    return realtime_assessment_service.list_tuning_tasks(
        status=status,
        loop_id=loop_id,
        asset_id=asset_id,
        limit=limit,
    )


@router.post("/auto-tuning/tasks/{task_id}/prepare")
def prepare_auto_tuning_task(task_id: str, body: PrepareAutoTuningTaskBody) -> dict[str, Any]:
    try:
        return realtime_assessment_service.prepare_tuning_task(
            task_id,
            PrepareAutoTuningTaskRequest(
                confirm=body.confirm,
                use_llm_advisor=body.use_llm_advisor,
                selected_window_index=body.selected_window_index,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
