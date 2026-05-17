"""Experience APIs for learning snapshots and observed outcomes."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.experience import experience_store

router = APIRouter(tags=["experience"])


class AttachOutcomeBody(BaseModel):
    skill_name: str = Field(..., min_length=1)
    snapshot_id: str = Field(..., min_length=1)
    outcome: dict[str, Any] = Field(default_factory=dict)


@router.get("/experience/skills")
def list_experience_skills() -> dict[str, Any]:
    return experience_store.list_skills()


@router.get("/experience/snapshots")
def list_experience_snapshots(
    skill_name: str | None = None,
    only_with_outcome: bool = False,
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    return experience_store.list_snapshots(
        skill_name=skill_name,
        only_with_outcome=only_with_outcome,
        limit=limit,
    )


@router.post("/experience/outcomes")
def attach_experience_outcome(body: AttachOutcomeBody) -> dict[str, Any]:
    try:
        return experience_store.attach_outcome(
            skill_name=body.skill_name,
            snapshot_id=body.snapshot_id,
            outcome=body.outcome,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
