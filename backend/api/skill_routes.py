"""Skill metadata APIs for the agent settings surface."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from core.skills import registry

router = APIRouter(tags=["skills"])


@router.get("/skills")
async def list_skills() -> dict[str, Any]:
    """Return all registered core and external skills with workflow metadata."""
    items = registry.all_metadata()
    stages: dict[str, int] = {}
    risks: dict[str, int] = {}
    for item in items:
        stage = str(item.get("stage") or "general")
        risk = str(item.get("risk_level") or "low")
        stages[stage] = stages.get(stage, 0) + 1
        risks[risk] = risks.get(risk, 0) + 1
    return {
        "total": len(items),
        "items": items,
        "summary": {
            "stages": stages,
            "risks": risks,
        },
    }
