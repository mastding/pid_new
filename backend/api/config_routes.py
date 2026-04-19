"""System configuration API endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from config import settings

router = APIRouter(tags=["config"])


@router.get("/system-config")
async def get_config():
    """Return current runtime configuration (without secrets)."""
    return {
        "model_name": settings.model_name,
        "model_api_url": settings.model_api_url,
        "history_data_api_url": settings.history_data_api_url,
        "knowledge_graph_api_url": settings.knowledge_graph_api_url,
    }
