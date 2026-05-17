"""System configuration API endpoints."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from config import settings
from core.data_sources import data_source_config_store
from core.policies.loop_priors import MIN_REASONABLE_T, MODEL_ORDER, REALITY_T_RANGES
from core.policies.refinement import (
    REFINEMENT_MODEL_FALLBACKS,
    refinement_fallback_rule,
)

router = APIRouter(tags=["config"])


class DataSourceItem(BaseModel):
    id: str | None = None
    source_name: str = Field(..., min_length=1)
    source_type: str = Field(..., min_length=1)
    enabled: bool = True
    host: str | None = None
    port: int | None = Field(None, ge=1, le=65535)
    database: str | None = None
    username: str | None = None
    password: str | None = None
    polling_interval_s: int | None = Field(None, ge=0, le=86400)


class DataSourcesBody(BaseModel):
    items: list[DataSourceItem]


@router.get("/system-config")
async def get_config():
    """Return current runtime configuration (without secrets)."""
    return {
        "model_name": settings.model_name,
        "model_api_url": settings.model_api_url,
        "history_data_api_url": settings.history_data_api_url,
        "knowledge_graph_api_url": settings.knowledge_graph_api_url,
    }


@router.get("/policy-config")
async def get_policy_config():
    """Return read-only runtime policy configuration used by tuning pipeline."""
    refinement_rule = refinement_fallback_rule()
    return {
        "loop_priors": {
            "model_order": MODEL_ORDER,
            "min_reasonable_t": MIN_REASONABLE_T,
            "reality_t_ranges": {
                loop_type: {"min": value[0], "max": value[1]}
                for loop_type, value in REALITY_T_RANGES.items()
            },
        },
        "refinement": {
            "fallback_rule": {
                "min_confidence": refinement_rule.min_confidence,
                "min_r2": refinement_rule.min_r2,
                "min_window_quality": refinement_rule.min_window_quality,
                "max_model_pool_size": refinement_rule.max_model_pool_size,
            },
            "model_fallbacks": REFINEMENT_MODEL_FALLBACKS,
        },
    }


@router.get("/data-sources/config")
def get_data_sources_config() -> dict:
    """Return configured data sources without secrets."""
    return data_source_config_store.load()


@router.put("/data-sources/config")
def update_data_sources_config(body: DataSourcesBody) -> dict:
    """Persist data-source connection metadata without echoing secrets."""
    return data_source_config_store.save(body.model_dump())
