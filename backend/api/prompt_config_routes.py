"""Prompt configuration API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.prompt_config import store

router = APIRouter(tags=["prompt-config"])


class PromptConfigUpdate(BaseModel):
    assistant_system_prompt: str | None = None
    assistant_developer_prompt: str | None = None
    assistant_response_schema: str | None = None
    window_policy_system_prompt: str | None = None
    window_policy_user_prompt_template: str | None = None
    identification_review_system_prompt: str | None = None
    identification_review_user_prompt_template: str | None = None


def _validate_non_empty(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        raise HTTPException(400, f"{label}不能为空")
    return text


@router.get("/prompt-config")
async def get_prompt_config():
    """Get current prompt configuration."""
    return store.get()


@router.put("/prompt-config")
async def update_prompt_config(body: PromptConfigUpdate):
    """Update prompt configuration and persist it to local JSON."""
    cfg = store.update(
        assistant_system_prompt=_validate_non_empty(body.assistant_system_prompt, "AI 助手系统提示词"),
        assistant_developer_prompt=_validate_non_empty(body.assistant_developer_prompt, "AI 助手流程约束提示词"),
        assistant_response_schema=_validate_non_empty(body.assistant_response_schema, "AI 助手响应格式说明"),
        window_policy_system_prompt=_validate_non_empty(body.window_policy_system_prompt, "选窗策略提示词"),
        window_policy_user_prompt_template=_validate_non_empty(body.window_policy_user_prompt_template, "选窗用户提示词模板"),
        identification_review_system_prompt=_validate_non_empty(body.identification_review_system_prompt, "辨识评审提示词"),
        identification_review_user_prompt_template=_validate_non_empty(body.identification_review_user_prompt_template, "辨识评审用户提示词模板"),
    )
    return {"status": "ok", "config": cfg}


@router.post("/prompt-config/reset")
async def reset_prompt_config():
    """Reset prompt configuration to defaults."""
    cfg = store.reset_defaults()
    return {"status": "ok", "config": cfg}
