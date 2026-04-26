"""模型配置 API 端点。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.model_config import store

router = APIRouter(tags=["model-config"])


class ModelConfigUpdate(BaseModel):
    model_api_url: str | None = None
    model_api_key: str | None = None
    model_name: str | None = None


@router.get("/model-config")
async def get_model_config():
    """获取当前模型配置（API Key 脱敏）。"""
    return store.get().masked()


@router.put("/model-config")
async def update_model_config(body: ModelConfigUpdate):
    """更新模型配置并持久化到文件，保存后立即生效。"""
    if body.model_api_url is not None:
        url = body.model_api_url.strip()
        if url and not url.startswith(("http://", "https://")):
            raise HTTPException(400, "model_api_url 必须以 http:// 或 https:// 开头")
    if body.model_name is not None and not body.model_name.strip():
        raise HTTPException(400, "model_name 不能为空")

    cfg = store.update(
        model_api_url=body.model_api_url.strip() if body.model_api_url is not None else None,
        model_api_key=body.model_api_key.strip() if body.model_api_key is not None else None,
        model_name=body.model_name.strip() if body.model_name is not None else None,
    )
    return {"status": "ok", "config": cfg.masked()}


@router.post("/model-config/test")
async def test_model_config():
    """测试当前模型配置的连通性与鉴权。

    用 1-token 请求验证 API URL 可达、Key 有效、模型名存在。
    """
    cfg = store.get()
    if not cfg.is_configured():
        raise HTTPException(400, "模型配置不完整：API URL 或 Key 为空，无法测试")

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=cfg.model_api_key,
            base_url=cfg.model_api_url,
            timeout=15.0,
        )
        client.chat.completions.create(
            model=cfg.model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return {"status": "ok", "message": f"连接成功：{cfg.model_name} @ {cfg.model_api_url}"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
