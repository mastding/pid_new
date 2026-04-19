"""会话历史 REST 端点：列表 / 详情 / 删除。

  GET    /api/sessions?limit=100&kind=tune
  GET    /api/sessions/{task_id}
  DELETE /api/sessions/{task_id}
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from core.session_log import delete_session, get_session, list_sessions

router = APIRouter(tags=["sessions"])


@router.get("/sessions")
def api_list_sessions(
    limit: int = Query(100, ge=1, le=500),
    kind: str | None = Query(None, regex="^(tune|consult)$"),
) -> dict[str, Any]:
    items = list_sessions(limit=limit, kind=kind)
    return {"total": len(items), "items": items}


@router.get("/sessions/{task_id}")
def api_get_session(task_id: str) -> dict[str, Any]:
    sess = get_session(task_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"会话 {task_id} 不存在")
    return sess


@router.delete("/sessions/{task_id}")
def api_delete_session(task_id: str) -> dict[str, Any]:
    ok = delete_session(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"会话 {task_id} 不存在")
    return {"deleted": task_id}
