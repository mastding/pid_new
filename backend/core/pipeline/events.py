"""SSE event types for streaming pipeline progress."""
from __future__ import annotations

from typing import Any


def stage_event(stage: str, status: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a pipeline stage event for SSE streaming."""
    return {
        "type": "stage",
        "stage": stage,
        "status": status,
        **({"data": data} if data else {}),
    }


def error_event(message: str, stage: str = "", error_code: str = "") -> dict[str, Any]:
    return {
        "type": "error",
        "message": message,
        "stage": stage,
        "error_code": error_code,
    }


def result_event(data: dict[str, Any]) -> dict[str, Any]:
    return {"type": "result", "data": data}


def consultant_event(content: str, tool_name: str = "", tool_result: Any = None) -> dict[str, Any]:
    event: dict[str, Any] = {"type": "consultant", "content": content}
    if tool_name:
        event["tool_name"] = tool_name
    if tool_result is not None:
        event["tool_result"] = tool_result
    return event
