"""Streaming AI assistant endpoints for dialogue mode."""
from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from core import assistant_sessions
from core.history.store import assess_loop, get_loop, get_loop_features, get_loop_monitoring
from core.model_config import store as model_cfg_store
from core.prompt_config import store as prompt_cfg_store

router = APIRouter(tags=["assistant"])


class AssistantChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AssistantStreamRequest(BaseModel):
    messages: list[AssistantChatMessage] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class AssistantSessionCreate(BaseModel):
    title: str | None = None
    loop_id: str | None = None


class AssistantSessionUpdate(BaseModel):
    title: str | None = None
    loop_id: str | None = None


class AssistantSessionStreamRequest(BaseModel):
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def _reasoning_from_delta(delta: Any) -> str:
    value = getattr(delta, "reasoning_content", None)
    if value:
        return str(value)
    extra = getattr(delta, "model_extra", None)
    if isinstance(extra, dict) and extra.get("reasoning_content"):
        return str(extra["reasoning_content"])
    return ""


def _public_reasoning_summary(text: str) -> str:
    if not text:
        return ""
    return "模型已结合会话历史、当前回路上下文和可用监控指标生成分析。"


def _build_loop_context(context: dict[str, Any]) -> dict[str, Any]:
    loop_id = str(context.get("loop_id") or "").strip()
    if not loop_id:
        return {"status": "no_loop_selected"}
    loop = get_loop(loop_id)
    if not loop:
        return {"status": "loop_not_found", "loop_id": loop_id}

    start_time = context.get("start_time")
    end_time = context.get("end_time")
    payload: dict[str, Any] = {
        "status": "ok",
        "loop": {
            "loop_id": loop_id,
            "loop_type": loop.get("loop_type"),
            "source_filename": loop.get("source_filename"),
            "start_time": loop.get("start_time"),
            "end_time": loop.get("end_time"),
        },
    }
    try:
        monitoring = get_loop_monitoring(loop_id, start_time=start_time, end_time=end_time)
        payload["monitoring"] = monitoring.get("monitoring") if isinstance(monitoring, dict) else monitoring
    except Exception as exc:
        payload["monitoring_error"] = str(exc)[:300]
    try:
        features = get_loop_features(loop_id, start_time=start_time, end_time=end_time)
        if isinstance(features, dict):
            payload["features"] = {
                "identity": features.get("identity"),
                "data_profile": features.get("data_profile"),
                "pv_stats": features.get("pv_stats"),
                "mv_stats": features.get("mv_stats"),
                "constraint_raw": features.get("constraint_raw"),
                "pv_mv_relation_raw": features.get("pv_mv_relation_raw"),
                "operating_condition_profile": features.get("operating_condition_profile"),
                "oscillation_raw": features.get("oscillation_raw"),
                "performance_raw": features.get("performance_raw"),
            }
    except Exception as exc:
        payload["features_error"] = str(exc)[:300]
    try:
        payload["assessment"] = assess_loop(loop_id, start_time=start_time, end_time=end_time)
    except Exception as exc:
        payload["assessment_error"] = str(exc)[:300]
    return payload


async def _stream_llm(
    *,
    messages: list[dict[str, str]],
    context: dict[str, Any],
    on_done: Any | None = None,
):
    prompt_cfg = prompt_cfg_store.get()
    model_cfg = model_cfg_store.get()

    yield _sse({"type": "thinking_step", "content": "读取当前会话、选中回路和页面上下文。"})
    loop_context = _build_loop_context(context)
    yield _sse({"type": "tool_event", "name": "load_loop_context", "status": loop_context.get("status", "ok")})
    yield _sse({"type": "thinking_step", "content": "整理可展示的分析过程摘要，并检查高风险动作边界。"})

    if not model_cfg.model_api_key or not model_cfg.model_api_url:
        yield _sse({
            "type": "error",
            "message": "模型配置未完成，请先在系统设置 / 模型配置中填写 API 地址、Key 和模型名称。",
        })
        yield _sse({"type": "done"})
        return

    context_json = json.dumps(
        {"page_context": context, "loop_context": loop_context},
        ensure_ascii=False,
        default=str,
    )
    history_json = json.dumps(messages[-20:], ensure_ascii=False, default=str)
    system_prompt = "\n\n".join([
        prompt_cfg.assistant_system_prompt,
        prompt_cfg.assistant_developer_prompt,
        "对话模式规则：你是 PID 智能整定对话助手。可以解释、诊断、建议下一步，但不能声称已下发参数或已执行高风险整定动作。",
        "界面需要展示“分析过程摘要”和“完整回复”。不要输出隐藏思维链原文；请在正文前用“分析过程摘要：”给出可展示摘要。",
        "如果建议后续动作，请在回答末尾用“建议动作：”列出短按钮文案，例如 查看趋势与频谱、生成整定先验、进入整定任务。",
        f"响应格式参考：\n{prompt_cfg.assistant_response_schema}",
    ])
    user_prompt = "\n\n".join([
        "当前上下文 JSON：",
        context_json[:24000],
        "",
        "会话历史 JSON：",
        history_json[:12000],
        "",
        "请回答最后一条用户消息。输出中文，保留完整说明，面向工程师可读。",
    ])

    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    raw_events: list[dict[str, Any]] = []
    stream_error = ""
    try:
        client = AsyncOpenAI(
            api_key=model_cfg.model_api_key,
            base_url=model_cfg.model_api_url,
            timeout=90.0,
        )
        stream = await client.chat.completions.create(
            model=model_cfg.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1600,
            stream=True,
        )
        reasoning_announced = False
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = _reasoning_from_delta(delta)
            if reasoning:
                reasoning_parts.append(reasoning)
                if not reasoning_announced:
                    reasoning_announced = True
                    summary = "模型正在结合回路画像、监控状态和会话历史形成判断。"
                    raw_events.append({"type": "thinking_step", "content": summary})
                    yield _sse({"type": "thinking_step", "content": summary})
            content = getattr(delta, "content", None)
            if content:
                answer_parts.append(str(content))
                yield _sse({"type": "answer_delta", "content": content})
    except Exception as exc:
        stream_error = str(exc)[:500]
        yield _sse({"type": "error", "message": stream_error})

    answer = "".join(answer_parts)
    reasoning_summary = _public_reasoning_summary("".join(reasoning_parts))
    if on_done and (answer.strip() or reasoning_summary.strip()):
        await on_done(answer, reasoning_summary, raw_events)
    yield _sse({
        "type": "done",
        "answer": answer,
        "reasoning_summary": reasoning_summary,
        "error": stream_error or None,
    })


async def _assistant_sse(request: AssistantStreamRequest):
    messages = [item.model_dump() for item in request.messages]
    async for item in _stream_llm(messages=messages, context=request.context):
        yield item


@router.post("/assistant/stream")
async def assistant_stream(request: AssistantStreamRequest):
    """Backward-compatible stateless assistant stream."""
    return StreamingResponse(
        _assistant_sse(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/assistant/sessions")
def list_assistant_sessions(limit: int = Query(100, ge=1, le=300)) -> dict[str, Any]:
    items = assistant_sessions.list_sessions(limit=limit)
    return {"total": len(items), "items": items}


@router.post("/assistant/sessions")
def create_assistant_session(body: AssistantSessionCreate) -> dict[str, Any]:
    session = assistant_sessions.create_session(title=body.title, loop_id=body.loop_id)
    return {"session": session}


@router.get("/assistant/sessions/{session_id}")
def get_assistant_session(session_id: str) -> dict[str, Any]:
    session = assistant_sessions.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"对话 {session_id} 不存在")
    return {"session": session}


@router.put("/assistant/sessions/{session_id}")
def update_assistant_session(session_id: str, body: AssistantSessionUpdate) -> dict[str, Any]:
    session = assistant_sessions.update_session(session_id, title=body.title, loop_id=body.loop_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"对话 {session_id} 不存在")
    return {"session": session}


@router.delete("/assistant/sessions/{session_id}")
def delete_assistant_session(session_id: str) -> dict[str, Any]:
    if not assistant_sessions.delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"对话 {session_id} 不存在")
    return {"deleted": session_id}


@router.post("/assistant/sessions/{session_id}/stream")
async def stream_assistant_session(session_id: str, request: AssistantSessionStreamRequest):
    session = assistant_sessions.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"对话 {session_id} 不存在")

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message is required")
    assistant_sessions.append_message(session_id, role="user", content=user_message)
    merged_context = dict(request.context)
    if session.get("loop_id") and not merged_context.get("loop_id"):
        merged_context["loop_id"] = session.get("loop_id")
    messages = [
        {"role": item.get("role", "user"), "content": item.get("content", "")}
        for item in (assistant_sessions.get_session(session_id) or {}).get("messages", [])
    ]

    async def on_done(answer: str, reasoning_summary: str, raw_events: list[dict[str, Any]]) -> None:
        if not answer.strip() and not reasoning_summary.strip():
            return
        assistant_sessions.append_message(
            session_id,
            role="assistant",
            content=answer,
            reasoning_summary=reasoning_summary,
            raw_events=raw_events,
        )

    return StreamingResponse(
        _stream_llm(messages=messages, context=merged_context, on_done=on_done),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
