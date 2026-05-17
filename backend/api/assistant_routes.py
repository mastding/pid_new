"""Streaming AI assistant endpoints for dialogue mode."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from core import assistant_sessions
from core.history.store import assess_loop, get_loop, get_loop_features, get_loop_monitoring
from core.model_config import store as model_cfg_store
from core.prompt_config import store as prompt_cfg_store
from core.realtime import realtime_assessment_service

router = APIRouter(tags=["assistant"])


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


def _compact(value: Any, max_chars: int = 360) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return text if len(text) <= max_chars else f"{text[:max_chars]}..."


def _assistant_skill_plan(message: str, loop_context: dict[str, Any]) -> list[dict[str, Any]]:
    text = message.lower()
    realtime = loop_context.get("realtime_assessment") if isinstance(loop_context.get("realtime_assessment"), dict) else {}
    monitoring = loop_context.get("monitoring") if isinstance(loop_context.get("monitoring"), dict) else {}
    features = loop_context.get("features") if isinstance(loop_context.get("features"), dict) else {}
    assessment = loop_context.get("assessment") if isinstance(loop_context.get("assessment"), dict) else {}

    plan: list[dict[str, Any]] = []

    def add(name: str, reason: str, inputs: dict[str, Any], outputs: dict[str, Any], risk_level: str = "medium") -> None:
        plan.append({
            "type": "tool_event",
            "name": name,
            "status": "ready",
            "risk_level": risk_level,
            "reason": reason,
            "inputs_summary": inputs,
            "outputs_summary": outputs,
            "detail": f"{reason}；输入：{_compact(inputs, 140)}；输出：{_compact(outputs, 180)}",
        })

    add(
        "load_loop_context",
        "读取当前回路、时间范围和页面上下文",
        {"loop_id": (loop_context.get("loop") or {}).get("loop_id"), "status": loop_context.get("status")},
        {"has_monitoring": bool(monitoring), "has_realtime_assessment": bool(realtime)},
        "low",
    )
    if any(word in text for word in ["数据", "质量", "缺失", "噪声", "异常", "data"]):
        add("assess_loop_monitoring", "用户关注数据质量或监控健康度", {"data_health": monitoring.get("data_health")}, {"status": monitoring.get("status"), "overall_score": monitoring.get("overall_score")}, "low")
    if any(word in text for word in ["工况", "波动", "负荷", "区间", "condition"]):
        add("summarize_data", "用户关注工况、波动或历史画像", {"operating_condition_profile": features.get("operating_condition_profile")}, {"pv_stats": features.get("pv_stats"), "mv_stats": features.get("mv_stats")}, "low")
    if any(word in text for word in ["窗口", "选窗", "评估", "准入", "window"]):
        add("assess_loop_assessment", "用户关注窗口评估或整定准入", {"assessment_summary": assessment.get("summary")}, {"tuning_readiness": assessment.get("tuning_readiness")}, "medium")
    if any(word in text for word in ["harris", "cpk", "指标", "可靠", "诊断", "根因", "整定", "pid", "建议"]):
        add("diagnose_realtime_assessment", "用户关注指标诊断、根因或整定建议", {"metrics": realtime.get("metrics"), "ontology_missing_fields": realtime.get("ontology_missing_fields")}, {"diagnosis": realtime.get("diagnosis"), "decision": realtime.get("decision")}, "medium")
    if any(word in text for word in ["整定", "pid", "推荐", "下发", "任务"]):
        add("decide_realtime_tuning_action", "用户关注是否进入整定流程", {"diagnosis": realtime.get("diagnosis"), "decision": realtime.get("decision")}, {"required_confirmations": (realtime.get("decision") or {}).get("required_confirmations")}, "high")

    review_terms = ["result", "review", "simulation", "curve", "rating", "final_rating", "\u590d\u6838", "\u7ed3\u679c", "\u4eff\u771f", "\u66f2\u7ebf", "\u8bc4\u5206", "\u4e0a\u7ebf"]
    if any(word in text for word in review_terms):
        add("review_auto_tuning_result", "review tuning result, simulation curve, and adoption readiness", {"latest_realtime_decision": realtime.get("decision")}, {"requires": ["evaluation", "pid_params", "simulation_curve_review"]}, "high")

    return plan[:6]


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
    try:
        latest = realtime_assessment_service.latest(loop_id=loop_id, limit=1)
        items = latest.get("items") or []
        if items:
            snapshot = items[0]
            payload["realtime_assessment"] = {
                "snapshot_id": snapshot.get("snapshot_id"),
                "created_at": snapshot.get("created_at"),
                "time_window": snapshot.get("time_window"),
                "risk_level": snapshot.get("risk_level"),
                "score": snapshot.get("score"),
                "decision": snapshot.get("decision"),
                "metrics": [
                    {
                        "name": item.get("name"),
                        "value": item.get("value"),
                        "level": item.get("level"),
                        "confidence": item.get("confidence"),
                        "success": item.get("success"),
                    }
                    for item in (snapshot.get("metrics") or [])
                ],
                "diagnosis": [
                    {
                        "root_cause": item.get("root_cause"),
                        "confidence": item.get("confidence"),
                        "severity": item.get("severity"),
                        "action": item.get("action"),
                    }
                    for item in (snapshot.get("diagnosis") or [])[:5]
                ],
                "ontology_missing_fields": ((snapshot.get("ontology") or {}).get("missing_fields") or []),
                "skill_trace": [
                    {
                        "skill_name": item.get("skill_name"),
                        "risk_level": item.get("risk_level"),
                        "status": item.get("status"),
                        "duration_ms": item.get("duration_ms"),
                        "outputs_summary": item.get("outputs_summary"),
                    }
                    for item in (snapshot.get("skill_trace") or [])[:8]
                ],
            }
    except Exception as exc:
        payload["realtime_assessment_error"] = str(exc)[:300]
    return payload


async def _stream_llm(
    *,
    messages: list[dict[str, str]],
    context: dict[str, Any],
    on_done: Any | None = None,
):
    prompt_cfg = prompt_cfg_store.get()
    model_cfg = model_cfg_store.get()

    raw_events: list[dict[str, Any]] = []
    event = {"type": "thinking_step", "content": "读取当前会话、选中回路和页面上下文。"}
    raw_events.append(event)
    yield _sse(event)
    loop_context = _build_loop_context(context)
    event = {"type": "tool_event", "name": "load_loop_context", "status": loop_context.get("status", "ok")}
    raw_events.append(event)
    yield _sse(event)
    user_message = messages[-1]["content"] if messages else ""
    skill_plan = _assistant_skill_plan(user_message, loop_context)
    for event in skill_plan[1:]:
        raw_events.append(event)
        yield _sse(event)
    event = {"type": "thinking_step", "content": "整理可展示的分析过程摘要，并检查高风险动作边界。"}
    raw_events.append(event)
    yield _sse(event)

    if not model_cfg.model_api_key or not model_cfg.model_api_url:
        yield _sse({
            "type": "error",
            "message": "模型配置未完成，请先在系统设置 / 模型配置中填写 API 地址、Key 和模型名称。",
        })
        yield _sse({"type": "done"})
        return

    context_json = json.dumps(
        {"page_context": context, "loop_context": loop_context, "assistant_skill_plan": skill_plan},
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
