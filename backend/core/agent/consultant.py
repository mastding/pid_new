"""PID tuning consultant - single LLM agent with tool-calling loop.

This replaces the entire AutoGen multi-agent system (~2500 lines across
agent_factory.py, workflow_runner.py, event_mapper.py, agents_multiagent.py)
with a ~80-line tool-calling loop using the OpenAI-compatible API.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Callable

from openai import AsyncOpenAI

from config import settings
from core.agent.prompts import SYSTEM_PROMPT
from core.agent.tools import TOOL_DEFINITIONS


async def _collect_stream(
    response: Any,
    on_chunk: Callable[[str], None] | None = None,
) -> tuple[list[Any], str]:
    """Collect streaming response, extracting tool calls and text content."""
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    text_parts: list[str] = []

    async for chunk in response:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue

        if delta.content:
            text_parts.append(delta.content)
            if on_chunk:
                on_chunk(delta.content)

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_by_index:
                    tool_calls_by_index[idx] = {
                        "id": tc_delta.id or "",
                        "function": {"name": "", "arguments": ""},
                    }
                entry = tool_calls_by_index[idx]
                if tc_delta.id:
                    entry["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        entry["function"]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        entry["function"]["arguments"] += tc_delta.function.arguments

    tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
    return tool_calls, "".join(text_parts)


async def run_consultant(
    *,
    messages: list[dict[str, Any]],
    tool_handlers: dict[str, Callable[..., Any]],
    max_iterations: int = 10,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run the PID consultant agent with ReAct-style tool-calling loop.

    Args:
        messages: Conversation history (user/assistant messages).
        tool_handlers: Map of tool_name -> async handler function.
        max_iterations: Max tool-calling rounds before forcing a response.
        on_event: Optional callback for streaming events.

    Yields:
        Events: {"type": "text_chunk", "content": "..."} for streaming text,
                {"type": "tool_call", "name": "...", "args": {...}, "result": {...}},
                {"type": "final", "content": "..."} for the final answer.
    """
    client = AsyncOpenAI(
        api_key=settings.model_api_key,
        base_url=settings.model_api_url,
    )

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = await client.chat.completions.create(
            model=settings.model_name,
            messages=full_messages,
            tools=TOOL_DEFINITIONS,
            stream=True,
        )

        text_chunks: list[str] = []

        def on_chunk(chunk: str) -> None:
            text_chunks.append(chunk)

        tool_calls, text = await _collect_stream(response, on_chunk)

        if not tool_calls:
            yield {"type": "final", "content": text, "iterations": iteration}
            return

        # Append assistant message with tool calls
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if text:
            assistant_msg["content"] = text
        assistant_msg["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                },
            }
            for tc in tool_calls
        ]
        full_messages.append(assistant_msg)

        # Execute each tool call
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
            except json.JSONDecodeError:
                args = {}

            handler = tool_handlers.get(func_name)
            if handler is None:
                result = {"error": f"Unknown tool: {func_name}"}
            else:
                try:
                    result = await asyncio.wait_for(
                        handler(**args) if asyncio.iscoroutinefunction(handler) else asyncio.to_thread(handler, **args),
                        timeout=60.0,
                    )
                except asyncio.TimeoutError:
                    result = {"error": "工具执行超时 (60s)"}
                except Exception as exc:
                    result = {"error": str(exc)}

            yield {
                "type": "tool_call",
                "name": func_name,
                "args": args,
                "result": result,
            }

            full_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    # Max iterations reached
    yield {
        "type": "final",
        "content": "已达到最大迭代次数，请查看上方工具调用结果。",
        "iterations": iteration,
    }
