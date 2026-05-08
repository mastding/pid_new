"""Ontology context retrieval through registered MCP tools.

This module keeps the tuning pipeline independent from a specific ontology
server. It looks for an enabled MCP server exposing a natural-language `chat`
tool, asks for loop-level ontology facts, and returns a compact JSON payload
that can be injected into the LLM window-selection prompt.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from core.mcp_client import call_tool, list_tools
from core.mcp_config import store as mcp_store

# 单个 MCP server 的 list_tools / call_tool 超时（秒）。
# - list_tools 一般 < 1s，给 8s 容忍冷启动
# - call_tool 走的是本体 chat 工具，内部是 LLM 推理，实测 30-60s 量级，
#   设 90s 留一点安全边界；超出说明本体服务确实异常，会回退到默认策略
_LIST_TOOLS_TIMEOUT_S = 8.0
_CALL_TOOL_TIMEOUT_S = 90.0


def _extract_text_from_mcp_result(result: dict[str, Any]) -> str:
    """Extract readable text from common MCP tools/call result shapes."""
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        inner = structured.get("result")
        if isinstance(inner, dict) and isinstance(inner.get("content"), str):
            return inner["content"].strip()
        if isinstance(structured.get("content"), str):
            return structured["content"].strip()

    content = result.get("content")
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            # Some MCP servers wrap the real answer as a JSON string in text.
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                texts.append(text)
                continue
            if isinstance(parsed, dict):
                parsed_result = parsed.get("result")
                if isinstance(parsed_result, dict) and isinstance(parsed_result.get("content"), str):
                    texts.append(parsed_result["content"])
                elif isinstance(parsed.get("content"), str):
                    texts.append(parsed["content"])
                else:
                    texts.append(text)
            else:
                texts.append(text)
        return "\n\n".join(t.strip() for t in texts if t.strip())

    if isinstance(result.get("text"), str):
        return result["text"].strip()
    return ""


def _build_loop_ontology_query(loop_name: str, loop_type: str) -> str:
    loop_hint = loop_name.strip() or "当前回路"
    type_hint = loop_type.strip() or "unknown"
    return (
        f"请查询 PID 回路 {loop_hint} 的本体知识，回路类型提示为 {type_hint}。"
        "请重点返回：1) 本体中的实际回路位号或别名；2) PV/CV、MV、SP、DV 的名称和物理含义；"
        "3) 对象类型、阶次、正/反作用或过程增益方向；4) 时间尺度、滤波时间、死区、典型动态先验；"
        "5) 常见工况/扰动场景；6) 用这些本体知识评审窗口辨识时应优先选择或避开的窗口特征。"
        "请用简明中文输出，可包含条目。"
    )


async def fetch_loop_ontology_context_via_mcp(
    *,
    loop_name: str,
    loop_type: str,
    max_chars: int = 8000,
) -> dict[str, Any] | None:
    """Fetch loop ontology context from the first enabled MCP `chat` tool.

    Returns None when no suitable MCP tool is available or every call fails.
    The caller should treat failure as non-fatal and continue with deterministic
    or frontend-provided context.
    """
    query = _build_loop_ontology_query(loop_name, loop_type)
    last_error = ""
    for server in mcp_store.list():
        if not server.enabled:
            continue
        try:
            tools = await asyncio.wait_for(list_tools(server), timeout=_LIST_TOOLS_TIMEOUT_S)
        except asyncio.TimeoutError:
            last_error = f"{server.name}: tools/list timeout (>{_LIST_TOOLS_TIMEOUT_S:.0f}s)"
            continue
        except Exception as exc:  # MCP down / auth / protocol mismatch
            last_error = f"{server.name}: tools/list failed: {exc}"
            continue

        chat_tool = next((tool for tool in tools if tool.get("name") == "chat"), None)
        if not chat_tool:
            continue

        try:
            result = await asyncio.wait_for(
                call_tool(server, "chat", {"query": query}),
                timeout=_CALL_TOOL_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            last_error = f"{server.name}: chat timeout (>{_CALL_TOOL_TIMEOUT_S:.0f}s)"
            continue
        except Exception as exc:
            last_error = f"{server.name}: chat failed: {exc}"
            continue

        text = _extract_text_from_mcp_result(result)
        if not text:
            last_error = f"{server.name}: chat returned empty content"
            continue
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...（MCP 本体检索结果已截断）"
        return {
            "source": "registered_mcp_tool",
            "server_id": server.id,
            "server_name": server.name,
            "tool": "chat",
            "query": query,
            "content": text,
        }

    if last_error:
        return {
            "source": "registered_mcp_tool",
            "error": last_error,
            "content": "",
        }
    return None
