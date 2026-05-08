"""Minimal MCP client helpers for configured servers."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from urllib.parse import urljoin

import httpx

from core.mcp_config import McpServer

MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_CLIENT_INFO = {"name": "pid-v2", "version": "2.0.0"}


class McpClientError(RuntimeError):
    """Raised when an MCP server cannot be called."""


def parse_raw_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise McpClientError("raw_json must be a JSON object")
    return value


def effective_config(server: McpServer) -> dict[str, Any]:
    config = parse_raw_json(server.raw_json)
    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        return config

    selected = servers.get(server.name)
    if not isinstance(selected, dict):
        selected = next((item for item in servers.values() if isinstance(item, dict)), None)
    if not isinstance(selected, dict):
        return config

    merged = {**config, **selected}
    merged.pop("mcpServers", None)
    return merged


def headers_from_config(config: dict[str, Any]) -> dict[str, str]:
    raw_headers = config.get("headers", {})
    if not isinstance(raw_headers, dict):
        return {}
    return {str(key): str(value) for key, value in raw_headers.items()}


def protocol_version(config: dict[str, Any]) -> str:
    value = config.get("protocolVersion") or config.get("protocol_version")
    return str(value).strip() if value else MCP_PROTOCOL_VERSION


def initialize_payload(config: dict[str, Any] | None = None, request_id: int = 1) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": protocol_version(config or {}),
            "capabilities": {},
            "clientInfo": MCP_CLIENT_INFO,
        },
    }


def json_from_sse_text(text: str) -> dict[str, Any] | None:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for block in normalized.split("\n\n"):
        data_lines = [
            line.removeprefix("data:").strip()
            for line in block.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        try:
            value = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def summarize_initialize_result(data: dict[str, Any]) -> tuple[str, str]:
    if data.get("jsonrpc") != "2.0":
        return "error", "MCP initialize response is not JSON-RPC 2.0"
    if data.get("error"):
        return "error", f"MCP initialize returned error: {data['error']}"
    result = data.get("result")
    if not isinstance(result, dict):
        return "error", "MCP initialize response did not contain a result object"
    server_info = result.get("serverInfo") if isinstance(result.get("serverInfo"), dict) else {}
    capabilities = result.get("capabilities") if isinstance(result.get("capabilities"), dict) else {}
    return (
        "ok",
        "MCP initialize succeeded: "
        f"{server_info.get('name') or 'unknown server'}, "
        f"protocol={result.get('protocolVersion') or 'unknown'}, "
        f"capabilities={', '.join(sorted(capabilities.keys())) or 'none'}",
    )


async def _read_sse_endpoint(resp: httpx.Response) -> str | None:
    event_name = ""
    data_lines: list[str] = []
    line_count = 0
    async for line in resp.aiter_lines():
        line_count += 1
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
            data_lines = []
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
        elif not line:
            data = "\n".join(data_lines).strip()
            if event_name == "endpoint" and data:
                return data
            event_name = ""
            data_lines = []
        if line_count > 120:
            return None
    return None


def _response_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except json.JSONDecodeError:
        data = json_from_sse_text(resp.text)
    if not isinstance(data, dict):
        raise McpClientError(
            f"MCP response is not JSON or SSE JSON: content-type={resp.headers.get('content-type', '')}"
        )
    return data


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> tuple[dict[str, Any], str | None]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    resp = await client.post(url, content=body, headers=headers)
    if resp.status_code >= 400:
        raise McpClientError(f"MCP request failed: HTTP {resp.status_code}, body={resp.text[:500]}")
    session_id = resp.headers.get("mcp-session-id") or resp.headers.get("Mcp-Session-Id")
    return _response_json(resp), session_id


async def _post_notification(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    resp = await client.post(url, content=body, headers=headers)
    if resp.status_code >= 400:
        raise McpClientError(f"MCP notification failed: HTTP {resp.status_code}, body={resp.text[:500]}")


async def _with_streamable_http_session(
    server: McpServer,
    config: dict[str, Any],
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json; charset=utf-8",
        "MCP-Protocol-Version": protocol_version(config),
        **headers_from_config(config),
    }
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        init_data, session_id = await _post_json(client, server.url.strip(), initialize_payload(config), headers)
        if init_data.get("error"):
            return init_data
        session_headers = dict(headers)
        if session_id:
            session_headers["Mcp-Session-Id"] = session_id
        await _post_notification(
            client,
            server.url.strip(),
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            session_headers,
        )
        data, _ = await _post_json(
            client,
            server.url.strip(),
            {"jsonrpc": "2.0", "id": 2, "method": method, "params": params or {}},
            session_headers,
        )
        return data


async def _with_sse_session(
    server: McpServer,
    config: dict[str, Any],
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stream_headers = {
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        **headers_from_config(config),
    }
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        async with client.stream("GET", server.url.strip(), headers=stream_headers) as resp:
            if resp.status_code >= 400:
                raise McpClientError(f"SSE connection failed: HTTP {resp.status_code}")
            endpoint = await asyncio.wait_for(_read_sse_endpoint(resp), timeout=8.0)
        if not endpoint:
            raise McpClientError("SSE endpoint opened, but no MCP endpoint event was received")
        post_url = urljoin(server.url.strip(), endpoint)
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json; charset=utf-8",
            **headers_from_config(config),
        }
        init_data, _ = await _post_json(client, post_url, initialize_payload(config), headers)
        if init_data.get("error"):
            return init_data
        await _post_notification(client, post_url, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}, headers)
        data, _ = await _post_json(client, post_url, {"jsonrpc": "2.0", "id": 2, "method": method, "params": params or {}}, headers)
        return data


async def _read_stdio_json_line(stream: asyncio.StreamReader) -> dict[str, Any] | None:
    for _ in range(80):
        line = await asyncio.wait_for(stream.readline(), timeout=30.0)
        if not line:
            return None
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


async def _with_stdio_session(
    config: dict[str, Any],
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command = config.get("command")
    args = config.get("args", [])
    env = config.get("env")
    cwd = config.get("cwd")
    if not isinstance(command, str) or not command.strip():
        raise McpClientError("stdio transport requires raw_json.command")
    if args is None:
        args = []
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise McpClientError("stdio raw_json.args must be a string array")
    if env is not None and not isinstance(env, dict):
        raise McpClientError("stdio raw_json.env must be an object")
    if cwd is not None and not isinstance(cwd, str):
        raise McpClientError("stdio raw_json.cwd must be a string")

    proc = await asyncio.create_subprocess_exec(
        command,
        *args,
        cwd=cwd,
        env={**os.environ, **{k: str(v) for k, v in env.items()}} if isinstance(env, dict) else None,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write((json.dumps(initialize_payload(config), ensure_ascii=False) + "\n").encode("utf-8"))
        await proc.stdin.drain()
        init_data = await _read_stdio_json_line(proc.stdout)
        if not init_data:
            raise McpClientError("stdio MCP initialize did not return JSON")
        if init_data.get("error"):
            return init_data
        proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}, ensure_ascii=False) + "\n").encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.write((json.dumps({"jsonrpc": "2.0", "id": 2, "method": method, "params": params or {}}, ensure_ascii=False) + "\n").encode("utf-8"))
        await proc.stdin.drain()
        data = await _read_stdio_json_line(proc.stdout)
        if not data:
            raise McpClientError("stdio MCP request did not return JSON")
        return data
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()


async def mcp_request(server: McpServer, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    config = effective_config(server)
    if server.transport == "streamable-http":
        return await _with_streamable_http_session(server, config, method, params)
    if server.transport == "sse":
        return await _with_sse_session(server, config, method, params)
    if server.transport == "stdio":
        return await _with_stdio_session(config, method, params)
    raise McpClientError(f"Unsupported transport: {server.transport}")


async def initialize_check(server: McpServer) -> dict[str, str]:
    config = effective_config(server)
    try:
        if server.transport == "streamable-http":
            data = await _with_streamable_http_session(server, config, "ping")
        elif server.transport == "sse":
            data = await _with_sse_session(server, config, "ping")
        elif server.transport == "stdio":
            data = await _with_stdio_session(config, "ping")
        else:
            return {"status": "error", "message": f"Unsupported transport: {server.transport}"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
    if data.get("error"):
        # initialize succeeded if the follow-up ping is unknown/unsupported.
        return {"status": "ok", "message": "MCP initialize succeeded; follow-up ping returned an MCP error"}
    return {"status": "ok", "message": "MCP initialize succeeded"}


async def list_tools(server: McpServer) -> list[dict[str, Any]]:
    data = await mcp_request(server, "tools/list", {})
    if data.get("error"):
        raise McpClientError(f"tools/list returned error: {data['error']}")
    result = data.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
        raise McpClientError("tools/list response did not contain result.tools")
    return [tool for tool in result["tools"] if isinstance(tool, dict)]


async def call_tool(server: McpServer, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    data = await mcp_request(server, "tools/call", {"name": tool_name, "arguments": arguments or {}})
    if data.get("error"):
        raise McpClientError(f"tools/call returned error: {data['error']}")
    result = data.get("result")
    if not isinstance(result, dict):
        raise McpClientError("tools/call response did not contain a result object")
    return result
