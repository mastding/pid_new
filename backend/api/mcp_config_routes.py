"""MCP server configuration REST endpoints."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.mcp_config import SUPPORTED_TRANSPORTS, store
from core.mcp_client import McpClientError, call_tool, initialize_check, list_tools
from core.ontology_query_config import store as ontology_query_store

router = APIRouter(tags=["mcp-config"])

MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_CLIENT_INFO = {"name": "pid-v2", "version": "2.0.0"}


class McpServerCreate(BaseModel):
    name: str
    url: str = ""
    transport: str = "sse"
    raw_json: str = ""
    enabled: bool = True
    description: str = ""


class McpServerUpdate(BaseModel):
    name: str | None = None
    url: str | None = None
    transport: str | None = None
    raw_json: str | None = None
    enabled: bool | None = None
    description: str | None = None


class McpToolCallRequest(BaseModel):
    arguments: dict[str, Any] = {}


class OntologyQueryConfigUpdate(BaseModel):
    query_template: str | None = None


def _validate_transport(transport: str) -> None:
    if transport not in SUPPORTED_TRANSPORTS:
        raise HTTPException(400, f"transport must be one of {SUPPORTED_TRANSPORTS}")


def _validate_url_for_transport(transport: str, url: str) -> None:
    if transport in {"sse", "streamable-http"}:
        if not url.strip():
            raise HTTPException(400, f"url is required when transport={transport}")
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"}:
            raise HTTPException(400, "url must start with http:// or https://")


def _parse_raw_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"raw_json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HTTPException(400, "raw_json must be a JSON object")
    return value


def _validate_raw_json(raw: str) -> None:
    _parse_raw_json(raw)


def _effective_mcp_config(config: dict[str, Any], server_name: str) -> dict[str, Any]:
    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        return config

    selected = servers.get(server_name)
    if not isinstance(selected, dict):
        selected = next((item for item in servers.values() if isinstance(item, dict)), None)
    if not isinstance(selected, dict):
        return config

    merged = {**config, **selected}
    merged.pop("mcpServers", None)
    return merged


def _headers_from_config(config: dict[str, Any]) -> dict[str, str]:
    raw_headers = config.get("headers", {})
    if not isinstance(raw_headers, dict):
        return {}
    return {str(key): str(value) for key, value in raw_headers.items()}


def _protocol_version(config: dict[str, Any]) -> str:
    value = config.get("protocolVersion") or config.get("protocol_version")
    return str(value).strip() if value else MCP_PROTOCOL_VERSION


def _initialize_payload(config: dict[str, Any] | None = None, request_id: int = 1) -> dict[str, Any]:
    protocol_version = _protocol_version(config or {})
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": protocol_version,
            "capabilities": {},
            "clientInfo": MCP_CLIENT_INFO,
        },
    }


def _summarize_initialize_result(data: dict[str, Any]) -> tuple[str, str]:
    if data.get("jsonrpc") != "2.0":
        return "error", "MCP initialize response is not JSON-RPC 2.0"
    if data.get("error"):
        return "error", f"MCP initialize returned error: {data['error']}"

    result = data.get("result")
    if not isinstance(result, dict):
        return "error", "MCP initialize response did not contain a result object"

    protocol = result.get("protocolVersion") or "unknown"
    server_info = result.get("serverInfo") if isinstance(result.get("serverInfo"), dict) else {}
    server_name = server_info.get("name") or "unknown server"
    capabilities = result.get("capabilities") if isinstance(result.get("capabilities"), dict) else {}
    capability_names = ", ".join(sorted(capabilities.keys())) or "none"
    return (
        "ok",
        f"MCP initialize succeeded: {server_name}, protocol={protocol}, capabilities={capability_names}",
    )


def _json_from_sse_text(text: str) -> dict[str, Any] | None:
    for block in text.split("\n\n"):
        data_lines = [
            line.removeprefix("data:").strip()
            for line in block.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


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


async def _test_streamable_http(url: str, config: dict[str, Any]) -> dict[str, str]:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "MCP-Protocol-Version": _protocol_version(config),
        **_headers_from_config(config),
    }
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.post(url, json=_initialize_payload(config), headers=headers)
    except Exception as exc:
        return {"status": "error", "message": f"Cannot reach streamable-http MCP endpoint: {exc}"}

    if resp.status_code >= 400:
        return {
            "status": "error",
            "message": f"MCP initialize failed: HTTP {resp.status_code}, body={resp.text[:500]}",
        }

    content_type = resp.headers.get("content-type", "")
    try:
        data = resp.json()
    except json.JSONDecodeError:
        data = _json_from_sse_text(resp.text) if "text/event-stream" in content_type else None
    if not isinstance(data, dict):
        return {
            "status": "error",
            "message": f"MCP initialize response is not JSON or SSE JSON: content-type={content_type}",
        }

    status, message = _summarize_initialize_result(data)
    return {"status": status, "message": message}


async def _test_sse(url: str, config: dict[str, Any]) -> dict[str, str]:
    headers = {
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        **_headers_from_config(config),
    }
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code >= 400:
                    return {
                        "status": "error",
                        "message": f"SSE connection failed: HTTP {resp.status_code}",
                    }
                content_type = resp.headers.get("content-type", "")
                if "text/event-stream" not in content_type:
                    return {
                        "status": "error",
                        "message": f"SSE endpoint did not return text/event-stream: {content_type or 'unknown'}",
                    }
                try:
                    endpoint = await asyncio.wait_for(_read_sse_endpoint(resp), timeout=8.0)
                except asyncio.TimeoutError:
                    endpoint = None

            if not endpoint:
                return {
                    "status": "error",
                    "message": "SSE endpoint opened, but no MCP endpoint event was received",
                }

            post_url = urljoin(url, endpoint)
            post_headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                **_headers_from_config(config),
            }
            init_resp = await client.post(post_url, json=_initialize_payload(config), headers=post_headers)
    except Exception as exc:
        return {"status": "error", "message": f"SSE MCP initialize failed: {exc}"}

    if init_resp.status_code >= 400:
        return {
            "status": "error",
            "message": f"SSE MCP initialize failed: HTTP {init_resp.status_code}, body={init_resp.text[:500]}",
        }

    try:
        data = init_resp.json()
    except json.JSONDecodeError:
        return {
            "status": "error",
            "message": f"SSE MCP initialize response is not JSON: {init_resp.text[:500]}",
        }
    status, message = _summarize_initialize_result(data)
    return {"status": status, "message": message}


async def _read_stdio_json_line(stream: asyncio.StreamReader) -> dict[str, Any] | None:
    for _ in range(40):
        line = await asyncio.wait_for(stream.readline(), timeout=8.0)
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


async def _test_stdio(config: dict[str, Any]) -> dict[str, str]:
    command = config.get("command")
    args = config.get("args", [])
    env = config.get("env", None)
    cwd = config.get("cwd", None)
    if not isinstance(command, str) or not command.strip():
        return {"status": "error", "message": "stdio transport requires raw_json.command"}
    if args is None:
        args = []
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        return {"status": "error", "message": "stdio raw_json.args must be a string array"}
    if env is not None and not isinstance(env, dict):
        return {"status": "error", "message": "stdio raw_json.env must be an object"}
    if cwd is not None and not isinstance(cwd, str):
        return {"status": "error", "message": "stdio raw_json.cwd must be a string"}

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=cwd,
            env={**os.environ, **{k: str(v) for k, v in env.items()}} if isinstance(env, dict) else None,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write((json.dumps(_initialize_payload(config)) + "\n").encode("utf-8"))
        await proc.stdin.drain()
        data = await _read_stdio_json_line(proc.stdout)
        if not data:
            stderr = ""
            if proc.stderr:
                try:
                    raw_stderr = await asyncio.wait_for(proc.stderr.read(500), timeout=1.0)
                    stderr = raw_stderr.decode("utf-8", errors="replace")
                except asyncio.TimeoutError:
                    pass
            return {"status": "error", "message": f"stdio MCP initialize did not return JSON. stderr={stderr[:500]}"}
        status, message = _summarize_initialize_result(data)
        return {"status": status, "message": message}
    except FileNotFoundError:
        return {"status": "error", "message": f"stdio command not found: {command}"}
    except Exception as exc:
        return {"status": "error", "message": f"stdio MCP initialize failed: {exc}"}
    finally:
        if proc and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()


@router.get("/mcp-servers")
async def list_mcp_servers() -> dict[str, Any]:
    items = [server.model_dump() for server in store.list()]
    return {"total": len(items), "items": items}


def _ontology_query_config_payload() -> dict[str, Any]:
    config = ontology_query_store.get()
    return {
        **config.model_dump(),
        "placeholders": [
            {"name": "$loop_name", "description": "当前回路位号，例如 5203_FIC_10103"},
            {"name": "$loop_type", "description": "当前回路类型，例如 flow、pressure、temperature、level"},
        ],
        "preview": ontology_query_store.render(loop_name="5203_FIC_10103", loop_type="flow"),
    }


@router.get("/mcp-ontology-query-config")
async def get_mcp_ontology_query_config() -> dict[str, Any]:
    return _ontology_query_config_payload()


@router.put("/mcp-ontology-query-config")
async def update_mcp_ontology_query_config(body: OntologyQueryConfigUpdate) -> dict[str, Any]:
    query_template = body.query_template
    if query_template is not None and not query_template.strip():
        raise HTTPException(400, "query_template cannot be empty")
    ontology_query_store.update(query_template=query_template.strip() if query_template is not None else None)
    return {"status": "ok", "config": _ontology_query_config_payload()}


@router.post("/mcp-ontology-query-config/reset")
async def reset_mcp_ontology_query_config() -> dict[str, Any]:
    ontology_query_store.reset_defaults()
    return {"status": "ok", "config": _ontology_query_config_payload()}


@router.post("/mcp-servers")
async def create_mcp_server(body: McpServerCreate) -> dict[str, Any]:
    _validate_transport(body.transport)
    _validate_url_for_transport(body.transport, body.url)
    _validate_raw_json(body.raw_json)
    try:
        server = store.create(
            name=body.name.strip(),
            url=body.url.strip(),
            transport=body.transport,
            raw_json=body.raw_json,
            enabled=body.enabled,
            description=body.description.strip(),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"status": "ok", "server": server.model_dump()}


@router.put("/mcp-servers/{server_id}")
async def update_mcp_server(server_id: str, body: McpServerUpdate) -> dict[str, Any]:
    if body.transport is not None:
        _validate_transport(body.transport)
    if body.raw_json is not None:
        _validate_raw_json(body.raw_json)
    if body.transport is not None or body.url is not None:
        existing = store.get(server_id)
        if not existing:
            raise HTTPException(404, f"MCP server {server_id} does not exist")
        effective_transport = body.transport or existing.transport
        effective_url = body.url if body.url is not None else existing.url
        _validate_url_for_transport(effective_transport, effective_url)
    try:
        server = store.update(
            server_id,
            name=body.name.strip() if body.name is not None else None,
            url=body.url.strip() if body.url is not None else None,
            transport=body.transport,
            raw_json=body.raw_json,
            enabled=body.enabled,
            description=body.description.strip() if body.description is not None else None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not server:
        raise HTTPException(404, f"MCP server {server_id} does not exist")
    return {"status": "ok", "server": server.model_dump()}


@router.delete("/mcp-servers/{server_id}")
async def delete_mcp_server(server_id: str) -> dict[str, Any]:
    if not store.delete(server_id):
        raise HTTPException(404, f"MCP server {server_id} does not exist")
    return {"status": "ok", "deleted": server_id}


@router.post("/mcp-servers/{server_id}/test")
async def test_mcp_server(server_id: str) -> dict[str, str]:
    server = store.get(server_id)
    if not server:
        raise HTTPException(404, f"MCP server {server_id} does not exist")

    return await initialize_check(server)


@router.get("/mcp-servers/{server_id}/tools")
async def list_mcp_server_tools(server_id: str) -> dict[str, Any]:
    server = store.get(server_id)
    if not server:
        raise HTTPException(404, f"MCP server {server_id} does not exist")
    try:
        tools = await list_tools(server)
    except McpClientError as exc:
        raise HTTPException(502, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"MCP tools/list failed: {exc}") from exc
    return {"server_id": server_id, "total": len(tools), "items": tools}


@router.post("/mcp-servers/{server_id}/tools/{tool_name}/call")
async def call_mcp_server_tool(
    server_id: str,
    tool_name: str,
    body: McpToolCallRequest,
) -> dict[str, Any]:
    server = store.get(server_id)
    if not server:
        raise HTTPException(404, f"MCP server {server_id} does not exist")
    try:
        result = await call_tool(server, tool_name, body.arguments)
    except McpClientError as exc:
        raise HTTPException(502, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"MCP tools/call failed: {exc}") from exc
    return {"status": "ok", "server_id": server_id, "tool": tool_name, "result": result}
