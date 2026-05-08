"""MCP 服务配置存储。

管理已注册的 MCP（Model Context Protocol）服务，每条记录至少包含：
  - id           稳定 UUID，用于增删改查
  - name         服务名称（用户可读）
  - url          服务地址（HTTP/SSE/streamable-http transport 必填；stdio 可留空）
  - transport    "stdio" | "sse" | "streamable-http"
  - raw_json     通用 MCP JSON 配置原文，用于覆盖 / 补充顶层字段（command/args/env/headers 等）
  - enabled      是否启用
  - description  可选备注

线程安全：内存缓存 + JSON 文件双层；写操作加锁。文件路径：backend/var/config/mcp_servers.json
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

CONFIG_FILE = Path(__file__).resolve().parent.parent / "var" / "config" / "mcp_servers.json"

Transport = Literal["stdio", "sse", "streamable-http"]
SUPPORTED_TRANSPORTS: tuple[str, ...] = ("stdio", "sse", "streamable-http")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class McpServer(BaseModel):
    """单条 MCP 服务配置记录。"""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    url: str = ""
    transport: Transport = "sse"
    raw_json: str = ""  # 用户填的通用 MCP JSON 原文（不强制校验，便于直接复制粘贴）
    enabled: bool = True
    description: str = ""
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)

    @field_validator("transport")
    @classmethod
    def _check_transport(cls, v: str) -> str:
        if v not in SUPPORTED_TRANSPORTS:
            raise ValueError(f"transport 必须是 {SUPPORTED_TRANSPORTS} 之一")
        return v

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name 不能为空")
        return v


class McpConfigStore:
    """线程安全的 MCP 服务列表存储。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._servers: list[McpServer] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if CONFIG_FILE.is_file():
            try:
                raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                items = raw.get("servers") if isinstance(raw, dict) else None
                if isinstance(items, list):
                    self._servers = [McpServer(**item) for item in items if isinstance(item, dict)]
            except Exception:
                # 文件损坏不致命：当作空列表，下一次写入会覆盖。
                self._servers = []
        self._loaded = True

    def _persist(self) -> None:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "servers": [s.model_dump() for s in self._servers],
        }
        CONFIG_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 读 ─────────────────────────────────────────────────────────────────

    def list(self) -> list[McpServer]:
        with self._lock:
            self._ensure_loaded()
            return list(self._servers)

    def get(self, server_id: str) -> McpServer | None:
        with self._lock:
            self._ensure_loaded()
            return next((s for s in self._servers if s.id == server_id), None)

    # ── 写 ─────────────────────────────────────────────────────────────────

    def create(self, **fields: Any) -> McpServer:
        with self._lock:
            self._ensure_loaded()
            server = McpServer(**fields)
            # 名称冲突时拒绝（避免后续 LLM tool 调用歧义）
            if any(s.name == server.name for s in self._servers):
                raise ValueError(f"已存在名称为 {server.name!r} 的 MCP 服务")
            self._servers.append(server)
            self._persist()
            return server

    def update(self, server_id: str, **fields: Any) -> McpServer | None:
        with self._lock:
            self._ensure_loaded()
            idx = next((i for i, s in enumerate(self._servers) if s.id == server_id), -1)
            if idx < 0:
                return None
            existing = self._servers[idx].model_dump()
            for key in ("name", "url", "transport", "raw_json", "enabled", "description"):
                if key in fields and fields[key] is not None:
                    existing[key] = fields[key]
            existing["updated_at"] = _now_iso()
            new_server = McpServer(**existing)
            # 改名后检查冲突
            if any(s.name == new_server.name and s.id != server_id for s in self._servers):
                raise ValueError(f"已存在名称为 {new_server.name!r} 的 MCP 服务")
            self._servers[idx] = new_server
            self._persist()
            return new_server

    def delete(self, server_id: str) -> bool:
        with self._lock:
            self._ensure_loaded()
            before = len(self._servers)
            self._servers = [s for s in self._servers if s.id != server_id]
            if len(self._servers) == before:
                return False
            self._persist()
            return True


store = McpConfigStore()
