"""Persistent chat sessions for the monitoring dialogue mode."""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent / "var" / "assistant_sessions"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _session_dir(session_id: str) -> Path:
    return ROOT / session_id


def _session_path(session_id: str) -> Path:
    return _session_dir(session_id) / "session.json"


def _read_session(session_id: str) -> dict[str, Any] | None:
    path = _session_path(session_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_session(session: dict[str, Any]) -> dict[str, Any]:
    session_id = str(session["id"])
    path = _session_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return session


def create_session(*, title: str | None = None, loop_id: str | None = None) -> dict[str, Any]:
    now = _now_iso()
    session = {
        "id": uuid.uuid4().hex[:12],
        "kind": "assistant",
        "title": title or "新对话",
        "loop_id": loop_id,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    return _write_session(session)


def list_sessions(limit: int = 100) -> list[dict[str, Any]]:
    if not ROOT.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in ROOT.glob("*/session.json"):
        try:
            session = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        messages = session.get("messages") if isinstance(session.get("messages"), list) else []
        items.append({
            "id": session.get("id"),
            "kind": "assistant",
            "title": session.get("title") or "未命名对话",
            "loop_id": session.get("loop_id"),
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at"),
            "message_count": len(messages),
        })
    items.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return items[:limit]


def get_session(session_id: str) -> dict[str, Any] | None:
    return _read_session(session_id)


def delete_session(session_id: str) -> bool:
    directory = _session_dir(session_id)
    if not directory.is_dir():
        return False
    shutil.rmtree(directory, ignore_errors=True)
    return True


def update_session(session_id: str, *, title: str | None = None, loop_id: str | None = None) -> dict[str, Any] | None:
    session = _read_session(session_id)
    if not session:
        return None
    if title is not None:
        session["title"] = title.strip() or session.get("title") or "未命名对话"
    if loop_id is not None:
        session["loop_id"] = loop_id or None
    session["updated_at"] = _now_iso()
    return _write_session(session)


def append_message(
    session_id: str,
    *,
    role: str,
    content: str,
    reasoning_summary: str | None = None,
    raw_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    session = _read_session(session_id)
    if not session:
        return None
    message = {
        "id": uuid.uuid4().hex[:12],
        "role": role,
        "content": content,
        "reasoning_summary": reasoning_summary or "",
        "raw_events": raw_events or [],
        "created_at": _now_iso(),
    }
    messages = session.setdefault("messages", [])
    messages.append(message)
    if role == "user" and (not session.get("title") or session.get("title") == "新对话"):
        session["title"] = content.strip()[:32] or "新对话"
    session["updated_at"] = _now_iso()
    _write_session(session)
    return message
