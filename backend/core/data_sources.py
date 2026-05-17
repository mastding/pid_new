"""Runtime data-source configuration store."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


_CONFIG_PATH = Path(__file__).resolve().parents[1] / "var" / "config" / "data_sources.json"
_SECRET_MASK = "******"


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _default_config() -> dict[str, Any]:
    return {
        "items": [
            {
                "id": "history_upload_default",
                "source_name": "历史文件导入",
                "source_type": "history_upload",
                "enabled": True,
                "host": "",
                "port": None,
                "database": "",
                "username": "",
                "secret_present": False,
                "polling_interval_s": 0,
                "updated_at": _now_iso(),
            }
        ]
    }


def _sanitize(item: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(item)
    sanitized.pop("password", None)
    sanitized.pop("secret", None)
    if sanitized.get("secret_present"):
        sanitized["password"] = _SECRET_MASK
    return sanitized


class DataSourceConfigStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else _CONFIG_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return _default_config()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return _default_config()
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return _default_config()
        return {"items": [_sanitize(item) for item in items if isinstance(item, dict)]}

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        current_by_id = {
            str(item.get("id")): item
            for item in (self._load_raw().get("items") or [])
            if isinstance(item, dict) and item.get("id")
        }
        normalized: list[dict[str, Any]] = []
        for raw in payload.get("items") or []:
            if not isinstance(raw, dict):
                continue
            source_id = str(raw.get("id") or f"ds_{uuid.uuid4().hex[:8]}")
            previous = current_by_id.get(source_id, {})
            password = raw.get("password")
            secret_present = bool(previous.get("secret_present"))
            secret_value = previous.get("secret")
            if password and password != _SECRET_MASK:
                secret_present = True
                secret_value = str(password)
            item = {
                "id": source_id,
                "source_name": str(raw.get("source_name") or "未命名数据源"),
                "source_type": str(raw.get("source_type") or "history_upload"),
                "enabled": bool(raw.get("enabled", True)),
                "host": str(raw.get("host") or ""),
                "port": int(raw["port"]) if raw.get("port") not in {None, ""} else None,
                "database": str(raw.get("database") or ""),
                "username": str(raw.get("username") or ""),
                "secret": secret_value,
                "secret_present": secret_present,
                "polling_interval_s": int(raw.get("polling_interval_s") or 0),
                "updated_at": _now_iso(),
            }
            normalized.append(item)
        if not normalized:
            normalized = _default_config()["items"]
        self.path.write_text(json.dumps({"items": normalized}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"items": [_sanitize(item) for item in normalized]}

    def _load_raw(self) -> dict[str, Any]:
        if not self.path.exists():
            return _default_config()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return _default_config()
        return data if isinstance(data, dict) else _default_config()


data_source_config_store = DataSourceConfigStore()
