"""Runtime configuration for ontology MCP query text."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from threading import Lock
from typing import Any

from pydantic import BaseModel

CONFIG_FILE = Path(__file__).resolve().parent.parent / "var" / "config" / "ontology_query_config.json"

DEFAULT_ONTOLOGY_QUERY_TEMPLATE = """请查询 PID 回路 $loop_name 的本体知识，回路类型提示为 $loop_type。
请重点返回：
1. 本体中的实际回路位号或别名；
2. PV/CV、MV、SP、DV 的名称和物理含义；
3. 对象类型、阶次、正/反作用或过程增益方向；
4. 时间尺度、滤波时间、死区、典型动态先验；
5. 常见工况和扰动场景；
6. 用这些本体知识评审回路辨识窗口时，应该优先选择或避开的窗口特征。
请用简明中文输出，可以包含条目。"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class OntologyQueryConfig(BaseModel):
    query_template: str = DEFAULT_ONTOLOGY_QUERY_TEMPLATE
    updated_at: str = ""


def render_query_template(template: str, *, loop_name: str, loop_type: str) -> str:
    loop_hint = loop_name.strip() or "当前回路"
    type_hint = loop_type.strip() or "unknown"
    rendered = Template(template or DEFAULT_ONTOLOGY_QUERY_TEMPLATE).safe_substitute(
        loop_name=loop_hint,
        loop_type=type_hint,
    )
    return rendered.strip()


class OntologyQueryConfigStore:
    """Thread-safe JSON-backed ontology query configuration."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._config = self._with_timestamp(OntologyQueryConfig())
        self._loaded = False

    @staticmethod
    def _with_timestamp(config: OntologyQueryConfig) -> OntologyQueryConfig:
        data = config.model_dump()
        if not data.get("updated_at"):
            data["updated_at"] = _now_iso()
        return OntologyQueryConfig(**data)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if CONFIG_FILE.is_file():
            try:
                raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                self._config = self._with_timestamp(OntologyQueryConfig(**raw))
            except Exception:
                self._config = self._with_timestamp(OntologyQueryConfig())
        self._loaded = True

    def get(self) -> OntologyQueryConfig:
        with self._lock:
            self._ensure_loaded()
            return self._config

    def update(self, *, query_template: str | None = None) -> OntologyQueryConfig:
        with self._lock:
            self._ensure_loaded()
            existing: dict[str, Any] = self._config.model_dump()
            if query_template is not None:
                existing["query_template"] = query_template
            existing["updated_at"] = _now_iso()
            self._config = OntologyQueryConfig(**existing)
            self._save()
            return self._config

    def reset_defaults(self) -> OntologyQueryConfig:
        with self._lock:
            self._config = self._with_timestamp(OntologyQueryConfig(updated_at=_now_iso()))
            self._loaded = True
            self._save()
            return self._config

    def render(self, *, loop_name: str, loop_type: str) -> str:
        config = self.get()
        return render_query_template(config.query_template, loop_name=loop_name, loop_type=loop_type)

    def _save(self) -> None:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            self._config.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )


store = OntologyQueryConfigStore()
