"""运行时模型配置存储。

支持运行时读写 + JSON 文件持久化。
LLM 调用方不应直接读 settings.xxx，而应通过本模块的 store.get() 获取当前配置。
这样用户通过管理页面修改配置后无需重启服务即可生效。
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel

CONFIG_FILE = Path(__file__).resolve().parent.parent / "var" / "config" / "model.json"


class ModelConfig(BaseModel):
    model_api_url: str = ""
    model_api_key: str = ""
    model_name: str = "qwen-plus"

    def masked(self) -> dict[str, str]:
        key = self.model_api_key
        if len(key) > 8:
            masked_key = key[:4] + "****" + key[-4:]
        elif key:
            masked_key = key[:2] + "****"
        else:
            masked_key = ""
        return {
            "model_api_url": self.model_api_url,
            "model_api_key": masked_key,
            "model_name": self.model_name,
        }

    def is_configured(self) -> bool:
        return bool(self.model_api_url and self.model_api_key)


class ModelConfigStore:
    """线程安全的内存 + 文件双层模型配置存储。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._config: ModelConfig = ModelConfig()
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from config import settings
        if CONFIG_FILE.is_file():
            try:
                raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                self._config = ModelConfig(**raw)
            except Exception:
                self._config = ModelConfig(
                    model_api_url=settings.model_api_url,
                    model_api_key=settings.model_api_key,
                    model_name=settings.model_name,
                )
        else:
            self._config = ModelConfig(
                model_api_url=settings.model_api_url,
                model_api_key=settings.model_api_key,
                model_name=settings.model_name,
            )
        self._loaded = True

    def get(self) -> ModelConfig:
        with self._lock:
            self._ensure_loaded()
            return self._config

    def update(self, **kwargs: Any) -> ModelConfig:
        """更新配置并落盘。只更新传入的非 None 字段。"""
        with self._lock:
            self._ensure_loaded()
            existing = self._config.model_dump()
            for k in ("model_api_url", "model_api_key", "model_name"):
                if k in kwargs and kwargs[k] is not None:
                    existing[k] = kwargs[k]
            self._config = ModelConfig(**existing)
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_FILE.write_text(
                self._config.model_dump_json(indent=2, exclude_none=True),
                encoding="utf-8",
            )
        return self._config


store = ModelConfigStore()
