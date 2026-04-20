"""Application configuration via environment variables."""
from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    model_api_url: str = ""
    model_api_key: str = ""
    model_name: str = "qwen-plus"

    # External services
    history_data_api_url: str = "http://localhost:8080/api/history"
    knowledge_graph_api_url: str = "http://localhost:8081/api/knowledge"

    # Server
    host: str = "0.0.0.0"
    port: int = 4444
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5873"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
