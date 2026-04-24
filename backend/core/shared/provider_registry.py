"""Provider registry for pluggable algorithm implementations."""
from __future__ import annotations

from typing import Any, Callable


class ProviderRegistry:
    """In-memory registry for providers grouped by capability category."""

    def __init__(self) -> None:
        self._providers: dict[str, dict[str, Any]] = {}

    def register(self, category: str, provider: Any) -> Any:
        category_key = (category or "").strip()
        if not category_key:
            raise ValueError("provider category must not be empty")
        name = getattr(provider, "name", "")
        if not name:
            raise ValueError("provider must define a non-empty name")
        self._providers.setdefault(category_key, {})
        if name in self._providers[category_key]:
            raise ValueError(f"provider already registered: {category_key}/{name}")
        self._providers[category_key][name] = provider
        return provider

    def get(self, category: str, name: str) -> Any | None:
        return self._providers.get(category, {}).get(name)

    def require(self, category: str, name: str) -> Any:
        provider = self.get(category, name)
        if provider is None:
            raise KeyError(f"unknown provider: {category}/{name}")
        return provider

    def names(self, category: str) -> list[str]:
        return sorted(self._providers.get(category, {}).keys())

    def categories(self) -> list[str]:
        return sorted(self._providers.keys())


provider_registry = ProviderRegistry()


def register_provider(category: str) -> Callable[[type], type]:
    """Decorator to register a provider class by category."""

    def _decorator(provider_cls: type) -> type:
        provider_registry.register(category, provider_cls())
        return provider_cls

    return _decorator
