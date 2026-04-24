"""Shared infrastructure for core modules."""

from core.shared.provider_registry import ProviderRegistry, provider_registry, register_provider

__all__ = [
    "ProviderRegistry",
    "provider_registry",
    "register_provider",
]
