"""Base interfaces for tuning providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTuningProvider(ABC):
    category = "tuning"
    name: str

    @abstractmethod
    def tune(
        self,
        *,
        K: float,
        T: float,
        L: float,
        dt: float,
        loop_type: str,
        model_type: str,
        model_params: dict[str, Any] | None = None,
        confidence: float = 1.0,
        nrmse: float = 0.0,
        r2: float = 1.0,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class BaseSingleStrategyTuningProvider(BaseTuningProvider):
    """Base class for one concrete tuning strategy such as IMC or Lambda."""

    strategy: str

    def tune(
        self,
        *,
        K: float,
        T: float,
        L: float,
        dt: float,
        loop_type: str,
        model_type: str,
        model_params: dict[str, Any] | None = None,
        confidence: float = 1.0,
        nrmse: float = 0.0,
        r2: float = 1.0,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError
