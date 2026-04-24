"""Base interfaces for dead-time providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDeadTimeProvider(ABC):
    category = "dead_time"
    name: str

    @abstractmethod
    def estimate(
        self,
        *,
        mv: np.ndarray,
        pv: np.ndarray,
        dt: float,
        loop_type: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError
