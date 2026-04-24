"""Base interfaces for window detection providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseWindowDetectionProvider(ABC):
    category = "window_detection"
    name: str

    @abstractmethod
    def detect(
        self,
        *,
        df: pd.DataFrame,
        dt: float,
        loop_type: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError
