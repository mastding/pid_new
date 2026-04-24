"""Base interfaces for identification providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseIdentificationProvider(ABC):
    category = "identification"
    name: str

    @abstractmethod
    def identify(
        self,
        *,
        cleaned_df: pd.DataFrame,
        candidate_windows: list[dict[str, Any]],
        actual_dt: float,
        loop_type: str,
        quality_metrics: dict[str, Any] | None = None,
        force_model_types: list[str] | None = None,
        force_l_hint: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError
