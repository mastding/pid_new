"""Process model data structures."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    FO = "FO"
    FOPDT = "FOPDT"
    SOPDT = "SOPDT"
    IPDT = "IPDT"


class ProcessModel(BaseModel):
    """Identified process model parameters."""

    model_type: ModelType = ModelType.FOPDT
    K: float = 0.0
    T: float = 0.0
    T1: float = 0.0
    T2: float = 0.0
    L: float = 0.0

    # Fit quality
    r2_score: float = 0.0
    normalized_rmse: float = 0.0
    raw_rmse: float = 0.0
    success: bool = False

    def to_tuning_params(self, dt: float, n_points: int) -> dict[str, float]:
        """Convert to K/T/L dict for tuning rules."""
        if self.model_type == ModelType.FO:
            return {"K": self.K, "T": max(self.T, dt), "L": 0.0}
        if self.model_type == ModelType.IPDT:
            surrogate_t = max(n_points * dt / 4.0, dt * 20.0)
            return {"K": max(abs(self.K), 1e-3), "T": surrogate_t, "L": max(self.L, dt)}
        if self.model_type == ModelType.SOPDT:
            return {"K": self.K, "T": max(self.T1 + self.T2, dt), "L": max(self.L, 0.0)}
        return {"K": self.K, "T": max(self.T, dt), "L": max(self.L, 0.0)}


class ModelConfidence(BaseModel):
    """Model identification confidence assessment."""

    confidence: float = Field(0.0, ge=0.0, le=1.0)
    quality: str = "poor"
    recommendation: str = ""
    r2_score: float = 0.0
    rmse_score: float = 0.0


class IdentificationResult(BaseModel):
    """Complete result from system identification."""

    model: ProcessModel = Field(default_factory=ProcessModel)
    confidence: ModelConfidence = Field(default_factory=ModelConfidence)
    window_source: str = ""
    fit_preview: dict[str, Any] = Field(default_factory=dict)
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    selection_reason: str = ""
