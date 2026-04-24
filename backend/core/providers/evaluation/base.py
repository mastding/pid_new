"""Base interfaces for evaluation providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluationProvider(ABC):
    category = "evaluation"
    name: str

    @abstractmethod
    def evaluate(
        self,
        *,
        Kp: float,
        Ki: float,
        Kd: float,
        model_type: str,
        model_params: dict[str, Any] | None,
        K: float,
        T: float,
        L: float,
        dt: float,
        loop_type: str,
        confidence: float = 1.0,
        tuning_unreliable: bool = False,
        tuning_unreliable_reason: str = "",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class BaseSimulationProvider(ABC):
    category = "evaluation_simulation"
    name: str

    @abstractmethod
    def simulate(
        self,
        *,
        model_params: dict[str, Any],
        pid_params: dict[str, float],
        sp_initial: float,
        sp_final: float,
        n_steps: int,
        dt: float,
        loop_type: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class BasePerformanceScoringProvider(ABC):
    category = "evaluation_scoring"
    name: str

    @abstractmethod
    def score(self, *, simulation: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError


class BaseRealityCheckProvider(ABC):
    category = "evaluation_reality_check"
    name: str

    @abstractmethod
    def check(
        self,
        *,
        model_params: dict[str, Any],
        pid_params: dict[str, float],
        perf_score: float,
        confidence: float,
        loop_type: str,
        n_steps: int,
        dt: float,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError
