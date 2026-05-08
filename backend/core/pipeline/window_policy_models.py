"""Structured policy objects for window-candidate decisions."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


GainSign = Literal["positive", "negative", "unknown"]
AlgorithmPlanState = Literal["preferred", "available", "deprioritized", "disabled"]


class OntologyFacts(BaseModel):
    """Facts extracted from ontology/MCP context for a loop."""

    loop_id: str = ""
    source: Literal["mcp", "frontend", "default", "none"] = "none"
    confidence: float = 0.0
    process_direction: GainSign = "unknown"
    expected_dead_time_range_s: tuple[float, float] | None = None
    expected_time_constant_range_s: tuple[float, float] | None = None
    min_excitation_pct: float | None = None
    max_noise_ratio: float | None = None
    avoid_conditions: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    raw_answer: str | None = None


class WindowSelectionPolicy(BaseModel):
    """Policy consumed by future window algorithm providers and the UI."""

    loop_id: str = ""
    loop_type: str = ""
    policy_version: str = "phase1-default"
    confidence: float = 0.0
    preferred_algorithm_families: list[str] = Field(default_factory=list)
    deprioritized_algorithm_families: list[str] = Field(default_factory=list)
    disabled_algorithm_families: list[str] = Field(default_factory=list)
    algorithm_plan: list[dict[str, Any]] = Field(default_factory=list)
    min_mv_excitation: float | None = None
    min_sp_excitation: float | None = None
    max_mv_saturation_ratio: float | None = None
    max_pv_noise_ratio: float | None = None
    min_pv_response: float | None = None
    max_drift_ratio: float | None = None
    expected_dead_time_range_s: tuple[float, float] | None = None
    expected_time_constant_range_s: tuple[float, float] | None = None
    expected_gain_sign: GainSign = "unknown"
    min_window_points: int = 30
    min_window_duration_s: float = 0.0
    max_window_points: int | None = None
    pre_window_s: float | None = None
    post_window_s: float | None = None
    steady_scan_window_s: float | None = None
    steady_scan_step_s: float | None = None
    merge_gap_s: float | None = None
    max_candidates_per_family: int = 6
    allowed_operating_states: list[str] = Field(default_factory=list)
    avoid_operating_states: list[str] = Field(default_factory=list)
    scoring_weights: dict[str, float] = Field(default_factory=dict)
    hard_guards: list[dict[str, Any]] = Field(default_factory=list)
    soft_penalties: list[dict[str, Any]] = Field(default_factory=list)
    field_usage: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""
    ontology_facts: OntologyFacts = Field(default_factory=OntologyFacts)


class WindowCandidateDecision(BaseModel):
    """Phase-1 admission result for formal vs diagnostic identification."""

    selected_window_indices: list[int] = Field(default_factory=list)
    rejected_window_indices: list[int] = Field(default_factory=list)
    fallback_window_indices: list[int] = Field(default_factory=list)
    formal_identification_allowed: bool = True
    diagnostic_identification_allowed: bool = True
    stop_reason: str | None = None
    primary_reason: str = ""
    ontology_evidence: list[dict[str, Any]] = Field(default_factory=list)
    data_evidence: list[dict[str, Any]] = Field(default_factory=list)
    window_judgements: list[dict[str, Any]] = Field(default_factory=list)
    recommended_identification_plan: dict[str, Any] = Field(default_factory=dict)
    risk_flags: list[str] = Field(default_factory=list)
