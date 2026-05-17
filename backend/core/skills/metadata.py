"""Shared metadata for workflow skills."""
from __future__ import annotations

from typing import Literal, TypedDict


RiskLevel = Literal["low", "medium", "high", "critical"]


class SkillEffect(TypedDict, total=False):
    key: str
    description: str


class SkillMetadata(TypedDict, total=False):
    risk_level: RiskLevel
    preconditions: list[str]
    effects: list[SkillEffect]
    stage: str
    deterministic_gate: bool
