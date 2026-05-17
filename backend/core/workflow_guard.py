"""Guardrails for LLM-planned skill execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.skills.base import LoopContext
from core.skills.registry import registry


_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class GuardDecision:
    allowed: bool
    reason: str = ""
    risk_level: str = "low"
    unmet_preconditions: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "unmet_preconditions": self.unmet_preconditions or [],
        }


def _has_context_value(ctx: LoopContext, key: str) -> bool:
    if key == "cleaned_df":
        return ctx.cleaned_df is not None
    if key == "dt":
        return ctx.dt is not None
    if key == "candidate_windows":
        return bool(ctx.candidate_windows)
    if key == "selected_window_index":
        return ctx.selected_window_index is not None
    if key == "model":
        return bool(ctx.model)
    if key == "pid_params":
        return bool(ctx.pid_params)
    if key == "evaluation":
        return bool(ctx.evaluation)
    if key.startswith("data_profile."):
        cur: Any = ctx.data_profile
        for part in key.split(".")[1:]:
            if not isinstance(cur, dict) or part not in cur:
                return False
            cur = cur.get(part)
        return cur is not None
    return False


class WorkflowGuard:
    """Small deterministic guard used before executing LLM-selected skills."""

    def __init__(self, *, max_llm_risk_level: str = "medium") -> None:
        self.max_llm_risk_level = max_llm_risk_level

    def check(self, skill_name: str, ctx: LoopContext, *, initiated_by: str = "system") -> GuardDecision:
        try:
            meta = registry.metadata(skill_name)
        except KeyError as exc:
            return GuardDecision(False, str(exc), "unknown", [])

        risk = str(meta.get("risk_level") or "low")
        preconditions = [str(item) for item in meta.get("preconditions", [])]
        unmet = [item for item in preconditions if not _has_context_value(ctx, item)]
        if unmet:
            return GuardDecision(False, f"技能 {skill_name} 前置条件未满足", risk, unmet)

        if initiated_by == "llm" and _RISK_ORDER.get(risk, 99) > _RISK_ORDER.get(self.max_llm_risk_level, 1):
            return GuardDecision(False, f"LLM 不能直接越级调用 {risk} 风险技能 {skill_name}", risk, [])

        return GuardDecision(True, "allowed", risk, [])

    def check_action(
        self,
        action_name: str,
        *,
        risk_level: str,
        preconditions: dict[str, bool],
        initiated_by: str = "system",
    ) -> GuardDecision:
        """Guard non-skill workflow actions such as creating tuning tasks."""
        risk = str(risk_level or "low")
        unmet = [name for name, ok in preconditions.items() if not ok]
        if unmet:
            return GuardDecision(False, f"动作 {action_name} 前置条件未满足", risk, unmet)
        if initiated_by == "llm" and _RISK_ORDER.get(risk, 99) > _RISK_ORDER.get(self.max_llm_risk_level, 1):
            return GuardDecision(False, f"LLM 不能直接越级执行 {risk} 风险动作 {action_name}", risk, [])
        return GuardDecision(True, "allowed", risk, [])


workflow_guard = WorkflowGuard()
